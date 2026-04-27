import json
import math
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.integrations import use_kernel_forward_from_hub


MODEL_NAME = "Qwen3-1.7B"
QWEN_CONFIG = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)


@use_kernel_forward_from_hub("RMSNorm")
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@dataclass
class EngramConfig:
    layer_ids: List[int] = field(default_factory=lambda: [2])
    backbone_hidden_size: int = QWEN_CONFIG.hidden_size
    item_hidden_size: int = 256


class Engram(nn.Module):
    """
    推理专用的最小 Engram:

    - 不做 tokenizer 压缩 / ngram hash / multi-head embedding
    - 直接接收按 token 对齐好的 `sasrec_token_states`
    - 只保留 key/value 投影和 gate 计算
    - 为 beam search 保留 cache/reset/reorder 接口，但当前实现为无状态 no-op
    """

    def __init__(self, layer_id, config: EngramConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.key_proj = nn.Linear(self.config.item_hidden_size, self.config.backbone_hidden_size, bias=False)
        self.value_proj = nn.Linear(self.config.item_hidden_size, self.config.backbone_hidden_size, bias=False)
        self.norm1 = Qwen3RMSNorm(self.config.backbone_hidden_size)
        self.norm2 = Qwen3RMSNorm(self.config.backbone_hidden_size)
        self._init_engram()

    def _init_engram(self):
        for layer in [self.key_proj, self.value_proj]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for norm in [self.norm1, self.norm2]:
            nn.init.ones_(norm.weight)

    def reset_inference_cache(self):
        return None

    def reorder_inference_cache(self, beam_idx: torch.LongTensor):
        del beam_idx
        return None

    def _prepare_item_states(
        self,
        hidden_states: torch.Tensor,
        item_attention_mask: torch.Tensor | None = None,
        sasrec_token_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        if sasrec_token_states is None:
            item_states = torch.zeros(
                batch_size,
                seq_len,
                self.config.item_hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            item_states = sasrec_token_states.to(device=hidden_states.device)
            if item_states.shape[:2] != (batch_size, seq_len):
                raise ValueError(
                    f"sasrec_token_states shape {tuple(item_states.shape)} "
                    f"must start with {(batch_size, seq_len)}"
                )
            if item_states.shape[-1] != self.config.item_hidden_size:
                raise ValueError(
                    f"sasrec_token_states hidden size {item_states.shape[-1]} "
                    f"!= config.item_hidden_size {self.config.item_hidden_size}"
                )

        if item_attention_mask is None:
            return item_states

        mask = item_attention_mask.unsqueeze(-1).to(device=hidden_states.device, dtype=item_states.dtype)
        return item_states * mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        item_attention_mask: torch.Tensor | None = None,
        is_first_iteration: bool = False,
        sasrec_token_states: torch.Tensor | None = None,
    ):
        del input_ids, use_cache, is_first_iteration

        item_states = self._prepare_item_states(
            hidden_states=hidden_states,
            item_attention_mask=item_attention_mask,
            sasrec_token_states=sasrec_token_states,
        )

        key = self.key_proj(item_states)
        normed_key = self.norm1(key)

        query = hidden_states
        normed_query = self.norm2(query)
        gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.config.backbone_hidden_size)
        gate = gate.sigmoid().unsqueeze(-1)

        value = gate * self.value_proj(item_states)

        if item_attention_mask is not None:
            mask = item_attention_mask.unsqueeze(-1).to(value.device, dtype=value.dtype)
            value = value * mask
            gate = gate * mask

        return value, gate

    def save_all_params(self, save_path: str, initial_params: dict = None, warn_threshold: float = 1e-6):
        del initial_params, warn_threshold
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        params = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dtype == torch.bfloat16:
                param_np = param.detach().cpu().float().numpy().copy()
            else:
                param_np = param.detach().cpu().numpy().copy()
            params[name] = param_np

        if not params:
            return None

        if save_path.endswith(".npy"):
            np.save(save_path, params)
        elif save_path.endswith(".json"):
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({k: v.tolist() for k, v in params.items()}, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError("仅支持保存为 .npy 或 .json 文件")

        return params

    def load_all_params(self, load_path: str, min_norm_threshold: float = 1e-6):
        del min_norm_threshold
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"参数文件不存在：{load_path}")

        if load_path.endswith(".npy"):
            params = np.load(load_path, allow_pickle=True).item()
        elif load_path.endswith(".json"):
            with open(load_path, "r", encoding="utf-8") as f:
                params = json.load(f)
            params = {name: np.array(value, dtype=np.float32) for name, value in params.items()}
        else:
            raise ValueError("仅支持加载 .npy 或 .json 文件")

        loaded_count = 0
        invalid_params = []

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name not in params:
                    continue
                param_np = params[name]
                param_tensor = torch.from_numpy(param_np).to(device=param.device, dtype=param.dtype)
                param.copy_(param_tensor)
                loaded_count += 1

        return loaded_count, invalid_params


__all__ = ["EngramConfig", "Engram"]
