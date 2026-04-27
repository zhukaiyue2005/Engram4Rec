import json
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LogitLensAnalyzer:
    """Compute layer-to-final distribution distance on token positions."""

    def __init__(self, vocab_size: int, eps: float = 1e-12):
        self.vocab_size = vocab_size
        self.eps = eps

    def _get_log_probs_and_probs(self, logits: torch.Tensor):
        log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
        probs = log_probs.exp().clamp_min(self.eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return log_probs, probs

    def _compute_exact_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """KL(P_layer || P_final)."""
        p_log_probs, p_probs = self._get_log_probs_and_probs(p_logits)
        q_log_probs, _ = self._get_log_probs_and_probs(q_logits)
        kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
        return kl.clamp_min(0.0)

    def _compute_reverse_kl(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        """KL(P_final || P_layer)."""
        p_log_probs, _ = self._get_log_probs_and_probs(p_logits)
        q_log_probs, q_probs = self._get_log_probs_and_probs(q_logits)
        kl = (q_probs * (q_log_probs - p_log_probs)).sum(dim=-1)
        return kl.clamp_min(0.0)

    def _compute_exact_js(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
        p_log_probs, p_probs = self._get_log_probs_and_probs(p_logits)
        q_log_probs, q_probs = self._get_log_probs_and_probs(q_logits)
        m_probs = 0.5 * (p_probs + q_probs)
        m_probs = m_probs.clamp_min(self.eps)
        m_probs = m_probs / m_probs.sum(dim=-1, keepdim=True)
        m_log_probs = m_probs.log()
        kl_p_m = (p_probs * (p_log_probs - m_log_probs)).sum(dim=-1)
        kl_q_m = (q_probs * (q_log_probs - m_log_probs)).sum(dim=-1)
        return (0.5 * (kl_p_m + kl_q_m)).clamp_min(0.0)

    def _compute_monte_carlo_kl(
        self, p_logits: torch.Tensor, q_logits: torch.Tensor, num_samples: int = 10000
    ) -> torch.Tensor:
        batch_size, seq_len, vocab_size = p_logits.shape
        p_log_probs, p_probs = self._get_log_probs_and_probs(p_logits)
        q_log_probs, _ = self._get_log_probs_and_probs(q_logits)
        p_probs_flat = p_probs.view(-1, vocab_size)
        p_log_probs_flat = p_log_probs.view(-1, vocab_size)
        q_log_probs_flat = q_log_probs.view(-1, vocab_size)
        indices = torch.multinomial(p_probs_flat, num_samples, replacement=True)
        sampled_log_p = torch.gather(p_log_probs_flat, 1, indices)
        sampled_log_q = torch.gather(q_log_probs_flat, 1, indices)
        kl = (sampled_log_p - sampled_log_q).mean(dim=1)
        return kl.view(batch_size, seq_len).clamp_min(0.0)

    def _compute_importance_sampling_kl(
        self, p_logits: torch.Tensor, q_logits: torch.Tensor, num_samples: int = 10000
    ) -> torch.Tensor:
        batch_size, seq_len, vocab_size = p_logits.shape
        p_log_probs, p_probs = self._get_log_probs_and_probs(p_logits)
        q_log_probs, q_probs = self._get_log_probs_and_probs(q_logits)

        proposal_probs = 0.5 * (p_probs + q_probs)
        proposal_probs = proposal_probs.clamp_min(self.eps)
        proposal_probs = proposal_probs / proposal_probs.sum(dim=-1, keepdim=True)
        proposal_log_probs = proposal_probs.log()

        proposal_flat = proposal_probs.view(-1, vocab_size)
        p_probs_flat = p_probs.view(-1, vocab_size)
        p_log_probs_flat = p_log_probs.view(-1, vocab_size)
        q_log_probs_flat = q_log_probs.view(-1, vocab_size)
        proposal_log_flat = proposal_log_probs.view(-1, vocab_size)

        indices = torch.multinomial(proposal_flat, num_samples, replacement=True)
        sampled_p = torch.gather(p_probs_flat, 1, indices)
        sampled_log_p = torch.gather(p_log_probs_flat, 1, indices)
        sampled_log_q = torch.gather(q_log_probs_flat, 1, indices)
        sampled_log_m = torch.gather(proposal_log_flat, 1, indices)

        importance_weight = sampled_p / sampled_log_m.exp().clamp_min(self.eps)
        kl = (importance_weight * (sampled_log_p - sampled_log_q)).mean(dim=1)
        return kl.view(batch_size, seq_len).clamp_min(0.0)

    def compute_layer_distances(
        self,
        all_hidden_states: List[torch.Tensor],
        lm_head: nn.Module,
        last_hidden_state: Optional[torch.Tensor] = None,
        final_norm: Optional[nn.Module] = None,
        method: str = "exact_kl",
        mc_samples: int = 10000,
        position_chunk_size: int = 32,
        position_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        if last_hidden_state is None:
            last_hidden_state = all_hidden_states[-1]
            if final_norm is not None:
                last_hidden_state = final_norm(last_hidden_state)

        layer_values = {}
        batch_size, seq_len, hidden_size = last_hidden_state.shape
        if position_mask is not None:
            position_mask = position_mask.to(device=last_hidden_state.device, dtype=torch.bool)
            if position_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_mask shape {tuple(position_mask.shape)} does not match hidden states "
                    f"shape {(batch_size, seq_len)}"
                )
            flat_mask = position_mask.reshape(batch_size * seq_len)
            last_hidden_flat = last_hidden_state.reshape(batch_size * seq_len, hidden_size)[flat_mask]
        else:
            flat_mask = None
            last_hidden_flat = last_hidden_state.reshape(batch_size * seq_len, hidden_size)

        for layer_idx, hidden_state in enumerate(all_hidden_states):
            with torch.no_grad():
                if final_norm is not None:
                    hidden_state = final_norm(hidden_state)
                hidden_flat = hidden_state.reshape(batch_size * seq_len, hidden_size)
                if flat_mask is not None:
                    hidden_flat = hidden_flat[flat_mask]
                metric_chunks = []
                n_positions = hidden_flat.shape[0]
                for start in range(0, n_positions, position_chunk_size):
                    end = min(start + position_chunk_size, n_positions)
                    layer_logits = lm_head(hidden_flat[start:end])
                    last_logits = lm_head(last_hidden_flat[start:end])
                    if method == "exact_kl":
                        metric_chunk = self._compute_exact_kl(layer_logits, last_logits)
                    elif method == "reverse_kl":
                        metric_chunk = self._compute_reverse_kl(layer_logits, last_logits)
                    elif method == "exact_js":
                        metric_chunk = self._compute_exact_js(layer_logits, last_logits)
                    elif method == "monte_carlo":
                        metric_chunk = self._compute_monte_carlo_kl(
                            layer_logits.unsqueeze(1), last_logits.unsqueeze(1), num_samples=mc_samples
                        ).squeeze(1)
                    elif method == "importance_sampling":
                        metric_chunk = self._compute_importance_sampling_kl(
                            layer_logits.unsqueeze(1), last_logits.unsqueeze(1), num_samples=mc_samples
                        ).squeeze(1)
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    metric_chunks.append(metric_chunk)
                metric = torch.cat(metric_chunks, dim=0)
                if flat_mask is None:
                    metric = metric.view(batch_size, seq_len)
            layer_values[layer_idx] = metric

        return {"metric_per_layer": layer_values, "n_layers": len(all_hidden_states)}


def _resolve_lm_head_and_norm(model, resume_from_checkpoint: str = ""):
    if resume_from_checkpoint:
        lora_model = model.base_model
        qwen3_causal_lm = lora_model.model
        lm_head = getattr(qwen3_causal_lm, "lm_head", None)
        final_norm = None
        if hasattr(qwen3_causal_lm, "model") and hasattr(qwen3_causal_lm.model, "norm"):
            final_norm = qwen3_causal_lm.model.norm
        return lm_head, final_norm

    lm_head = getattr(model, "lm_head", None)
    final_norm = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    return lm_head, final_norm


def evaluate(
    model,
    tokenizer,
    dataloader,
    max_length: int = 2048,
    method: str = "exact_kl",
    resume_from_checkpoint: str = "",
    mc_samples: int = 10000,
    position_chunk_size: int = 32,
):
    lm_head, final_norm = _resolve_lm_head_and_norm(model, resume_from_checkpoint=resume_from_checkpoint)
    analyzer = LogitLensAnalyzer(vocab_size=model.config.vocab_size, eps=1e-12)

    metric_layer_values: Dict[int, List[float]] = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="KL evaluate"):
            sentences = batch["sentences"]
            prompt_lens = batch["prompt_lens"]

            inputs = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states[1:]
            batch_size, seq_len = inputs["input_ids"].shape
            position_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=model.device)
            for i in range(len(sentences)):
                tokenized_length = int((inputs["attention_mask"][i] == 1).sum().item())
                sample_prompt_len = min(prompt_lens[i], tokenized_length)
                target_start = seq_len - tokenized_length + sample_prompt_len
                target_end = seq_len
                if target_start < target_end:
                    position_mask[i, target_start:target_end] = True

            if not position_mask.any():
                continue

            metric_result = analyzer.compute_layer_distances(
                all_hidden_states=hidden_states,
                final_norm=final_norm,
                lm_head=lm_head,
                method=method,
                mc_samples=mc_samples,
                position_chunk_size=position_chunk_size,
                position_mask=position_mask,
            )

            for layer_idx, layer_metric in metric_result["metric_per_layer"].items():
                if layer_idx not in metric_layer_values:
                    metric_layer_values[layer_idx] = []
                if layer_metric.numel() > 0:
                    metric_layer_values[layer_idx].extend(layer_metric.tolist())

    metric_layer_averages = {
        layer_idx: sum(values) / len(values) for layer_idx, values in metric_layer_values.items() if values
    }
    return metric_layer_averages


def save_result_json(result: Dict[int, float], save_path: str, metadata: Optional[dict] = None):
    payload = {
        "layer_metric_averages": {str(k): v for k, v in sorted(result.items())},
    }
    if metadata is not None:
        payload["metadata"] = metadata
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
