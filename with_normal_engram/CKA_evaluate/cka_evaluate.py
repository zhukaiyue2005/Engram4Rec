import json
import os
import gc
import sys
from typing import List

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen3ForCausalLM as Qwen3ForCausalLMWithoutEngram

from cka_evaluate_batch import (
    compute_cka_similarity_matrix,
    compute_soft_alignment_index,
    extract_final_token_hidden_states,
    plot_cka_results,
)

os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_DIR), "data")
DEFAULT_TEST_FILE = os.path.join(DATA_ROOT, "Industrial_and_Scientific_dataset", "valid.jsonl")
LOCAL_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_cache")
os.environ["HF_HOME"] = LOCAL_HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(LOCAL_HF_HOME, "datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from huggingface_hub.utils._validators import validate_repo_id


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__

ENGRAM_DIR = os.path.join(PROJECT_DIR, "Engram_Insert_code")
sys.path.append(ENGRAM_DIR)

from engram_demo_v1 import Engram, EngramConfig
from modeling_qwen3 import Qwen3ForCausalLM

try:
    import fire
except ModuleNotFoundError:
    fire = None


def _parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
    values = []
    if isinstance(arg_value, (tuple, list)):
        for x in arg_value:
            if isinstance(x, str):
                values.extend([p.strip() for p in x.split(",") if p.strip()])
            elif x is not None:
                values.append(x)
    elif isinstance(arg_value, str):
        values = [p.strip() for p in arg_value.split(",") if p.strip()]
    elif arg_value is not None:
        values = [arg_value]

    out = []
    for x in values:
        try:
            out.append(int(x))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{arg_name} 中包含非法整数值: {x!r}, 原始输入: {arg_value!r}") from exc
    return out


def _normalize_prompt(row: dict) -> str:
    prompt_col = row.get("prompt", "")
    if isinstance(prompt_col, list):
        prompt = ""
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                prompt = item.get("content", "")
                break
        if prompt:
            return prompt
        return str(prompt_col)
    return str(prompt_col)


def _normalize_target_item_title(row: dict) -> str:
    title = row.get("target_item_title")
    if title:
        return str(title).strip()

    completion = str(row.get("completion", "")).strip()
    completion = completion.replace("<|im_end|>", "").strip()
    return completion.strip().strip('"').strip()


def _last_content_token_offset(tokenizer, token_ids: List[int], target_text: str) -> int:
    quote_only_tokens = {'"', "'", "“", "”", "‘", "’", "`", "``", "''"}
    for offset in range(len(token_ids) - 1, -1, -1):
        token_text = tokenizer.decode(
            [token_ids[offset]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        if token_text and token_text not in quote_only_tokens:
            return offset
    raise ValueError(f"target item title has no content token: {target_text!r}")


def _process_one_sample(row: dict, tokenizer, cutoff_len: int, hidden_state_target: str) -> dict:
    prompt = _normalize_prompt(row)

    target_token_position = None
    if hidden_state_target == "target_item_id":
        if "target_item_id" not in row:
            raise KeyError("hidden_state_target=target_item_id requires target_item_id in each JSON row")

        prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer.encode(str(row["target_item_id"]), add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"target_item_id produced no tokens: {row.get('target_item_id')!r}")

        input_ids = prompt_ids + target_ids
        target_token_position = len(input_ids) - 1
        if len(input_ids) > cutoff_len:
            overflow = len(input_ids) - cutoff_len
            input_ids = input_ids[overflow:]
            target_token_position -= overflow
        if target_token_position < 0:
            raise ValueError(f"target_item_id token was truncated: {row.get('target_item_id')!r}")
    elif hidden_state_target == "target_item_title":
        target_title = _normalize_target_item_title(row)
        if not target_title:
            raise ValueError("hidden_state_target=target_item_title requires target_item_title or completion")

        prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
        open_quote_ids = tokenizer.encode('"', add_special_tokens=False)
        target_ids = tokenizer.encode(target_title, add_special_tokens=False)
        close_quote_ids = tokenizer.encode('"\n', add_special_tokens=False)
        if not target_ids:
            raise ValueError(f"target item title produced no tokens: {target_title!r}")

        input_ids = prompt_ids + open_quote_ids + target_ids + close_quote_ids
        target_token_offset = _last_content_token_offset(tokenizer, target_ids, target_title)
        target_token_position = len(prompt_ids) + len(open_quote_ids) + target_token_offset
        if len(input_ids) > cutoff_len:
            overflow = len(input_ids) - cutoff_len
            input_ids = input_ids[overflow:]
            target_token_position -= overflow
        if target_token_position < 0:
            raise ValueError(f"target item title token was truncated: {target_title!r}")
    elif hidden_state_target == "sequence_last":
        input_ids = tokenizer(prompt, truncation=True, max_length=cutoff_len, add_special_tokens=True)["input_ids"]
    else:
        raise ValueError(f"Unsupported hidden_state_target: {hidden_state_target!r}")

    return {
        "input_ids": input_ids,
        "target_token_position": target_token_position,
        "prompt": prompt,
        "raw_row": row,
    }


def _left_pad_collate(samples: List[dict], pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in samples)
    input_ids = []
    attention_mask = []
    valid_lengths = []
    target_token_positions = []
    prompts = []
    raw_rows = []

    for x in samples:
        seq = x["input_ids"]
        pad_len = max_len - len(seq)
        input_ids.append([pad_token_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        valid_lengths.append(len(seq))
        target_token_position = x.get("target_token_position")
        if target_token_position is None:
            target_token_positions.append(-1)
        else:
            target_token_positions.append(pad_len + int(target_token_position))
        prompts.append(x["prompt"])
        raw_rows.append(x["raw_row"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "valid_lengths": valid_lengths,
        "target_token_positions": torch.tensor(target_token_positions, dtype=torch.long),
        "prompts": prompts,
        "raw_rows": raw_rows,
    }


def _build_dataloader(
    tokenizer,
    test_file: str,
    cutoff_len: int,
    batch_size: int,
    max_samples: int = 0,
    hidden_state_target: str = "target_item_id",
):
    dataset = load_dataset(
        "json",
        data_files={"test": test_file},
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    )["test"]
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[Info] CKA 使用样本数: {len(dataset)}", flush=True)
    processed = [_process_one_sample(row, tokenizer, cutoff_len, hidden_state_target) for row in dataset]
    return DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: _left_pad_collate(samples, tokenizer.pad_token_id),
    )


def _normalize_load_device(load_device: str) -> str:
    normalized = (load_device or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"load_device 只支持 auto/cuda/cpu，收到: {load_device!r}")
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return normalized


def _build_model_load_kwargs(load_device: str):
    resolved = _normalize_load_device(load_device)
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if resolved == "cpu":
        return resolved, {
            **common_kwargs,
            "device_map": {"": "cpu"},
            "torch_dtype": torch.float32,
        }

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    device_index = Accelerator().process_index
    return resolved, {
        **common_kwargs,
        "quantization_config": bnb_config,
        "device_map": {"": device_index},
    }


def _load_engram_model(
    base_model: str,
    resume_from_checkpoint: str,
    engram_layer_ids,
    engram_float32: bool,
    load_device: str,
):
    resolved_device, load_kwargs = _build_model_load_kwargs(load_device)
    try:
        model = Qwen3ForCausalLM.from_pretrained(base_model, **load_kwargs)
    except torch.OutOfMemoryError:
        if resolved_device != "cpu":
            print("[Warn] GPU 加载 Engram 模型 OOM，自动回退到 CPU 加载。", flush=True)
            resolved_device, load_kwargs = _build_model_load_kwargs("cpu")
            model = Qwen3ForCausalLM.from_pretrained(base_model, **load_kwargs)
        else:
            raise
    if resolved_device != "cpu":
        model = prepare_model_for_kbit_training(model)

    engram_config = EngramConfig()
    layer_ids = _parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if not layer_ids:
        raise ValueError("engram_layer_ids 不能为空")
    engram_config.layer_ids = layer_ids
    model.attach_engram(engram_config)
    model._reset_engram_parameters()

    def _load_engram_params(target_model, checkpoint_dir):
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            print(f"[Warn] 未找到 Engram 参数目录: {engram_params_dir}", flush=True)
            return
        unwrapped_model = target_model.module if hasattr(target_model, "module") else target_model
        loaded = 0
        for _, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                param_path = os.path.join(engram_params_dir, f"engram_layer_{layer_id}.npy")
                if os.path.exists(param_path):
                    module.load_all_params(param_path)
                    loaded += 1
                else:
                    print(f"[Warn] Engram 层 {layer_id} 参数缺失: {param_path}", flush=True)
        print(f"[Info] Engram 参数加载层数: {loaded}", flush=True)

    def _convert_engram_params_to_float32(target_model):
        for name, param in target_model.named_parameters():
            if "engram" in name.lower():
                param.data = param.data.to(torch.float32)

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        if engram_float32:
            _convert_engram_params_to_float32(model)
        _load_engram_params(model, resume_from_checkpoint)

    model.eval()
    return model


def _load_baseline_model(base_model: str, baseline_checkpoint: str = "", load_device: str = "auto"):
    resolved_device, load_kwargs = _build_model_load_kwargs(load_device)
    try:
        model = Qwen3ForCausalLMWithoutEngram.from_pretrained(base_model, **load_kwargs)
    except torch.OutOfMemoryError:
        if resolved_device != "cpu":
            print("[Warn] GPU 加载 baseline 模型 OOM，自动回退到 CPU 加载。", flush=True)
            resolved_device, load_kwargs = _build_model_load_kwargs("cpu")
            model = Qwen3ForCausalLMWithoutEngram.from_pretrained(base_model, **load_kwargs)
        else:
            raise
    if resolved_device != "cpu":
        model = prepare_model_for_kbit_training(model)

    if baseline_checkpoint:
        if not os.path.exists(baseline_checkpoint):
            raise FileNotFoundError(f"baseline_checkpoint 不存在: {baseline_checkpoint}")
        model = PeftModel.from_pretrained(model, baseline_checkpoint)

    model.eval()
    return model


def _evaluate_sequential(
    base_model: str,
    resume_from_checkpoint_with_engram: str,
    resume_from_checkpoint_without_engram: str,
    engram_layer_ids: str,
    engram_float32: bool,
    dataloader,
    batch_size: int,
    k: int,
    load_device: str,
):
    print("=" * 60)
    print("Step 1: Loading WITH Engram model and extracting hidden states")
    model_with_engram = _load_engram_model(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint_with_engram,
        engram_layer_ids=engram_layer_ids,
        engram_float32=engram_float32,
        load_device=load_device,
    )
    device = str(model_with_engram.device)
    hidden_states_with_engram = extract_final_token_hidden_states(
        model_with_engram,
        dataloader,
        device=device,
    )
    print(f"Engram model has {len(hidden_states_with_engram)} layers")
    del model_with_engram
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Step 2: Loading WITHOUT Engram model and extracting hidden states")
    print("=" * 60)
    model_without_engram = _load_baseline_model(
        base_model=base_model,
        baseline_checkpoint=resume_from_checkpoint_without_engram,
        load_device=load_device,
    )
    device = str(model_without_engram.device)
    hidden_states_without_engram = extract_final_token_hidden_states(
        model_without_engram,
        dataloader,
        device=device,
    )
    print(f"Baseline model has {len(hidden_states_without_engram)} layers")
    del model_without_engram
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("Step 3: Computing CKA Similarity Matrix")
    cka_device = "cuda" if torch.cuda.is_available() and str(model_without_engram.device) != "cpu" else "cpu"
    S = compute_cka_similarity_matrix(
        hidden_states_with_engram,
        hidden_states_without_engram,
        device=cka_device,
    )

    print(f"\nCKA Similarity Matrix shape: {S.shape}")
    print(f"Mean CKA: {S.mean():.4f}")
    print(f"Max CKA: {S.max():.4f}")
    print(f"Min CKA: {S.min():.4f}")

    best_matches = S.argmax(dim=1)
    print("\nBest matching layers (Engram -> Baseline):")
    for i, match in enumerate(best_matches):
        print(f"  Engram layer {i:2d} -> Baseline layer {match:2d} (CKA={S[i, match]:.4f})")

    print("\n" + "=" * 60)
    print("Step 4: Computing Soft Alignment Index")
    alignment_indices = compute_soft_alignment_index(S, k=k)
    print(f"\nSoft Alignment Indices (top-{k} weighted centroid):")
    for i, idx in enumerate(alignment_indices):
        print(f"  Engram layer {i:2d} aligns to Baseline layer {idx:.4f}")

    return {
        "cka_similarity_matrix": S.cpu().numpy(),
        "alignment_indices": alignment_indices.cpu().numpy(),
        "best_matches": best_matches.cpu().numpy(),
        "num_layers_engram": len(hidden_states_with_engram),
        "num_layers_baseline": len(hidden_states_without_engram),
        "mean_cka": S.mean().item(),
        "max_cka": S.max().item(),
        "batch_size": batch_size,
    }


def inference(
    batch_size: int = 4,
    resume_from_checkpoint_with_engram: str = "",
    resume_from_checkpoint_without_engram: str = "",
    base_model: str = "Qwen3-1.7B",
    test_file: str = DEFAULT_TEST_FILE,
    cutoff_len: int = 2048,
    engram_layer_ids: str = "6,13,20",
    engram_float32: bool = False,
    k: int = 5,
    max_samples: int = 0,
    hidden_state_target: str = "target_item_id",
    load_device: str = "auto",
    result_json: str = "",
    plot_path: str = "",
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    dataloader = _build_dataloader(
        tokenizer,
        test_file=test_file,
        cutoff_len=cutoff_len,
        batch_size=batch_size,
        max_samples=max_samples,
        hidden_state_target=hidden_state_target,
    )

    result = _evaluate_sequential(
        base_model=base_model,
        resume_from_checkpoint_with_engram=resume_from_checkpoint_with_engram,
        resume_from_checkpoint_without_engram=resume_from_checkpoint_without_engram,
        engram_layer_ids=engram_layer_ids,
        engram_float32=engram_float32,
        dataloader=dataloader,
        batch_size=batch_size,
        k=k,
        load_device=load_device,
    )
    parsed_engram_layer_ids = _parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    result["engram_layer_ids"] = parsed_engram_layer_ids

    if result_json:
        os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
        baseline_model_type = "qwen_base" if not resume_from_checkpoint_without_engram else "qwen_lora_checkpoint"
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cka_similarity_matrix": result["cka_similarity_matrix"].tolist(),
                    "alignment_indices": result["alignment_indices"].tolist(),
                    "best_matches": result["best_matches"].tolist(),
                    "num_layers_engram": result["num_layers_engram"],
                    "num_layers_baseline": result["num_layers_baseline"],
                    "mean_cka": result["mean_cka"],
                    "max_cka": result["max_cka"],
                    "batch_size": result["batch_size"],
                    "resume_from_checkpoint_with_engram": resume_from_checkpoint_with_engram,
                    "resume_from_checkpoint_without_engram": resume_from_checkpoint_without_engram,
                    "baseline_model_type": baseline_model_type,
                    "base_model": base_model,
                    "test_file": test_file,
                    "max_samples": max_samples,
                    "hidden_state_target": hidden_state_target,
                    "load_device": load_device,
                    "engram_layer_ids": parsed_engram_layer_ids,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[Info] 结果已保存到: {result_json}")

    if plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        plot_cka_results(result, save_path=plot_path, k=k)

    print(result)
    return result


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(inference)
    else:
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--resume_from_checkpoint_with_engram", type=str, default="")
        parser.add_argument("--resume_from_checkpoint_without_engram", type=str, default="")
        parser.add_argument("--base_model", type=str, default="Qwen3-1.7B")
        parser.add_argument("--test_file", type=str, default=DEFAULT_TEST_FILE)
        parser.add_argument("--cutoff_len", type=int, default=2048)
        parser.add_argument("--engram_layer_ids", type=str, default="6,13,20")
        parser.add_argument("--engram_float32", type=lambda x: str(x).lower() == "true", default=False)
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--max_samples", type=int, default=0)
        parser.add_argument("--hidden_state_target", type=str, default="target_item_id")
        parser.add_argument("--load_device", type=str, default="auto")
        parser.add_argument("--result_json", type=str, default="")
        parser.add_argument("--plot_path", type=str, default="")
        args = parser.parse_args()
        inference(**vars(args))
