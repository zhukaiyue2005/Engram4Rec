import json
import os
import re
import sys
import warnings
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import Qwen3ForCausalLM as Qwen3ForCausalLMWithoutEngram

from cka_evaluate_batch import evaluate, plot_cka_results

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


def _find_span_by_decode(tokenizer, pattern_ids: List[int], input_ids: List[int], warn_text: str = ""):
    start_idx = None
    end_idx = None
    target_text = tokenizer.decode(
        pattern_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    for idx in range(len(input_ids) - len(pattern_ids) + 1):
        window_text = tokenizer.decode(
            input_ids[idx : idx + len(pattern_ids)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()
        if window_text == target_text:
            start_idx = idx
            end_idx = idx + len(pattern_ids)
            break

    if start_idx is None and warn_text:
        warnings.warn(warn_text, UserWarning)
    return start_idx, end_idx


def _build_item_attention_mask(
    tokenizer,
    input_ids: List[int],
    history_start: int,
    history_end: int,
    completion_start: int,
    cans_start: Optional[int] = None,
    cans_end: Optional[int] = None,
) -> List[int]:
    item_attention_mask = [0] * len(input_ids)

    def is_sep_token(token_id: int) -> bool:
        tok_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        return tok_text.strip() in [",", "\n", "|", "<0x0A>", ""]

    for i in range(max(0, history_start), min(history_end, len(input_ids))):
        if not is_sep_token(input_ids[i]):
            item_attention_mask[i] = 1

    if cans_start is not None and cans_end is not None:
        for i in range(max(0, cans_start), min(cans_end, len(input_ids))):
            if not is_sep_token(input_ids[i]):
                item_attention_mask[i] = 1

    for i in range(max(0, completion_start), len(input_ids)):
        item_attention_mask[i] = 1

    def remove_phrase_spans(range_start: int, range_end: int, phrase_ids: List[int]):
        if len(phrase_ids) == 0:
            return
        for idx in np.where(np.array(input_ids[range_start:range_end]) == phrase_ids[0])[0]:
            real_idx = range_start + int(idx)
            if input_ids[real_idx : real_idx + len(phrase_ids)] == phrase_ids:
                for k in range(real_idx, real_idx + len(phrase_ids)):
                    item_attention_mask[k] = 0

    def remove_year_spans(range_start: int, range_end: int):
        i = range_start
        while i < range_end:
            tok_text = tokenizer.decode([input_ids[i]], clean_up_tokenization_spaces=False)
            if "(" in tok_text:
                buf = tok_text
                j = i
                while j + 1 < range_end and ")" not in buf and (j - i) < 10:
                    j += 1
                    buf += tokenizer.decode([input_ids[j]], clean_up_tokenization_spaces=False)
                normalized = re.sub(r"\s+", "", buf)
                if re.fullmatch(r"\(\d{4}\)[,.;:]?", normalized):
                    for k in range(i, j + 1):
                        item_attention_mask[k] = 0
                    i = j + 1
                    continue
            i += 1

    remove_phrase_spans(history_start, history_end, [9316, 416, 25])
    remove_year_spans(history_start, history_end)
    if cans_start is not None and cans_end is not None:
        remove_year_spans(cans_start, cans_end)

    return item_attention_mask


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


def _process_one_sample(row: dict, tokenizer, cutoff_len: int) -> dict:
    prompt = _normalize_prompt(row)
    input_ids = tokenizer(prompt, truncation=True, max_length=cutoff_len, add_special_tokens=True)["input_ids"]

    response_anchor_ids = tokenizer.encode("### Response:", add_special_tokens=False)
    before_anchor_ids = tokenizer.encode("before:", add_special_tokens=False)
    user_input_anchor_ids = tokenizer.encode("### User Input:", add_special_tokens=False)

    response_start, response_end = _find_span_by_decode(
        tokenizer,
        response_anchor_ids,
        input_ids,
        "Could not find response anchor in prompt during item_attention construction.",
    )
    _, before_end = _find_span_by_decode(tokenizer, before_anchor_ids, input_ids)
    _, user_input_end = _find_span_by_decode(tokenizer, user_input_anchor_ids, input_ids)

    history_start = before_end if before_end is not None else (user_input_end or 0)
    history_end = response_start if response_start is not None else len(input_ids)
    completion_start = response_end if response_end is not None else history_end

    item_attention_mask = _build_item_attention_mask(
        tokenizer,
        input_ids,
        history_start,
        history_end,
        completion_start=completion_start,
        cans_start=None,
        cans_end=None,
    )

    return {
        "input_ids": input_ids,
        "item_attention_mask": item_attention_mask,
        "prompt": prompt,
        "raw_row": row,
    }


def _left_pad_collate(samples: List[dict], pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in samples)
    input_ids = []
    attention_mask = []
    item_attention_mask = []
    valid_lengths = []
    prompts = []
    raw_rows = []

    for x in samples:
        seq = x["input_ids"]
        item_mask = x["item_attention_mask"]
        pad_len = max_len - len(seq)
        input_ids.append([pad_token_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        item_attention_mask.append([0] * pad_len + item_mask)
        valid_lengths.append(len(seq))
        prompts.append(x["prompt"])
        raw_rows.append(x["raw_row"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "item_attention_mask": torch.tensor(item_attention_mask, dtype=torch.long),
        "valid_lengths": valid_lengths,
        "prompts": prompts,
        "raw_rows": raw_rows,
    }


def _build_dataloader(tokenizer, test_file: str, cutoff_len: int, batch_size: int, max_samples: int = 0):
    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    if max_samples and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"[Info] CKA 使用样本数: {len(dataset)}", flush=True)
    processed = [_process_one_sample(row, tokenizer, cutoff_len) for row in dataset]
    return DataLoader(
        processed,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: _left_pad_collate(samples, tokenizer.pad_token_id),
    )


def _load_engram_model(
    base_model: str,
    resume_from_checkpoint: str,
    engram_layer_ids,
    engram_float32: bool,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = Qwen3ForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map=device_map,
    )
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


def _load_baseline_model(base_model: str, baseline_checkpoint: str = ""):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    model = Qwen3ForCausalLMWithoutEngram.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model)

    if baseline_checkpoint:
        if not os.path.exists(baseline_checkpoint):
            raise FileNotFoundError(f"baseline_checkpoint 不存在: {baseline_checkpoint}")
        model = PeftModel.from_pretrained(model, baseline_checkpoint)

    model.eval()
    return model


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
    )

    model_with_engram = _load_engram_model(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint_with_engram,
        engram_layer_ids=engram_layer_ids,
        engram_float32=engram_float32,
    )
    model_without_engram = _load_baseline_model(
        base_model=base_model,
        baseline_checkpoint=resume_from_checkpoint_without_engram,
    )

    result = evaluate(
        model_with_engram,
        model_without_engram,
        dataloader,
        batch_size=batch_size,
        k=k,
        device=str(model_with_engram.device),
    )

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
                    "engram_layer_ids": _parse_int_list_arg(engram_layer_ids, "engram_layer_ids"),
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
        parser.add_argument("--result_json", type=str, default="")
        parser.add_argument("--plot_path", type=str, default="")
        args = parser.parse_args()
        inference(**vars(args))
