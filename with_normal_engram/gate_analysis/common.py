import os
import random
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
DEFAULT_DATA_FILE = os.path.join(PARENT_DIR, "../data/Industrial_and_Scientific_dataset/test.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "engram_results")
DEFAULT_RESULTS_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "engram_gates.json")
FALLBACK_RESULTS_JSON = os.path.join(CURRENT_DIR, "engram_results_test", "engram_gates.json")


def resolve_results_json_path(path: Optional[str] = None) -> str:
    if path:
        return path
    if os.path.exists(DEFAULT_RESULTS_JSON):
        return DEFAULT_RESULTS_JSON
    if os.path.exists(FALLBACK_RESULTS_JSON):
        return FALLBACK_RESULTS_JSON
    return DEFAULT_RESULTS_JSON


def set_deterministic_mode(seed: int = 1958) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
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

    out: List[int] = []
    for x in values:
        try:
            out.append(int(x))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{arg_name} 中包含非法整数值: {x!r}, 原始输入: {arg_value!r}") from exc
    return out


def sort_layer_names(layer_names: Sequence[str]) -> List[str]:
    def layer_key(name: str) -> Tuple[int, str]:
        match = re.search(r"(\d+)", str(name))
        if match:
            return (int(match.group(1)), str(name))
        return (10**9, str(name))

    return sorted(layer_names, key=layer_key)


def parse_prompt_column(prompt_col) -> str:
    if isinstance(prompt_col, list):
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
        return ""
    if isinstance(prompt_col, str):
        return prompt_col
    return ""


def extract_extra_info(extra_info_col):
    if isinstance(extra_info_col, dict):
        ground_truth = extra_info_col.get("ground_truth", {})
        description = ground_truth.get("description", "")
        title = ground_truth.get("title", "")
        history_list = extra_info_col.get("historyList", [])
        item_list = extra_info_col.get("itemList", [])
        return description, title, history_list, item_list
    return "", "", [], []


def normalize_target_title(row: dict) -> str:
    target = str(row.get("target_item_title", "")).strip()
    if target:
        return target
    completion = str(row.get("completion", "")).strip().strip("\n")
    return completion.strip('"')


def extract_history_list(row: dict) -> List[str]:
    history_str = str(row.get("history_str", "")).strip()
    if history_str:
        return [item for item in history_str.split("::") if item]

    _, _, history_list, _ = extract_extra_info(row.get("extra_info", {}))
    return list(history_list)


def build_sample_text(row: dict, tokenizer):
    prompt_text = parse_prompt_column(row.get("prompt", ""))
    if not prompt_text and "messages" in row:
        prompt_text = parse_prompt_column(row.get("messages", ""))

    if "extra_info" in row:
        description, true_selection, history_list, item_list = extract_extra_info(row.get("extra_info", {}))
        completion_text = f"<think>{description}</think><answer>{true_selection}</answer>{tokenizer.eos_token}"
        full_text = f"{prompt_text} {completion_text}"
        return {
            "prompt": prompt_text,
            "completion": true_selection,
            "description": description,
            "historyList": history_list,
            "itemList": item_list,
            "history_text": ", ".join(history_list),
            "target_text": true_selection,
            "text": full_text,
        }

    target_text = normalize_target_title(row)
    if str(row.get("target_item_title", "")).strip():
        completion_text = f"\"{target_text}\""
    else:
        completion_text = str(row.get("completion", "")).rstrip("\n")
    description = str(row.get("description", "") or "")
    history_list = extract_history_list(row)
    item_list = list(row.get("itemList", [])) if isinstance(row.get("itemList"), list) else []
    full_text = f"{prompt_text}{completion_text}{tokenizer.eos_token}"
    return {
        "prompt": prompt_text,
        "completion": completion_text,
        "description": description,
        "historyList": history_list,
        "itemList": item_list,
        "history_text": extract_history_text_from_prompt(prompt_text, history_list),
        "target_text": target_text,
        "text": full_text,
    }


def extract_history_text_from_prompt(prompt_text: str, history_list: Sequence[str]) -> str:
    match = re.search(r'before:\s*(.*?)\n\n### Response:\n\Z', prompt_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    if history_list:
        return ",\t".join(f"\"{item}\"" for item in history_list)
    return ""


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not needle or len(needle) > len(haystack):
        return None
    end = len(haystack) - len(needle) + 1
    for start in range(end):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start, start + len(needle)
    return None


def locate_text_span(tokenizer, input_ids: Sequence[int], text: str) -> List[int]:
    if not text:
        return [0, 0]

    variants = [
        text,
        f" {text}",
        text.strip(),
        f" {text.strip()}",
        f"{text}\n",
        f"{text.strip()}\n",
        f" {text.strip()}\n",
    ]
    seen = set()
    for variant in variants:
        if not variant or variant in seen:
            continue
        seen.add(variant)
        token_ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        span = find_subsequence(input_ids, token_ids)
        if span is not None:
            return [int(span[0]), int(span[1])]
    return [0, 0]
