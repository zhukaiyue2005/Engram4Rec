import json
import math
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_demo_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))

# 配置路径
engram_dir = os.path.join(os.path.dirname(__file__), "Engram_Insert_code")
sys.path.append(engram_dir)
from engram import Engram, EngramConfig
from modeling_qwen3 import Qwen3ForCausalLM

os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"

from huggingface_hub.utils._validators import validate_repo_id


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__


def _get_hash(x: List[int]) -> str:
    return "-".join([str(i) for i in x])


class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix_allowed_tokens_fn, num_beams: int, base_model: str):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams
        self.count = 0
        self.prefix_index = 4 if "gpt2" in base_model.lower() else 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -1_000_000)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-self.prefix_index :].tolist()
                else:
                    hash_key = sent[-self.count :].tolist()
                allowed = self._prefix_allowed_tokens_fn(batch_id, hash_key)
                if allowed:
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0
        self.count += 1
        return scores + mask


def _build_constraint_dict(tokenizer, base_model: str, info_file: str) -> Dict[str, List[int]]:
    with open(info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    item_names = ["\"" + ln[: -len(ln.split("\t")[-1])].strip() + "\"\n" for ln in lines]
    response_texts = [f"### Response:\n{name}" for name in item_names]

    if "llama" in base_model.lower():
        prefix_ids = [tokenizer(t).input_ids[1:] for t in response_texts]
    else:
        prefix_ids = [tokenizer(t).input_ids for t in response_texts]

    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    hash_dict: Dict[str, set] = {}
    for ids in prefix_ids:
        ids = ids + [tokenizer.eos_token_id]
        for i in range(prefix_index, len(ids)):
            key = _get_hash(ids[:i]) if i == prefix_index else _get_hash(ids[prefix_index:i])
            if key not in hash_dict:
                hash_dict[key] = set()
            hash_dict[key].add(ids[i])
    return {k: list(v) for k, v in hash_dict.items()}


def _normalize_target(row: dict) -> str:
    target = row.get("target_item_title", "")
    if target:
        return target.strip()
    raw = str(row.get("completion", "")).strip().strip("\n")
    return raw.strip('"')


def _load_item_mappings(info_file: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    title2id: Dict[str, int] = {}
    id2title: Dict[int, str] = {}
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            title = "\t".join(parts[:-1]).strip()
            item_id = int(parts[-1])
            title2id[title] = item_id
            id2title[item_id] = title
    return title2id, id2title


def _split_history_titles(history_str: str) -> List[str]:
    history_str = str(history_str or "").strip()
    if not history_str:
        return []
    return [x.strip() for x in history_str.split("::") if x.strip()]


def _locate_history_char_spans(prompt_text: str, history_titles: List[str]) -> List[List[int]]:
    spans: List[List[int]] = []
    cursor = 0
    for title in history_titles:
        needle = f'"{title}"'
        start = prompt_text.find(needle, cursor)
        if start < 0:
            start = prompt_text.find(needle)
        if start < 0:
            raise ValueError(f"failed to locate title in prompt: {title!r}")
        end = start + len(needle)
        spans.append([start, end])
        cursor = end
    return spans


def _token_span_from_offset_mapping(
    offsets: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
) -> List[int]:
    token_indices = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if end <= char_start:
            continue
        if start >= char_end:
            break
        token_indices.append(idx)
    if not token_indices:
        raise ValueError(f"failed to map char span to token span: [{char_start}, {char_end})")
    return [token_indices[0], token_indices[-1] + 1]


def _trim_token_span_boundaries(
    prompt_text: str,
    offsets: List[Tuple[int, int]],
    token_span: List[int],
) -> List[int]:
    start_idx, end_idx = token_span

    while start_idx < end_idx:
        char_start, char_end = offsets[start_idx]
        token_text = prompt_text[char_start:char_end]
        if token_text and not all(ch.isspace() or not ch.isalnum() for ch in token_text):
            break
        start_idx += 1

    while end_idx > start_idx:
        char_start, char_end = offsets[end_idx - 1]
        token_text = prompt_text[char_start:char_end]
        if token_text and not all(ch.isspace() or not ch.isalnum() for ch in token_text):
            break
        end_idx -= 1

    return [start_idx, end_idx]


def _supplement_history_item_fields(
    row: dict,
    tokenizer,
    cutoff_len: int,
    title2id: Dict[str, int],
) -> Tuple[List[int], List[List[int]]]:
    history_item_ids = row.get("history_item_ids", []) or []
    history_item_token_spans = (
        row.get("history_item_token_spans", [])
        or row.get("history_item_spans", [])
        or []
    )

    if history_item_ids and history_item_token_spans:
        return history_item_ids, history_item_token_spans

    history_titles = row.get("history_item_titles", []) or _split_history_titles(row.get("history_str", ""))
    if not history_titles:
        return [], []

    prompt = row.get("prompt", "")
    if isinstance(prompt, list):
        prompt_text = ""
        for item in prompt:
            if isinstance(item, dict) and item.get("role") == "user":
                prompt_text = item.get("content", "")
                break
        if not prompt_text:
            prompt_text = str(prompt)
    else:
        prompt_text = str(prompt)

    raw_completion = str(row.get("completion", "")).rstrip("\n")
    if not raw_completion:
        target_title = str(row.get("target_item_title", "")).strip()
        raw_completion = f"\"{target_title}\"" if target_title else ""
    completion = raw_completion + tokenizer.eos_token
    full_text = prompt_text + completion

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    offset_mapping = tokenized["offset_mapping"]

    history_item_ids = []
    history_item_token_spans = []
    for title, (char_start, char_end) in zip(history_titles, _locate_history_char_spans(prompt_text, history_titles)):
        if title not in title2id:
            continue
        try:
            raw_span = _token_span_from_offset_mapping(offset_mapping, char_start, char_end)
        except ValueError:
            continue
        trimmed_span = _trim_token_span_boundaries(prompt_text, offset_mapping, raw_span)
        if trimmed_span[1] <= trimmed_span[0]:
            continue
        history_item_ids.append(int(title2id[title]))
        history_item_token_spans.append(trimmed_span)

    return history_item_ids, history_item_token_spans


def _compute_metrics(preds: List[List[str]], targets: List[str], topk: List[int]) -> Dict[str, float]:
    n = len(targets)
    out: Dict[str, float] = {}
    if n == 0:
        for k in topk:
            out[f"Hit@{k}"] = 0.0
            out[f"NDCG@{k}"] = 0.0
        return out

    for k in topk:
        hit, ndcg = 0.0, 0.0
        for pred_list, target in zip(preds, targets):
            rank = -1
            for i, p in enumerate(pred_list[:k]):
                if p == target:
                    rank = i + 1
                    break
            if rank != -1:
                hit += 1.0
                ndcg += 1.0 / math.log2(rank + 1.0)
        out[f"Hit@{k}"] = hit / n
        out[f"NDCG@{k}"] = ndcg / n
    return out


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
    """Parse CLI arg that may come from fire as str/tuple/list into int list."""
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
        except (TypeError, ValueError):
            raise ValueError(f"{arg_name} 中包含非法整数值: {x!r}, 原始输入: {arg_value!r}")
    return out


def _find_span_by_decode(tokenizer, pattern_ids: List[int], input_ids: List[int], warn_text: str = ""):
    start_idx = None
    end_idx = None
    target_text = tokenizer.decode(
        pattern_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
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


def _process_one_sample(row: dict, tokenizer, cutoff_len: int, title2id: Dict[str, int]) -> dict:
    prompt_col = row.get("prompt", "")
    if isinstance(prompt_col, list):
        prompt = ""
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                prompt = item.get("content", "")
                break
        if not prompt:
            prompt = str(prompt_col)
    else:
        prompt = str(prompt_col)
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

    history_item_ids, history_item_token_spans = _supplement_history_item_fields(
        row=row,
        tokenizer=tokenizer,
        cutoff_len=cutoff_len,
        title2id=title2id,
    )

    target = _normalize_target(row)
    return {
        "input_ids": input_ids,
        "item_attention_mask": item_attention_mask,
        "history_item_ids": history_item_ids,
        "history_item_token_spans": history_item_token_spans,
        "prompt": prompt,
        "target": target,
        "raw_row": row,
    }


def _left_pad_collate(samples: List[dict], pad_token_id: int, sasrec_item_embeddings: Optional[torch.Tensor] = None):
    max_len = max(len(x["input_ids"]) for x in samples)
    input_ids = []
    attention_mask = []
    item_attention_mask = []
    sasrec_token_states = []
    prompts = []
    targets = []
    raw_rows = []

    for x in samples:
        seq = x["input_ids"]
        item_mask = x["item_attention_mask"]
        pad_len = max_len - len(seq)
        input_ids.append([pad_token_id] * pad_len + seq)
        attention_mask.append([0] * pad_len + [1] * len(seq))
        item_attention_mask.append([0] * pad_len + item_mask)

        if sasrec_item_embeddings is not None:
            hidden_size = int(sasrec_item_embeddings.shape[1])
            token_states = torch.zeros((max_len, hidden_size), dtype=sasrec_item_embeddings.dtype)
            for item_id, span in zip(x.get("history_item_ids", []), x.get("history_item_token_spans", [])):
                if span is None or len(span) != 2:
                    continue
                start, end = int(span[0]), int(span[1])
                start += pad_len
                end += pad_len
                start = max(0, min(start, max_len))
                end = max(start, min(end, max_len))
                if end <= start:
                    continue
                if item_id < 0 or item_id >= sasrec_item_embeddings.shape[0]:
                    continue
                item_vec = sasrec_item_embeddings[int(item_id)]
                token_states[start:end] = item_vec.unsqueeze(0).expand(end - start, -1)
            sasrec_token_states.append(token_states)

        prompts.append(x["prompt"])
        targets.append(x["target"])
        raw_rows.append(x["raw_row"])

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "item_attention_mask": torch.tensor(item_attention_mask, dtype=torch.long),
        "prompts": prompts,
        "targets": targets,
        "raw_rows": raw_rows,
        "prompt_len": max_len,
    }
    if sasrec_item_embeddings is not None:
        batch["sasrec_token_states"] = torch.stack(sasrec_token_states, dim=0)
    return batch


def inference(
    batch_size: int = 4,
    resume_from_checkpoint: str = "",
    base_model: str = "",
    test_file: str = "../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    sasrec_checkpoint_path: str = "./SAS-checkpoints/sasrec_best.pt",
    max_new_tokens: int = 256,
    cutoff_len: int = 2048,
    eval_topk: str = "1,3,5,10",
    length_penalty: float = 0.0,
    engram_layer_ids: str = "10,20",
    engram_float32: bool = False,
    save_json: str = "",
    print_batch_output: bool = True,
    print_prompt: bool = True,
    print_topn: int = 10,
    prompt_preview_chars: int = 0,
):
    test_file = resolve_demo_path(test_file)
    info_file = resolve_demo_path(info_file)
    sasrec_checkpoint_path = resolve_demo_path(sasrec_checkpoint_path)
    save_json = resolve_demo_path(save_json)
    print(f"[data] test_file = {test_file}", flush=True)
    print(f"[data] info_file = {info_file}", flush=True)

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

    sasrec_item_embeddings = None
    if sasrec_checkpoint_path:
        sasrec_ckpt = torch.load(sasrec_checkpoint_path, map_location="cpu")
        sasrec_state_dict = sasrec_ckpt["model_state_dict"]
        sasrec_item_embeddings = sasrec_state_dict["item_embeddings.weight"].detach().cpu().to(torch.float32)
        engram_config.item_hidden_size = int(sasrec_item_embeddings.shape[1])
        print(f"[SASRec] loaded item embeddings from {sasrec_checkpoint_path}", flush=True)
        print(f"[SASRec] item embedding table shape = {tuple(sasrec_item_embeddings.shape)}", flush=True)

    model.attach_engram(engram_config)
    model._reset_engram_parameters()

    def _load_engram_params(target_model, checkpoint_dir):
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            print(f"[Warn] 未找到 Engram 参数目录: {engram_params_dir}", flush=True)
            return
        unwrapped_model = target_model.module if hasattr(target_model, "module") else target_model
        loaded = 0
        expected_layers = []
        print(f"[Info] Engram 参数目录: {engram_params_dir}", flush=True)
        for _, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                expected_layers.append(layer_id)
                param_path = os.path.join(engram_params_dir, f"engram_layer_{layer_id}.npy")
                print(f"[Info] 检查 Engram 层 {layer_id}: {param_path}", flush=True)
                if os.path.exists(param_path):
                    load_result = module.load_all_params(param_path)
                    loaded_param_count = load_result[0] if isinstance(load_result, tuple) else load_result
                    loaded += 1
                    print(
                        f"[Info] Engram 层 {layer_id} 加载成功 | 参数文件存在 | loaded_params={loaded_param_count}",
                        flush=True,
                    )
                else:
                    print(f"[Warn] Engram 层 {layer_id} 参数缺失: {param_path}", flush=True)
        print(f"[Info] 期望加载 Engram 层: {sorted(expected_layers)}", flush=True)
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

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    hash_dict = _build_constraint_dict(tokenizer, base_model, info_file)
    title2id, _ = _load_item_mappings(info_file)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(_get_hash(input_ids), [])

    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    processed = [_process_one_sample(row, tokenizer, cutoff_len, title2id) for row in dataset]
    prompts = [x["prompt"] for x in processed]
    targets = [x["target"] for x in processed]

    topk = sorted(set(_parse_int_list_arg(eval_topk, "eval_topk")))
    if not topk:
        raise ValueError(f"eval_topk 不能为空，当前值: {eval_topk}")
    if min(topk) <= 0:
        raise ValueError(f"eval_topk 中 K 必须是正整数，当前值: {eval_topk}")
    num_beams = max(topk)

    all_preds: List[List[str]] = []
    total = len(processed)
    steps = (total + batch_size - 1) // batch_size
    start_time = time.time()

    for step in range(steps):
        s = step * batch_size
        e = min(total, s + batch_size)
        batch = _left_pad_collate(processed[s:e], tokenizer.pad_token_id, sasrec_item_embeddings=sasrec_item_embeddings)

        model_inputs = {
            "input_ids": batch["input_ids"].to(model.device),
            "attention_mask": batch["attention_mask"].to(model.device),
            "item_attention_mask": batch["item_attention_mask"].to(model.device),
        }
        if "sasrec_token_states" in batch:
            model_inputs["sasrec_token_states"] = batch["sasrec_token_states"].to(model.device)
        prompt_len = batch["prompt_len"]

        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
        logits_processor = LogitsProcessorList(
            [
                ConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                    num_beams=num_beams,
                    base_model=base_model,
                )
            ]
        )

        with torch.no_grad():
            generation_output = model.generate(
                **model_inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        completion_ids = generation_output.sequences[:, prompt_len:]
        if "llama" in base_model.lower():
            decoded = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        decoded = [x.split("Response:\n")[-1].strip().strip('"').strip() for x in decoded]

        batch_preds: List[List[str]] = []
        for i in range(0, len(decoded), num_beams):
            one_pred = decoded[i : i + num_beams]
            all_preds.append(one_pred)
            batch_preds.append(one_pred)

        if print_batch_output:
            current_topn = num_beams if print_topn <= 0 else min(max(1, print_topn), num_beams)
            print(f"\n=== Batch {step + 1}/{steps} | samples {s}-{e - 1} ===", flush=True)
            for local_i, pred_list in enumerate(batch_preds):
                data_i = s + local_i
                print(f"[Sample {data_i}] target: {targets[data_i]}", flush=True)
                if print_prompt:
                    prompt_text = prompts[data_i].replace("\n", "\\n")
                    if prompt_preview_chars > 0 and len(prompt_text) > prompt_preview_chars:
                        prompt_text = prompt_text[:prompt_preview_chars] + "..."
                    print(f"[Sample {data_i}] prompt: {prompt_text}", flush=True)
                print(f"[Sample {data_i}] top{current_topn}: {pred_list[:current_topn]}", flush=True)

        done_steps = step + 1
        progress = (done_steps / steps) * 100 if steps > 0 else 100.0
        elapsed = time.time() - start_time
        avg_step_time = elapsed / done_steps if done_steps > 0 else 0.0
        eta = avg_step_time * (steps - done_steps)
        print(
            f"[Progress] {done_steps}/{steps} ({progress:.2f}%) | "
            f"elapsed: {_format_seconds(elapsed)} | eta: {_format_seconds(eta)}",
            flush=True,
        )

    metrics = _compute_metrics(all_preds, targets, topk)
    print("=== Metrics ===", flush=True)
    for k in topk:
        print(f"Hit@{k}: {metrics[f'Hit@{k}']:.6f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}", flush=True)

    if save_json:
        rows = []
        for i, row in enumerate(dataset):
            one = dict(row)
            one["target_title"] = targets[i]
            one["predict"] = all_preds[i]
            rows.append(one)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {save_json}", flush=True)


if __name__ == "__main__":
    fire.Fire(inference)
