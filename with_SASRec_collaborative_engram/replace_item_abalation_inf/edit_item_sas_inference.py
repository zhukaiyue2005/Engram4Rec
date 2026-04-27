import json
import math
import os
import re
import sys
from typing import Dict, List, Optional

try:
    import fire
except ImportError:
    fire = None
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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
ENGRAM_DIR = os.path.join(PROJECT_DIR, "Engram_Insert_code")
if ENGRAM_DIR not in sys.path:
    sys.path.append(ENGRAM_DIR)

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


DEFAULT_BASE_MODEL = "Qwen3-1.7B"
DEFAULT_TEST_FILE = os.path.join(PROJECT_DIR, "../data/Industrial_and_Scientific_dataset/test.jsonl")
DEFAULT_INFO_FILE = os.path.join(PROJECT_DIR, "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt")
DEFAULT_SASREC_CKPT = os.path.join(PROJECT_DIR, "SAS-checkpoints", "sasrec_best.pt")
DEFAULT_RESUME_CKPT = os.path.join(
    PROJECT_DIR,
    "output_sft_sasrec_item_spans_layers=6,13,20_tbs=128_lora=3e-4_engram=3e-4",
    "final_checkpoint_sft",
)
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "results")
DEFAULT_REPLACEMENT_PLAN_FILE = os.path.join(CURRENT_DIR, "replace_item_ids_plan.json")


def _read_sample_indices_file(sample_indices_file: str) -> List[int]:
    with open(sample_indices_file, "r", encoding="utf-8") as f:
        text = f.read()
    return _parse_int_list_arg(text, "sample_indices_file")


def _get_hash(x: List[int]) -> str:
    return "-".join(str(i) for i in x)


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


def _parse_int_list_arg(arg_value, arg_name: str) -> List[int]:
    values = []
    if isinstance(arg_value, (tuple, list)):
        for x in arg_value:
            if isinstance(x, str):
                values.extend([p.strip() for p in re.split(r"[\s,]+", x.strip()) if p.strip()])
            elif x is not None:
                values.append(x)
    elif isinstance(arg_value, str):
        values = [p.strip() for p in re.split(r"[\s,]+", arg_value.strip()) if p.strip()]
    elif arg_value is not None:
        values = [arg_value]

    out: List[int] = []
    for x in values:
        try:
            out.append(int(x))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{arg_name} 中包含非法整数值: {x!r}, 原始输入: {arg_value!r}") from exc
    return out


def _build_constraint_dict(tokenizer, base_model: str, info_file: str) -> Dict[str, List[int]]:
    with open(info_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    item_names = ['"' + ln[: -len(ln.split("\t")[-1])].strip() + '"\n' for ln in lines]
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
            hash_dict.setdefault(key, set()).add(ids[i])
    return {k: list(v) for k, v in hash_dict.items()}


def _normalize_target(row: dict) -> str:
    target = str(row.get("target_item_title", "")).strip()
    if target:
        return target
    raw = str(row.get("completion", "")).strip().strip("\n")
    return raw.strip('"')


def _load_dataset(test_file: str):
    return load_dataset("json", data_files={"test": test_file})["test"]


def _load_sample_from_dataset(dataset, sample_idx: int) -> dict:
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx 越界: {sample_idx}, 数据集长度: {len(dataset)}")
    return dict(dataset[sample_idx])


def _extract_prompt(row: dict) -> str:
    prompt_col = row.get("prompt", "")
    if isinstance(prompt_col, list):
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
        return str(prompt_col)
    return str(prompt_col)


def _build_token_states(
    history_item_ids: List[int],
    history_item_token_spans: List[List[int]],
    prompt_len: int,
    sasrec_item_embeddings: torch.Tensor,
) -> torch.Tensor:
    hidden_size = int(sasrec_item_embeddings.shape[1])
    token_states = torch.zeros((1, prompt_len, hidden_size), dtype=sasrec_item_embeddings.dtype)
    for item_id, span in zip(history_item_ids, history_item_token_spans):
        if span is None or len(span) != 2:
            continue
        start, end = int(span[0]), int(span[1])
        start = max(0, min(start, prompt_len))
        end = max(start, min(end, prompt_len))
        if end <= start:
            continue
        if item_id < 0 or item_id >= sasrec_item_embeddings.shape[0]:
            continue
        item_vec = sasrec_item_embeddings[int(item_id)]
        token_states[0, start:end] = item_vec.unsqueeze(0).expand(end - start, -1)
    return token_states


def _load_model(
    base_model: str,
    resume_from_checkpoint: str,
    engram_layer_ids: str,
    sasrec_checkpoint_path: str,
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

    layer_ids = _parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if not layer_ids:
        raise ValueError("engram_layer_ids 不能为空")

    engram_config = EngramConfig()
    engram_config.layer_ids = layer_ids

    sasrec_ckpt = torch.load(sasrec_checkpoint_path, map_location="cpu")
    sasrec_state_dict = sasrec_ckpt["model_state_dict"]
    sasrec_item_embeddings = sasrec_state_dict["item_embeddings.weight"].detach().cpu().to(torch.float32)
    engram_config.item_hidden_size = int(sasrec_item_embeddings.shape[1])

    model.attach_engram(engram_config)
    model._reset_engram_parameters()

    def _load_engram_params(target_model, checkpoint_dir):
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            raise FileNotFoundError(f"未找到 Engram 参数目录: {engram_params_dir}")
        unwrapped_model = target_model.module if hasattr(target_model, "module") else target_model
        loaded = 0
        for _, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                param_path = os.path.join(engram_params_dir, f"engram_layer_{layer_id}.npy")
                if os.path.exists(param_path):
                    module.load_all_params(param_path)
                    loaded += 1
        print(f"[Info] Engram 参数加载层数: {loaded}")

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
    return model, sasrec_item_embeddings


def _run_generate(
    model,
    tokenizer,
    prompt: str,
    sasrec_token_states: torch.Tensor,
    info_file: str,
    base_model: str,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
):
    return _run_generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        sasrec_token_states_batch=sasrec_token_states,
        info_file=info_file,
        base_model=base_model,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
    )[0]


def _run_generate_batch(
    model,
    tokenizer,
    prompt: str,
    sasrec_token_states_batch: torch.Tensor,
    info_file: str,
    base_model: str,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float,
):
    input_ids = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=True)["input_ids"]
    batch_size = int(sasrec_token_states_batch.shape[0])
    input_tensor = torch.tensor([input_ids for _ in range(batch_size)], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=model.device)

    hash_dict = _build_constraint_dict(tokenizer, base_model, info_file)

    def prefix_allowed_tokens_fn(batch_id, ids):
        return hash_dict.get(_get_hash(ids), [])

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
        output = model.generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            sasrec_token_states=sasrec_token_states_batch.to(model.device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logits_processor,
        )

    prompt_len = input_tensor.shape[1]
    completion_ids = output.sequences[:, prompt_len:]
    decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    decoded = [x.split("Response:\n")[-1].strip().strip('"').strip() for x in decoded]
    grouped = []
    for batch_idx in range(batch_size):
        start = batch_idx * num_beams
        end = start + num_beams
        grouped.append(decoded[start:end])
    return grouped


def _resolve_replacement_item_id(row: dict, replacement_item_id: int, use_target_item: bool) -> int:
    if replacement_item_id >= 0:
        return int(replacement_item_id)
    if use_target_item:
        target_item_id = int(row.get("target_item_id", -1))
        if target_item_id >= 0:
            return target_item_id
    raise ValueError("需要提供 replacement_item_id，或者启用 use_target_item=True")


def _load_replacement_plan_for_sample(replacement_plan_file: str, sample_idx: int) -> dict:
    with open(replacement_plan_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data.get("samples", []):
        if int(sample.get("sample_idx", -1)) == int(sample_idx):
            return sample
    raise ValueError(f"在 replacement_plan_file 中未找到 sample_idx={sample_idx}: {replacement_plan_file}")


def _get_plan_group_map(plan: dict) -> Dict[str, dict]:
    group_map: Dict[str, dict] = {}
    for group in plan.get("replace_item_ids_by_group", []):
        style_group = str(group.get("style_group", "")).strip()
        if style_group:
            group_map[style_group] = group
    return group_map


def _resolve_candidate_group_names(plan: dict, replacement_style_group: str) -> List[str]:
    group_map = _get_plan_group_map(plan)
    if replacement_style_group:
        if replacement_style_group not in group_map:
            raise ValueError(
                f"sample_idx={plan.get('sample_idx')} 在 replacement_plan_file 中没有 style_group={replacement_style_group}"
            )
        return [replacement_style_group]

    suggested_groups = [str(x).strip() for x in plan.get("suggested_opposite_style_groups", []) if str(x).strip()]
    resolved_groups = [group_name for group_name in suggested_groups if group_name in group_map]
    if resolved_groups:
        return resolved_groups

    fallback_groups = list(group_map.keys())
    if fallback_groups:
        return [fallback_groups[0]]
    raise ValueError(f"sample_idx={plan.get('sample_idx')} 没有任何 replace_item_ids_by_group")


def _resolve_replacement_item_ids(
    sample_idx: int,
    num_history_items: int,
    replacement_item_ids: str,
    replacement_style_group: str,
    replacement_plan_file: str,
    row: dict,
    replacement_item_id: int,
    use_target_item: bool,
) -> tuple[list[int], str]:
    def _expand_with_repeat(candidate_ids: List[int], required_count: int, source_name: str) -> tuple[list[int], str]:
        if not candidate_ids:
            raise ValueError(f"{source_name} 没有任何可用候选 item ids")
        if len(candidate_ids) >= required_count:
            return candidate_ids[:required_count], source_name
        expanded = list(candidate_ids)
        repeat_idx = 0
        while len(expanded) < required_count:
            expanded.append(candidate_ids[repeat_idx % len(candidate_ids)])
            repeat_idx += 1
        return expanded, f"{source_name}_with_repeat"

    explicit_ids = _parse_int_list_arg(replacement_item_ids, "replacement_item_ids")
    if explicit_ids:
        return _expand_with_repeat(explicit_ids, num_history_items, "explicit_item_ids")

    if replacement_style_group or os.path.exists(replacement_plan_file):
        plan = _load_replacement_plan_for_sample(replacement_plan_file, sample_idx)
        group_map = _get_plan_group_map(plan)
        candidate_group_names = _resolve_candidate_group_names(plan, replacement_style_group)
        chosen_group = group_map[candidate_group_names[0]]

        candidate_ids = [int(x) for x in chosen_group.get("candidate_item_ids", [])]
        return _expand_with_repeat(
            candidate_ids,
            num_history_items,
            str(chosen_group.get("style_group", "unknown_group")),
        )

    single_replacement_id = _resolve_replacement_item_id(row, replacement_item_id, use_target_item)
    return [single_replacement_id for _ in range(num_history_items)], "single_repeated_item_id"


def _compare_prediction_lists(original_preds: List[str], edited_preds: List[str]) -> dict:
    max_len = max(len(original_preds), len(edited_preds))
    changed_positions = []
    for idx in range(max_len):
        original_text = original_preds[idx] if idx < len(original_preds) else ""
        edited_text = edited_preds[idx] if idx < len(edited_preds) else ""
        if original_text != edited_text:
            changed_positions.append(
                {
                    "rank": idx + 1,
                    "original": original_text,
                    "edited": edited_text,
                }
            )
    return {
        "top1_changed": bool(original_preds and edited_preds and original_preds[0] != edited_preds[0]),
        "num_changed_positions": len(changed_positions),
        "changed_positions": changed_positions,
    }


def _compare_topn_ignore_rank(original_preds: List[str], edited_preds: List[str], topn: int = 10) -> dict:
    original_topn = list(original_preds[:topn])
    edited_topn = list(edited_preds[:topn])
    original_only = [x for x in original_topn if x not in edited_topn]
    edited_only = [x for x in edited_topn if x not in original_topn]
    return {
        "topn": topn,
        "original_topn": original_topn,
        "edited_topn": edited_topn,
        "original_only": original_only,
        "edited_only": edited_only,
        "total_different_items": len(original_only) + len(edited_only),
    }


def _write_log(
    log_path: str,
    prompt: str,
    result: dict,
):
    lines = []
    lines.append("=" * 100)
    lines.append(
        f"[Sample] {result.get('run_sample_idx', result['sample_idx'])} "
        f"(dataset_sample_idx={result['sample_idx']})"
    )
    lines.append(f"[Correct Target] item_id={result['target_item_id']} | text={result['target_text']}")
    lines.append(f"[Replacement Source] {result['replacement_source']}")
    lines.append("")
    lines.append("[Prompt]")
    lines.append(prompt)
    lines.append("")
    lines.append("[History Item IDs]")
    lines.append(str(result["history_item_ids"]))
    lines.append("[History Item Titles]")
    for idx, title in enumerate(result["history_item_titles"]):
        span = result["history_item_token_spans"][idx] if idx < len(result["history_item_token_spans"]) else None
        lines.append(f"- idx={idx} span={span} title={title}")
    lines.append("")
    lines.append("[Original Engram Generation]")
    for rank, pred in enumerate(result["original_engram_generation"]["top_predictions"], start=1):
        lines.append(f"- rank={rank} pred={pred}")
    lines.append("")
    lines.append("[Edited Engram Results]")
    for one in result["edited_results"]:
        lines.append(f"- mode={one['edit_mode']}")
        lines.append(f"  top1_changed={one['difference_summary']['top1_changed']}")
        lines.append(f"  num_changed_positions={one['difference_summary']['num_changed_positions']}")
        lines.append("  replacement_map:")
        for mapping in one["replacement_map"]:
            lines.append(
                f"    - idx={mapping['history_index']} span={mapping['span']} "
                f"original_item_id={mapping['original_item_id']} "
                f"replacement_item_id={mapping['replacement_item_id']} "
                f"title={mapping['history_item_title']}"
            )
        lines.append("  edited_top_predictions:")
        for rank, pred in enumerate(one["edited_engram_generation"]["top_predictions"], start=1):
            lines.append(f"    - rank={rank} pred={pred}")
        top10_diff = one.get("top10_ignore_rank_difference", {})
        lines.append("  top10_ignore_rank_difference:")
        lines.append(f"    - total_different_items={top10_diff.get('total_different_items', 0)}")
        lines.append(f"    - original_only_count={len(top10_diff.get('original_only', []))}")
        for text in top10_diff.get("original_only", []):
            lines.append(f"      - original_only={text}")
        lines.append(f"    - edited_only_count={len(top10_diff.get('edited_only', []))}")
        for text in top10_diff.get("edited_only", []):
            lines.append(f"      - edited_only={text}")
        lines.append("")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _build_summary_header() -> List[str]:
    return [
        "=" * 100,
        "[Edit Item SAS Summary]",
        "",
        "",
    ]


def _build_summary_lines_for_result(prompt: str, result: dict) -> List[str]:
    lines = []
    sample_idx = int(result["sample_idx"])
    run_sample_idx = int(result.get("run_sample_idx", sample_idx))
    lines.append("=" * 100)
    lines.append(f"[Sample] {run_sample_idx} (dataset_sample_idx={sample_idx})")
    lines.append(f"[Correct Target] item_id={result['target_item_id']} | text={result['target_text']}")
    lines.append(f"[Replacement Source] {result['replacement_source']}")
    lines.append("")
    lines.append("[Prompt]")
    lines.append(prompt)
    lines.append("")
    for one in result["edited_results"]:
        replacement_group = one.get("replacement_group", "")
        if replacement_group:
            lines.append(f"[Edited Result] mode={one['edit_mode']} | group={replacement_group}")
        else:
            lines.append(f"[Edited Result] mode={one['edit_mode']}")
        replacement_pairs = ", ".join(
            f"{m['original_item_id']}->{m['replacement_item_id']}" for m in one["replacement_map"]
        )
        lines.append(f"- replacements=({replacement_pairs})")
        top10_diff = one.get("top10_ignore_rank_difference", {})
        lines.append(f"- top10_total_different_items={top10_diff.get('total_different_items', 0)}")
        lines.append("- original_top10_items:")
        for text in top10_diff.get("original_topn", []):
            lines.append(f"  - {text}")
        lines.append("- edited_top10_items:")
        for text in top10_diff.get("edited_topn", []):
            lines.append(f"  - {text}")
        lines.append("- original_only_items:")
        for text in top10_diff.get("original_only", []):
            lines.append(f"  - {text}")
        lines.append("- edited_only_items:")
        for text in top10_diff.get("edited_only", []):
            lines.append(f"  - {text}")
        lines.append("")
        lines.append("")
    lines.append("")
    return lines


def _initialize_summary_log(summary_log_path: str):
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_build_summary_header()) + "\n")


def _append_summary_log(summary_log_path: str, prompt: str, result: dict):
    with open(summary_log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(_build_summary_lines_for_result(prompt, result)) + "\n")


def _write_summary_log(summary_log_path: str, prompt_by_sample: Dict[int, str], results: List[dict]):
    lines = []
    lines.extend(_build_summary_header())
    lines.append(f"num_samples = {len(results)}")
    lines.append("")
    for result in results:
        sample_idx = int(result["sample_idx"])
        prompt = prompt_by_sample.get(sample_idx, "")
        lines.extend(_build_summary_lines_for_result(prompt, result))
    with open(summary_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _run_one_sample(
    sample_idx: int,
    dataset,
    tokenizer,
    model,
    sasrec_item_embeddings,
    base_model: str = DEFAULT_BASE_MODEL,
    info_file: str = DEFAULT_INFO_FILE,
    replacement_mode: str = "all_history_distinct",
    replacement_style_group: str = "",
    replacement_item_ids: str = "",
    replacement_plan_file: str = DEFAULT_REPLACEMENT_PLAN_FILE,
    replacement_item_id: int = -1,
    use_target_item: bool = True,
    edit_span_indices: str = "all",
    max_new_tokens: int = 64,
    num_beams: int = 10,
    length_penalty: float = 0.0,
):
    row = _load_sample_from_dataset(dataset, sample_idx)
    prompt = _extract_prompt(row)
    target = _normalize_target(row)
    history_item_ids = [int(x) for x in row.get("history_item_ids", [])]
    history_item_token_spans = [list(span) for span in row.get("history_item_token_spans", [])]
    replacement_plan = None
    if (replacement_style_group or str(replacement_plan_file).strip()) and os.path.exists(replacement_plan_file):
        replacement_plan = _load_replacement_plan_for_sample(replacement_plan_file, sample_idx)

    prompt_ids = tokenizer(prompt, truncation=True, max_length=2048, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_ids)

    baseline_states = _build_token_states(
        history_item_ids=history_item_ids,
        history_item_token_spans=history_item_token_spans,
        prompt_len=prompt_len,
        sasrec_item_embeddings=sasrec_item_embeddings,
    )
    baseline_preds = None

    edited_results = []
    if replacement_mode == "all_history_distinct":
        plan_group_names: List[str] = []
        if replacement_plan is not None:
            plan_group_names = _resolve_candidate_group_names(replacement_plan, replacement_style_group)

        if plan_group_names and not replacement_item_ids:
            replacement_source = "suggested_opposite_style_groups"
            batch_states = [baseline_states]
            pending_variants = []
            for one_group_name in plan_group_names:
                resolved_replacement_ids, one_source = _resolve_replacement_item_ids(
                    sample_idx=sample_idx,
                    num_history_items=len(history_item_ids),
                    replacement_item_ids=replacement_item_ids,
                    replacement_style_group=one_group_name,
                    replacement_plan_file=replacement_plan_file,
                    row=row,
                    replacement_item_id=replacement_item_id,
                    use_target_item=use_target_item,
                )
                edited_item_ids = list(resolved_replacement_ids)
                edited_states = _build_token_states(
                    history_item_ids=edited_item_ids,
                    history_item_token_spans=history_item_token_spans,
                    prompt_len=prompt_len,
                    sasrec_item_embeddings=sasrec_item_embeddings,
                )
                batch_states.append(edited_states)
                pending_variants.append(
                    {
                        "replacement_group": one_group_name,
                        "replacement_source": one_source,
                        "edited_item_ids": edited_item_ids,
                    }
                )
            batch_preds = _run_generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                sasrec_token_states_batch=torch.cat(batch_states, dim=0),
                info_file=info_file,
                base_model=base_model,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
            )
            baseline_preds = batch_preds[0]
            for variant_idx, variant in enumerate(pending_variants, start=1):
                edited_item_ids = variant["edited_item_ids"]
                edited_preds = batch_preds[variant_idx]
                replacement_map = []
                for idx, (original_item_id, replacement_id, span) in enumerate(
                    zip(history_item_ids, edited_item_ids, history_item_token_spans)
                ):
                    replacement_map.append(
                        {
                            "history_index": idx,
                            "span": span,
                            "history_item_title": row.get("history_item_titles", [])[idx]
                            if idx < len(row.get("history_item_titles", []))
                            else "",
                            "original_item_id": int(original_item_id),
                            "replacement_item_id": int(replacement_id),
                        }
                    )
                edited_results.append(
                    {
                        "edit_mode": replacement_mode,
                        "replacement_group": variant["replacement_group"],
                        "replacement_source": variant["replacement_source"],
                        "replacement_map": replacement_map,
                        "original_engram_generation": {
                            "history_item_ids": history_item_ids,
                            "top_predictions": baseline_preds,
                        },
                        "edited_engram_generation": {
                            "history_item_ids": edited_item_ids,
                            "top_predictions": edited_preds,
                        },
                        "difference_summary": _compare_prediction_lists(baseline_preds, edited_preds),
                        "top10_ignore_rank_difference": _compare_topn_ignore_rank(baseline_preds, edited_preds, topn=10),
                    }
                )
        else:
            resolved_replacement_ids, replacement_source = _resolve_replacement_item_ids(
                sample_idx=sample_idx,
                num_history_items=len(history_item_ids),
                replacement_item_ids=replacement_item_ids,
                replacement_style_group=replacement_style_group,
                replacement_plan_file=replacement_plan_file,
                row=row,
                replacement_item_id=replacement_item_id,
                use_target_item=use_target_item,
            )
            edited_item_ids = list(resolved_replacement_ids)
            edited_states = _build_token_states(
                history_item_ids=edited_item_ids,
                history_item_token_spans=history_item_token_spans,
                prompt_len=prompt_len,
                sasrec_item_embeddings=sasrec_item_embeddings,
            )
            batch_preds = _run_generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                sasrec_token_states_batch=torch.cat([baseline_states, edited_states], dim=0),
                info_file=info_file,
                base_model=base_model,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
            )
            baseline_preds = batch_preds[0]
            edited_preds = batch_preds[1]
            replacement_map = []
            for idx, (original_item_id, replacement_id, span) in enumerate(
                zip(history_item_ids, edited_item_ids, history_item_token_spans)
            ):
                replacement_map.append(
                    {
                        "history_index": idx,
                        "span": span,
                        "history_item_title": row.get("history_item_titles", [])[idx]
                        if idx < len(row.get("history_item_titles", []))
                        else "",
                        "original_item_id": int(original_item_id),
                        "replacement_item_id": int(replacement_id),
                    }
                )
            edited_results.append(
                {
                    "edit_mode": replacement_mode,
                    "replacement_source": replacement_source,
                    "replacement_map": replacement_map,
                    "original_engram_generation": {
                        "history_item_ids": history_item_ids,
                        "top_predictions": baseline_preds,
                    },
                    "edited_engram_generation": {
                        "history_item_ids": edited_item_ids,
                        "top_predictions": edited_preds,
                    },
                    "difference_summary": _compare_prediction_lists(baseline_preds, edited_preds),
                    "top10_ignore_rank_difference": _compare_topn_ignore_rank(baseline_preds, edited_preds, topn=10),
                }
            )
    else:
        replacement_source = "single_span_edit"
        if str(edit_span_indices).strip().lower() == "all":
            target_indices = list(range(len(history_item_ids)))
        else:
            target_indices = _parse_int_list_arg(edit_span_indices, "edit_span_indices")

        replacement_id = _resolve_replacement_item_id(row, replacement_item_id, use_target_item)
        batch_states = [baseline_states]
        pending_variants = []
        for span_idx in target_indices:
            if span_idx < 0 or span_idx >= len(history_item_ids):
                raise IndexError(f"span_idx 越界: {span_idx}, 总 span 数: {len(history_item_ids)}")
            edited_item_ids = list(history_item_ids)
            original_item_id = int(edited_item_ids[span_idx])
            edited_item_ids[span_idx] = replacement_id

            edited_states = _build_token_states(
                history_item_ids=edited_item_ids,
                history_item_token_spans=history_item_token_spans,
                prompt_len=prompt_len,
                sasrec_item_embeddings=sasrec_item_embeddings,
            )
            batch_states.append(edited_states)
            pending_variants.append(
                {
                    "span_idx": span_idx,
                    "original_item_id": original_item_id,
                    "edited_item_ids": edited_item_ids,
                }
            )
        batch_preds = _run_generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            sasrec_token_states_batch=torch.cat(batch_states, dim=0),
            info_file=info_file,
            base_model=base_model,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )
        baseline_preds = batch_preds[0]
        for variant_idx, variant in enumerate(pending_variants, start=1):
            span_idx = variant["span_idx"]
            edited_item_ids = variant["edited_item_ids"]
            edited_preds = batch_preds[variant_idx]
            edited_results.append(
                {
                    "edit_mode": replacement_mode,
                    "replacement_map": [
                        {
                            "history_index": span_idx,
                            "span": history_item_token_spans[span_idx],
                            "history_item_title": row.get("history_item_titles", [])[span_idx]
                            if span_idx < len(row.get("history_item_titles", []))
                            else "",
                            "original_item_id": variant["original_item_id"],
                            "replacement_item_id": replacement_id,
                        }
                    ],
                    "original_engram_generation": {
                        "history_item_ids": history_item_ids,
                        "top_predictions": baseline_preds,
                    },
                    "edited_engram_generation": {
                        "history_item_ids": edited_item_ids,
                        "top_predictions": edited_preds,
                    },
                    "difference_summary": _compare_prediction_lists(baseline_preds, edited_preds),
                    "top10_ignore_rank_difference": _compare_topn_ignore_rank(baseline_preds, edited_preds, topn=10),
                }
            )

    if baseline_preds is None:
        baseline_preds = _run_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            sasrec_token_states=baseline_states,
            info_file=info_file,
            base_model=base_model,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    result = {
        "sample_idx": sample_idx,
        "target_text": target,
        "target_item_id": int(row.get("target_item_id", -1)),
        "replacement_source": replacement_source,
        "history_item_ids": history_item_ids,
        "history_item_titles": row.get("history_item_titles", []),
        "history_item_token_spans": history_item_token_spans,
        "original_engram_generation": {
            "history_item_ids": history_item_ids,
            "top_predictions": baseline_preds,
        },
        "edited_results": edited_results,
    }

    return result, prompt


def run(
    sample_idx: int = 0,
    sample_indices: str = "",
    sample_indices_file: str = "",
    base_model: str = DEFAULT_BASE_MODEL,
    resume_from_checkpoint: str = DEFAULT_RESUME_CKPT,
    test_file: str = DEFAULT_TEST_FILE,
    info_file: str = DEFAULT_INFO_FILE,
    sasrec_checkpoint_path: str = DEFAULT_SASREC_CKPT,
    engram_layer_ids: str = "6,13,20",
    engram_float32: bool = False,
    replacement_mode: str = "all_history_distinct",
    replacement_style_group: str = "",
    replacement_item_ids: str = "",
    replacement_plan_file: str = DEFAULT_REPLACEMENT_PLAN_FILE,
    replacement_item_id: int = -1,
    use_target_item: bool = True,
    edit_span_indices: str = "all",
    max_new_tokens: int = 64,
    num_beams: int = 10,
    length_penalty: float = 0.0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    save_json: bool = True,
    save_log: bool = True,
    summary_log_file: str = "",
):
    os.makedirs(output_dir, exist_ok=True)
    active_summary_log_file = ""
    if summary_log_file or save_log:
        active_summary_log_file = summary_log_file or os.path.join(output_dir, "edit_item_sas_summary.log")
        _initialize_summary_log(active_summary_log_file)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    dataset = _load_dataset(test_file)
    model, sasrec_item_embeddings = _load_model(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        engram_layer_ids=engram_layer_ids,
        sasrec_checkpoint_path=sasrec_checkpoint_path,
        engram_float32=engram_float32,
    )

    if str(sample_indices_file).strip():
        target_sample_indices = _read_sample_indices_file(sample_indices_file)
    elif str(sample_indices).strip():
        target_sample_indices = _parse_int_list_arg(sample_indices, "sample_indices")
    else:
        target_sample_indices = [int(sample_idx)]

    prompt_by_sample: Dict[int, str] = {}
    results: List[dict] = []
    for run_sample_idx, one_sample_idx in enumerate(target_sample_indices):
        result, prompt = _run_one_sample(
            sample_idx=one_sample_idx,
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            sasrec_item_embeddings=sasrec_item_embeddings,
            base_model=base_model,
            info_file=info_file,
            replacement_mode=replacement_mode,
            replacement_style_group=replacement_style_group,
            replacement_item_ids=replacement_item_ids,
            replacement_plan_file=replacement_plan_file,
            replacement_item_id=replacement_item_id,
            use_target_item=use_target_item,
            edit_span_indices=edit_span_indices,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )
        result["run_sample_idx"] = int(run_sample_idx)
        prompt_by_sample[one_sample_idx] = prompt
        results.append(result)
        if active_summary_log_file:
            _append_summary_log(active_summary_log_file, prompt=prompt, result=result)
            print(f"[Saved] {active_summary_log_file}")

        print(f"[Sample] {run_sample_idx} (dataset_sample_idx={one_sample_idx})")
        print(f"[Correct Target] item_id={result['target_item_id']} | text={result['target_text']}")
        for one in result["edited_results"]:
            replacement_pairs = ", ".join(
                f"{m['original_item_id']}->{m['replacement_item_id']}" for m in one["replacement_map"]
            )
            replacement_group = one.get("replacement_group", "")
            print(
                f"[Edit mode={one['edit_mode']}"
                f"{' group=' + replacement_group if replacement_group else ''}] "
                f"replacements=({replacement_pairs}) | "
                f"top10_total_different_items={one['top10_ignore_rank_difference']['total_different_items']}"
            )

        if save_json:
            save_path = os.path.join(output_dir, f"sample_{one_sample_idx}_edit_item_sas.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[Saved] {save_path}")

        if save_log:
            log_path = os.path.join(output_dir, f"sample_{one_sample_idx}_edit_item_sas.log")
            _write_log(log_path=log_path, prompt=prompt, result=result)
            print(f"[Saved] {log_path}")

    return results if len(results) > 1 else results[0]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("[edit_item_sas] no args provided; defaulting to sample_idx=0, use_target_item=True, edit_span_indices=all")
    if fire is None:
        run()
    else:
        fire.Fire(run)
