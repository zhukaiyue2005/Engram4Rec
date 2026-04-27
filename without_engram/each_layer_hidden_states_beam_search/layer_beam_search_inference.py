import argparse
import json
import os
import random
import sys
import time
from typing import List, Optional

# Work around a torchrun/static-TLS loader issue by initializing sklearn
# before torch/accelerate/peft/transformers pull in competing native libs.
try:
    from sklearn.metrics import roc_curve as _sklearn_roc_curve  # noqa: F401
except ModuleNotFoundError:
    _sklearn_roc_curve = None

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig, LogitsProcessorList, Qwen3ForCausalLM


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from inference import (  # noqa: E402
    ConstrainedLogitsProcessor,
    _build_constraint_dict,
    _compute_metrics,
    _format_seconds,
    _get_hash,
    _normalize_target,
    _parse_int_list_arg,
)


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值: {value!r}")


def _get_base_causal_lm(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _parse_layer_spec(layer_spec: str) -> List[int]:
    layers = []
    normalized = str(layer_spec).replace("，", ",").replace("、", ",").replace(" ", ",")
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = [x.strip() for x in part.split("-", 1)]
            start, end = int(left), int(right)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(part))
    return layers


def _resolve_target_layers(
    target_layers: str,
    target_layer: Optional[int],
    layer_index_base: int,
    num_layers: int,
) -> List[int]:
    if layer_index_base not in (0, 1):
        raise ValueError(f"layer_index_base 只能是 0 或 1，当前为 {layer_index_base}")

    if str(target_layers).strip().lower() in {"all", "*"}:
        requested_layers = list(range(layer_index_base, layer_index_base + num_layers))
    else:
        requested_layers = _parse_layer_spec(target_layers) if target_layers else []
    if not requested_layers and target_layer is not None:
        requested_layers = [target_layer]
    if not requested_layers:
        requested_layers = [0 if layer_index_base == 0 else 1]

    zero_based_layers = []
    for layer_id in requested_layers:
        zero_based = layer_id - layer_index_base
        if zero_based < 0 or zero_based >= num_layers:
            valid = f"0..{num_layers - 1}" if layer_index_base == 0 else f"1..{num_layers}"
            raise ValueError(
                f"层编号 {layer_id} 越界。当前 layer_index_base={layer_index_base}，有效范围是 {valid}"
            )
        zero_based_layers.append(zero_based)

    seen = set()
    out = []
    for layer_id in zero_based_layers:
        if layer_id in seen:
            continue
        seen.add(layer_id)
        out.append(layer_id)
    return out


def _capture_full_layers(model):
    base_model = _get_base_causal_lm(model)
    return base_model, list(base_model.model.layers)


def _truncate_model_to_layer(base_model, full_layers, target_layer: int):
    keep_layers = target_layer + 1
    base_model.model.layers = torch.nn.ModuleList(list(full_layers[:keep_layers]))
    base_model.config.num_hidden_layers = keep_layers
    base_model.model.config.num_hidden_layers = keep_layers
    return base_model


def _format_layer_label(zero_based_layer: int, layer_index_base: int) -> int:
    return zero_based_layer + layer_index_base


def _resolve_save_json_path(save_json: str, layer_label: int, num_layers_to_run: int) -> str:
    if not save_json:
        return ""
    if "{layer}" in save_json:
        return save_json.format(layer=layer_label)
    if num_layers_to_run <= 1:
        return save_json

    root, ext = os.path.splitext(save_json)
    if not ext:
        ext = ".json"
    return f"{root}_layer{layer_label}{ext}"


def _decode_completions(tokenizer, completion_ids, base_model: str) -> List[str]:
    if "llama" in base_model.lower():
        decoded = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    else:
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return [x.split("Response:\n")[-1].strip().strip('"').strip() for x in decoded]


def inference(
    batch_size: int = 4,
    resume_from_checkpoint: str = "",
    base_model: str = "",
    test_file: str = "../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file: str = "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    target_layer: Optional[int] = None,
    target_layers: str = "",
    layer_index_base: int = 0,
    max_new_tokens: int = 256,
    eval_topk: str = "1,3,5,10",
    length_penalty: float = 0.0,
    save_json: str = "",
    save_metrics_jsonl: str = "each_layer_hidden_states_beam_search/layer_metrics.jsonl",
    max_samples: int = 1000,
    sample_seed: int = 42,
    print_batch_output: bool = True,
    print_prompt: bool = True,
    print_topn: int = 10,
    prompt_preview_chars: int = 0,
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
        device_map=device_map,
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)

    base_causal_lm, full_layers = _capture_full_layers(model)
    original_num_layers = len(full_layers)
    target_layer_ids = _resolve_target_layers(
        target_layers=target_layers,
        target_layer=target_layer,
        layer_index_base=layer_index_base,
        num_layers=original_num_layers,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    hash_dict = _build_constraint_dict(tokenizer, base_model, info_file)

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        return hash_dict.get(_get_hash(input_ids), [])

    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    original_total = len(dataset)
    if max_samples and max_samples > 0 and original_total > max_samples:
        rng = random.Random(sample_seed)
        sampled_indices = sorted(rng.sample(range(original_total), max_samples))
        dataset = dataset.select(sampled_indices)
        print(
            f"📌 从数据集中采样 {len(dataset)} 条用于 beam search；"
            f"原始样本数={original_total}；"
            f"sample_seed={sample_seed}",
            flush=True,
        )
    elif max_samples and max_samples > 0:
        print(f"📌 数据集样本数={len(dataset)}，不超过 max_samples={max_samples}，全部使用", flush=True)
    prompts = [row["prompt"] for row in dataset]
    targets = [_normalize_target(row) for row in dataset]
    topk = sorted(set(_parse_int_list_arg(eval_topk, "eval_topk")))
    if not topk:
        raise ValueError(f"eval_topk不能为空，当前值: {eval_topk}")
    if min(topk) <= 0:
        raise ValueError(f"eval_topk中的K必须为正整数，当前值: {eval_topk}")
    num_beams = max(topk)

    total = len(prompts)
    steps = (total + batch_size - 1) // batch_size

    if save_metrics_jsonl:
        metrics_dir = os.path.dirname(save_metrics_jsonl)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)

    print(
        f"📌 层编号基准 layer_index_base={layer_index_base}；"
        f"将运行层={[_format_layer_label(x, layer_index_base) for x in target_layer_ids]}；"
        f"0-based内部层={target_layer_ids}；"
        f"embedding hidden state 不参与推理",
        flush=True,
    )

    for zero_based_layer in target_layer_ids:
        layer_label = _format_layer_label(zero_based_layer, layer_index_base)
        truncated_model = _truncate_model_to_layer(
            base_causal_lm,
            full_layers=full_layers,
            target_layer=zero_based_layer,
        )
        print(
            f"\n📌 开始 layer={layer_label} (0-based={zero_based_layer})；"
            f"保留层数={truncated_model.config.num_hidden_layers}/{original_num_layers}；"
            f"输出路径=decoder_layer_output -> model.norm -> lm_head",
            flush=True,
        )

        all_preds: List[List[str]] = []
        start_time = time.time()

        for step in range(steps):
            s = step * batch_size
            e = min(total, s + batch_size)
            batch_prompts = prompts[s:e]

            model_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            ).to(model.device)
            prompt_len = model_inputs["input_ids"].shape[1]

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
            decoded = _decode_completions(tokenizer, completion_ids, base_model)
            batch_preds: List[List[str]] = []
            for i in range(0, len(decoded), num_beams):
                one_pred = decoded[i : i + num_beams]
                all_preds.append(one_pred)
                batch_preds.append(one_pred)

            if print_batch_output:
                current_topn = num_beams if print_topn <= 0 else min(max(1, print_topn), num_beams)
                print(
                    f"\n=== Layer {layer_label} | Batch {step + 1}/{steps} | samples {s}-{e - 1} ===",
                    flush=True,
                )
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
                f"[Layer {layer_label}] [Progress] {done_steps}/{steps} ({progress:.2f}%) | "
                f"elapsed: {_format_seconds(elapsed)} | eta: {_format_seconds(eta)}",
                flush=True,
            )

        metrics = _compute_metrics(all_preds, targets, topk)
        print(f"=== Layer {layer_label} Metrics ===", flush=True)
        for k in topk:
            print(f"Hit@{k}: {metrics[f'Hit@{k}']:.6f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.6f}", flush=True)

        if save_metrics_jsonl:
            metric_row = {
                "layer": layer_label,
                "zero_based_layer": zero_based_layer,
                "layer_index_base": layer_index_base,
                "truncated_num_layers": zero_based_layer + 1,
                "num_original_layers": original_num_layers,
                "num_samples": total,
                "num_beams": num_beams,
                "topk": topk,
                "metrics": metrics,
                "elapsed_seconds": time.time() - start_time,
            }
            with open(save_metrics_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric_row, ensure_ascii=False) + "\n")
            print(f"saved layer metrics to: {save_metrics_jsonl}", flush=True)

        layer_save_json = _resolve_save_json_path(save_json, layer_label, len(target_layer_ids))
        if layer_save_json:
            save_dir = os.path.dirname(layer_save_json)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            rows = []
            for i, row in enumerate(dataset):
                one = dict(row)
                one["target_title"] = targets[i]
                one["predict"] = all_preds[i]
                one["target_layer"] = layer_label
                one["zero_based_layer"] = zero_based_layer
                one["layer_index_base"] = layer_index_base
                one["truncated_num_layers"] = zero_based_layer + 1
                rows.append(one)
            with open(layer_save_json, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
            print(f"saved predictions to: {layer_save_json}", flush=True)


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Use selected Qwen decoder layer outputs for constrained beam-search inference."
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--test_file", type=str, default="../data/Industrial_and_Scientific_dataset/valid.jsonl")
    parser.add_argument(
        "--info_file",
        type=str,
        default="../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    )
    parser.add_argument("--target_layer", type=int, default=None, help="兼容旧参数：单个目标层。")
    parser.add_argument(
        "--target_layers",
        type=str,
        default="",
        help='多个目标层，例如 "0,5,10"、"0-3,10" 或 "all"。优先级高于 --target_layer。',
    )
    parser.add_argument(
        "--layer_index_base",
        type=int,
        default=0,
        choices=[0, 1],
        help="层编号基准。0 表示第一层为 0；1 表示第一层为 1。脚本内部会统一转换为 0-based。",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--eval_topk", type=str, default="1,3,5,10")
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument(
        "--save_metrics_jsonl",
        type=str,
        default="each_layer_hidden_states_beam_search/layer_metrics.jsonl",
        help="每层结束后立即追加保存 Hit/NDCG 指标；传空字符串可关闭。",
    )
    parser.add_argument("--max_samples", type=int, default=1000, help="每次最多采样多少条数据；<=0 表示全量。")
    parser.add_argument("--sample_seed", type=int, default=42, help="max_samples 采样随机种子。")
    parser.add_argument("--print_batch_output", type=_str_to_bool, default=True)
    parser.add_argument("--print_prompt", type=_str_to_bool, default=True)
    parser.add_argument("--print_topn", type=int, default=10)
    parser.add_argument("--prompt_preview_chars", type=int, default=0)
    return parser


if __name__ == "__main__":
    inference(**vars(_build_arg_parser().parse_args()))
