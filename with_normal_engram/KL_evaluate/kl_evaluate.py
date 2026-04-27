import os
import re
import sys
from typing import Dict, List

from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig

from kl_evaluate_batch import evaluate, save_result_json

os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(PROJECT_DIR), "data")
DEFAULT_TEST_FILE = os.path.join(DATA_ROOT, "Industrial_and_Scientific_dataset", "test.jsonl")
LOCAL_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_cache")
os.environ["HF_HOME"] = LOCAL_HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(LOCAL_HF_HOME, "datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from huggingface_hub.utils._validators import validate_repo_id


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__

ENGRAM_DIR = os.path.join(PROJECT_DIR, "Engram_Insert_code")
if ENGRAM_DIR not in sys.path:
    sys.path.append(ENGRAM_DIR)

from accelerate import Accelerator
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


def _infer_engram_layer_ids(resume_from_checkpoint: str, engram_layer_ids="") -> List[int]:
    explicit_ids = _parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if explicit_ids:
        print(f"📌 使用显式传入的 engram_layer_ids: {explicit_ids}")
        return explicit_ids

    if not resume_from_checkpoint:
        raise ValueError("未提供 resume_from_checkpoint，且未显式传入 engram_layer_ids，无法推断 Engram 层")

    path_parts = [p for p in os.path.normpath(resume_from_checkpoint).split(os.sep) if p]
    for part in reversed(path_parts):
        match = re.search(r"(?:^|_)(?:layer|layers|engram_layer_ids?)=([0-9,，、-]+)", part)
        if match:
            raw = match.group(1).replace("，", ",").replace("、", ",").replace("-", ",")
            inferred = _parse_int_list_arg(raw, "checkpoint_layer_ids")
            if inferred:
                print(f"📌 从 checkpoint 名称推断 engram 层: {inferred} <- {part}")
                return inferred

    engram_params_dir = os.path.join(resume_from_checkpoint, "engram_params")
    if os.path.isdir(engram_params_dir):
        inferred = []
        for filename in os.listdir(engram_params_dir):
            match = re.fullmatch(r"engram_layer_(\d+)\.npy", filename)
            if match:
                inferred.append(int(match.group(1)))
        inferred = sorted(set(inferred))
        if inferred:
            print(f"📌 从 engram_params 文件推断 engram 层: {inferred}")
            return inferred

    raise ValueError(f"无法从 checkpoint 名称或参数文件推断 engram_layer_ids: {resume_from_checkpoint}")


def _normalize_completion(row: dict) -> str:
    completion = str(row.get("completion", "")).rstrip()
    if completion:
        return completion + "\n"
    target = str(row.get("target_item_title", "")).strip()
    if target:
        return f"\"{target}\"\n"
    return ""


def _build_eval_dataloader(
    tokenizer,
    test_file: str,
    batch_size: int,
    max_samples: int = 0,
    sample_seed: int = 42,
):
    dataset = load_dataset("json", data_files={"test": test_file})["test"]
    dataset_size = len(dataset)
    if max_samples and max_samples > 0:
        sample_count = min(max_samples, dataset_size)
        dataset = dataset.shuffle(seed=sample_seed).select(range(sample_count))
        print(f"[Info] 从 test 数据随机抽样 {sample_count}/{dataset_size} 条, seed={sample_seed}")

    def _map_row(row: dict) -> dict:
        prompt = str(row.get("prompt", ""))
        completion = _normalize_completion(row)
        return {
            "prompt": prompt,
            "completion": completion,
            "sentence": prompt + completion,
            "prompt_len": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
        }

    dataset = dataset.map(_map_row)

    def _collate_fn(batch: List[dict]) -> Dict[str, List]:
        return {
            "sentences": [x["sentence"] for x in batch],
            "prompts": [x["prompt"] for x in batch],
            "completions": [x["completion"] for x in batch],
            "prompt_lens": [x["prompt_len"] for x in batch],
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    return dataloader, len(dataset), dataset_size


def _load_model(
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

    layer_ids = _infer_engram_layer_ids(resume_from_checkpoint, engram_layer_ids)
    engram_config = EngramConfig()
    engram_config.layer_ids = layer_ids

    model = Qwen3ForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=bnb_config,
        local_files_only=True,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model)
    model.attach_engram(engram_config)
    model._reset_engram_parameters()

    def _load_engram_params(target_model, checkpoint_dir: str):
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            print(f"⚠️ 未找到Engram参数目录: {engram_params_dir}")
            return

        unwrapped_model = target_model.module if hasattr(target_model, "module") else target_model
        engram_loaded = 0
        for _, module in unwrapped_model.named_modules():
            if isinstance(module, Engram):
                layer_id = module.layer_id
                param_path = os.path.join(engram_params_dir, f"engram_layer_{layer_id}.npy")
                if os.path.exists(param_path):
                    module.load_all_params(param_path)
                    engram_loaded += 1
                else:
                    print(f"⚠️ Engram层 {layer_id} 参数文件缺失: {param_path}")
        print(f"📌 总计加载 {engram_loaded} 个Engram层参数")

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        if engram_float32:
            for name, param in model.named_parameters():
                if "engram" in name.lower():
                    param.data = param.data.to(torch.float32)
        _load_engram_params(model, resume_from_checkpoint)

    model.eval()
    return model


def inference(
    batch_size: int = 1,
    resume_from_checkpoint: str = "",
    base_model: str = "Qwen3-1.7B",
    test_file: str = DEFAULT_TEST_FILE,
    max_length: int = 2048,
    method: str = "exact_kl",
    engram_layer_ids: str = "",
    engram_float32: bool = False,
    save_json: str = "",
    mc_samples: int = 10000,
    position_chunk_size: int = 8,
    max_samples: int = 1000,
    sample_seed: int = 42,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    dataloader, eval_sample_count, dataset_size = _build_eval_dataloader(
        tokenizer,
        test_file=test_file,
        batch_size=batch_size,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )
    model = _load_model(
        base_model=base_model,
        resume_from_checkpoint=resume_from_checkpoint,
        engram_layer_ids=engram_layer_ids,
        engram_float32=engram_float32,
    )

    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        max_length=max_length,
        method=method,
        resume_from_checkpoint=resume_from_checkpoint,
        mc_samples=mc_samples,
        position_chunk_size=position_chunk_size,
    )

    print(f"=== {method} layer averages ===")
    for layer_idx in sorted(result):
        print(f"layer {layer_idx:02d}: {result[layer_idx]:.8f}")

    if save_json:
        os.makedirs(os.path.dirname(os.path.abspath(save_json)), exist_ok=True)
        save_result_json(
            result,
            save_json,
            metadata={
                "resume_from_checkpoint": resume_from_checkpoint,
                "base_model": base_model,
                "test_file": test_file,
                "method": method,
                "engram_layer_ids": _infer_engram_layer_ids(resume_from_checkpoint, engram_layer_ids),
                "max_length": max_length,
                "batch_size": batch_size,
                "mc_samples": mc_samples,
                "position_chunk_size": position_chunk_size,
                "max_samples": max_samples,
                "sample_seed": sample_seed,
                "eval_sample_count": eval_sample_count,
                "dataset_size": dataset_size,
            },
        )
        print(f"[Info] 结果已保存到: {save_json}")

    return result


if __name__ == "__main__":
    if fire is not None:
        fire.Fire(inference)
