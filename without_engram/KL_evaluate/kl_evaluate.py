import os
from typing import Dict, List

from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3ForCausalLM

from kl_evaluate_batch import evaluate, save_result_json

os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_HF_HOME = os.path.join(SCRIPT_DIR, ".hf_cache")
os.environ["HF_HOME"] = LOCAL_HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(LOCAL_HF_HOME, "datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)

from huggingface_hub.utils._validators import validate_repo_id


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__

from accelerate import Accelerator

try:
    import fire
except ModuleNotFoundError:
    fire = None


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

    if resume_from_checkpoint:
        print(f"📌 加载 without_engram LoRA checkpoint: {resume_from_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)

    model.eval()
    return model


def inference(
    batch_size: int = 1,
    resume_from_checkpoint: str = "",
    base_model: str = "Qwen3-1.7B",
    test_file: str = "../../data/Industrial_and_Scientific_dataset/test.jsonl",
    max_length: int = 2048,
    method: str = "exact_kl",
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
