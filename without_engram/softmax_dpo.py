import math
import os
import random
import sys

import bitsandbytes as bnb
import fire
import peft
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from huggingface_hub.utils._validators import validate_repo_id
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import Qwen3ForCausalLM
from transformers import get_cosine_schedule_with_warmup


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sft_wandb import setup_wandb_for_sft


os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__

# 兼容新版 peft：老版 DPOTrainer 仍引用 prepare_model_for_int8_training。
if not hasattr(peft, "prepare_model_for_int8_training"):
    peft.prepare_model_for_int8_training = prepare_model_for_kbit_training

from softmax_dpo_trainer import DPOTrainer


def _is_main_process() -> bool:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank == 0 and local_rank == 0


def _parse_prompt_column(prompt_col):
    if isinstance(prompt_col, list):
        for item in prompt_col:
            if isinstance(item, dict) and item.get("role") == "user":
                return item.get("content", "")
        return ""
    if isinstance(prompt_col, str):
        return prompt_col
    return ""


def _resolve_local_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(CURRENT_DIR, path_value))


def _normalize_completion(data_point):
    true_selection = data_point.get("target_item_title", "")
    if not true_selection:
        return str(data_point.get("completion", "")).rstrip("\n")
    return f"\"{true_selection}\""


def _load_all_titles_from_info(info_file: str):
    titles = []
    with open(info_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            title = parts[0].strip()
            if title:
                titles.append(title)
    unique_titles = sorted(set(titles))
    if not unique_titles:
        raise ValueError(f"商品信息文件中未解析到任何标题: {info_file}")
    return unique_titles


def _to_prompt_completion(data_point):
    return {
        "prompt": _parse_prompt_column(data_point.get("prompt", "")),
        "chosen": _normalize_completion(data_point).rstrip("\n"),
        "target_item_title": data_point.get("target_item_title", "").strip(),
    }


def _build_softmax_dpo_dataset(split_dataset, all_titles, neg_num, seed):
    rng = random.Random(seed)
    records = []
    for item in split_dataset:
        chosen_title = item["target_item_title"].strip().strip("\"")
        candidates = [title for title in all_titles if title != chosen_title]
        if not candidates:
            continue
        sample_size = min(neg_num, len(candidates))
        sample_negs = rng.sample(candidates, sample_size)
        row = {
            "prompt": item["prompt"],
            "chosen": item["chosen"],
        }
        for idx, rejected_title in enumerate(sample_negs, start=1):
            row[f"rejected{idx}"] = f"\"{rejected_title}\""
        for idx in range(sample_size + 1, neg_num + 1):
            # keep field count fixed for the custom trainer
            filler = candidates[(idx - sample_size - 1) % len(candidates)]
            row[f"rejected{idx}"] = f"\"{filler}\""
        records.append(row)
    if not records:
        raise ValueError("未构造出任何 S-DPO 样本，请检查训练数据。")
    return Dataset.from_list(records)


def _log_softmax_dpo_sample(dataset: Dataset, split_name: str, neg_num: int):
    if len(dataset) == 0:
        print(f"⚠️ {split_name} 数据集为空，无法打印样本预览。")
        return

    sample = dataset[0]
    print(f"\n=== {split_name} S-DPO样本预览 ===")
    print(f"{split_name} 样本数: {len(dataset)}")
    print("Prompt:")
    print(sample["prompt"])
    print("\nChosen（完整正确答案）:")
    print(sample["chosen"])
    print("\nRejected（抽样负样本）:")
    for idx in range(1, neg_num + 1):
        key = f"rejected{idx}"
        print(f"{key}: {sample.get(key, '')}")
    print(f"=== {split_name} S-DPO样本预览结束 ===\n")


def _load_model(model_name, device_map, bnb_config):
    model = Qwen3ForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        quantization_config=bnb_config,
        local_files_only=True,
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model)


def train(
    output_dir="",
    logging_dir="",
    model_name="",
    train_file="../data/Industrial_and_Scientific_dataset/train.jsonl",
    eval_file="../data/Industrial_and_Scientific_dataset/valid.jsonl",
    info_file="../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt",
    resume_from_checkpoint: str = "",
    wandb_project: str = "",
    wandb_name: str = "",
    beta: float = 0.1,
    neg_num: int = 3,
    batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 1024,
    eval_step=0.1,
):
    setup_wandb_for_sft(wandb_project=wandb_project, wandb_name=wandb_name)

    if not model_name or model_name == ".":
        print("错误: model_name参数为空或无效")
        print("请检查shell脚本中的--model_name参数设置")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.bos_token_id = 151643
    tokenizer.eos_token_id = 151645
    tokenizer.pad_token_id = 151643

    train_file = _resolve_local_path(train_file)
    eval_file = _resolve_local_path(eval_file)
    info_file = _resolve_local_path(info_file)

    train_raw = load_dataset("json", data_files={"train": train_file})["train"]
    train_raw = train_raw.shuffle(seed=42).map(_to_prompt_completion)

    val_raw = load_dataset("json", data_files={"validation": eval_file})["validation"]
    val_raw = val_raw.shuffle(seed=42).map(_to_prompt_completion)

    all_titles = _load_all_titles_from_info(info_file)
    train_data = _build_softmax_dpo_dataset(train_raw, all_titles, neg_num=neg_num, seed=42)
    val_data = _build_softmax_dpo_dataset(val_raw, all_titles, neg_num=neg_num, seed=43)
    if _is_main_process():
        _log_softmax_dpo_sample(train_data, "Train", neg_num)
        _log_softmax_dpo_sample(val_data, "Validation", neg_num)

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    base_model = _load_model(model_name, device_map=device_map, bnb_config=bnb_config)
    model_ref = _load_model(model_name, device_map=device_map, bnb_config=bnb_config)

    if not resume_from_checkpoint:
        raise ValueError("softmax_dpo.py 需要传入 --resume_from_checkpoint，作为 SFT 初始策略模型。")

    base_model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
    reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint, is_trainable=False)

    world_size = Accelerator().num_processes
    global_batch_size = batch_size * gradient_accumulation_steps * world_size
    steps_per_epoch = math.ceil(len(train_data) / global_batch_size)
    total_train_steps = steps_per_epoch * num_train_epochs
    if 0 < float(eval_step) <= 1:
        eval_interval_steps = round(steps_per_epoch * float(eval_step))
    else:
        eval_interval_steps = round(float(eval_step))
    eval_steps = max(1, eval_interval_steps)

    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for name, param in base_model.named_parameters()
                if param.requires_grad and "lora_" in name.lower()
            ],
            "lr": learning_rate,
            "weight_decay": 0,
        }
    ]

    optimizer = bnb.optim.PagedAdamW32bit(
        optimizer_grouped_parameters,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_train_steps * 0.05)),
        num_training_steps=max(1, total_train_steps),
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=100,
        eval_strategy="steps",
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to="wandb" if wandb_project else "none",
        run_name=wandb_name,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
        optimizers=(optimizer, lr_scheduler),
    )

    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    fire.Fire(train)
