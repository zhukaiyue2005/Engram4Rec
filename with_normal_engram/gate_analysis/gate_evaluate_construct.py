import json
import os
import sys

import fire
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig

from common import (
    DEFAULT_DATA_FILE,
    DEFAULT_OUTPUT_DIR,
    PARENT_DIR,
    build_sample_text,
    find_subsequence,
    locate_text_span,
    parse_int_list_arg,
    set_deterministic_mode,
)


os.environ["HUGGINGFACE_HUB_DISABLE_REPO_ID_VALIDATION"] = "1"
os.environ["TORCH_NN_MODULE_USE_DTENSOR"] = "0"
os.environ["USE_DTENSOR"] = "0"
os.environ["ACCELERATE_USE_DTENSOR"] = "0"
os.environ["USE_FLASH_ATTENTION"] = "0"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TENSORBOARD_LOGGING_DIR"] = "./tensorboard_logs"

from huggingface_hub.utils._validators import validate_repo_id


def _dummy_validate_repo_id(repo_id, *args, **kwargs):
    return


validate_repo_id.__code__ = _dummy_validate_repo_id.__code__

engram_dir = os.path.join(PARENT_DIR, "Engram_Insert_code")
if engram_dir not in sys.path:
    sys.path.append(engram_dir)

from engram_demo_v1 import Engram, EngramConfig
from modeling_qwen3 import Qwen3ForCausalLM


set_deterministic_mode(seed=1958)


def detect_engram_layer_ids(checkpoint_dir: str):
    engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
    if not checkpoint_dir or not os.path.isdir(engram_params_dir):
        return []

    layer_ids = []
    for name in os.listdir(engram_params_dir):
        if not (name.startswith("engram_layer_") and name.endswith(".npy")):
            continue
        layer_id_text = name[len("engram_layer_") : -len(".npy")]
        if layer_id_text.isdigit():
            layer_ids.append(int(layer_id_text))
    return sorted(set(layer_ids))


def iter_batches(rows, batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def inference(
    batch_size: int = 8,
    resume_from_checkpoint: str = "",
    base_model: str = "",
    data_file: str = DEFAULT_DATA_FILE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_length: int = 2048,
    engram_layer_ids: str = "auto",
    engram_float32: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "engram_gates.json")
    if os.path.exists(output_file):
        os.remove(output_file)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    if str(engram_layer_ids).strip().lower() == "auto":
        layer_ids = detect_engram_layer_ids(resume_from_checkpoint)
        if not layer_ids:
            raise ValueError(
                "engram_layer_ids=auto 需要 checkpoint 下存在 engram_params/engram_layer_*.npy，"
                f"当前 checkpoint: {resume_from_checkpoint!r}"
            )
        print(f"[Info] 自动识别 Engram 层: {','.join(str(x) for x in layer_ids)}")
    else:
        layer_ids = parse_int_list_arg(engram_layer_ids, "engram_layer_ids")
    if not layer_ids:
        raise ValueError("engram_layer_ids 不能为空")

    device_index = Accelerator().process_index
    device_map = {"": device_index}
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

    def load_engram_params(model, checkpoint_dir: str):
        engram_params_dir = os.path.join(checkpoint_dir, "engram_params")
        if not os.path.exists(engram_params_dir):
            print(f"⚠️ 未找到Engram参数目录: {engram_params_dir}")
            return

        unwrapped_model = model.module if hasattr(model, "module") else model
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
        print(f"\n📌 总计加载 {engram_loaded} 个Engram层参数")

    def convert_engram_params_to_float32(model):
        for name, param in model.named_parameters():
            if "engram" in name.lower():
                param.data = param.data.to(torch.float32)
        return model

    if resume_from_checkpoint:
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
        if engram_float32:
            model = convert_engram_params_to_float32(model)
        load_engram_params(model, resume_from_checkpoint)

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

    raw_dataset = load_dataset("json", data_files={"data": data_file})["data"]
    dataset = [build_sample_text(row, tokenizer) for row in raw_dataset]
    print("✅ 数据加载和预处理完成")
    print(dataset[0])

    processed_count = 0
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    with torch.no_grad(), open(output_file, "w", encoding="utf-8") as f:
        for batch_idx, batch_rows in enumerate(iter_batches(dataset, batch_size)):
            sentences = [row["text"] for row in batch_rows]
            inputs = tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

            engram_gates = outputs.engram_gates
            engram_layer_mapping = outputs.engram_layer_mapping
            batch_size_actual = len(sentences)

            for i in range(batch_size_actual):
                valid_length = int((inputs["attention_mask"][i] == 1).sum().item())
                token_ids = inputs["input_ids"][i][-valid_length:].detach().cpu().tolist()
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                row = batch_rows[i]
                response_ids = tokenizer.encode("### Response:", add_special_tokens=False)
                response_span = find_subsequence(token_ids, response_ids)
                target_range = locate_text_span(tokenizer, token_ids, row["completion"])
                if target_range == [0, 0]:
                    target_range = locate_text_span(tokenizer, token_ids, f"\"{row['target_text']}\"")
                completion_start = int(response_span[1]) if response_span is not None else int(target_range[0])

                sample_result = {
                    "sample_idx": batch_idx * batch_size + i,
                    "full_text": sentences[i],
                    "text": sentences[i],
                    "true_selection": row["completion"],
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "description": row["description"],
                    "history_list": row["historyList"],
                    "item_list": row["itemList"],
                    "historyList": row["historyList"],
                    "itemList": row["itemList"],
                    "history_text": row["history_text"],
                    "target_text": row["target_text"],
                    "tokenized_length": valid_length,
                    "history_range": locate_text_span(tokenizer, token_ids, row["history_text"]),
                    "completion_start": completion_start,
                    "cans_range": [0, 0],
                    "target_range": target_range,
                    "engram_analysis": {},
                }

                for layer_idx, list_idx in engram_layer_mapping.items():
                    gate_tensor = engram_gates[list_idx][i]
                    gate_values = gate_tensor[-valid_length:].detach().cpu().float().numpy()
                    sample_result["engram_analysis"][f"layer_{layer_idx}"] = {
                        "gate_shape": list(gate_values.shape),
                        "gate_values": gate_values.tolist(),
                        "gate_mean": float(gate_values.mean()),
                        "gate_std": float(gate_values.std()),
                        "gate_max": float(gate_values.max()),
                        "gate_min": float(gate_values.min()),
                        "tokens": tokens,
                    }

                f.write(json.dumps(sample_result, ensure_ascii=False) + "\n")
                processed_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"  已处理 {batch_idx + 1}/{total_batches} batches "
                    f"({min((batch_idx + 1) * batch_size, len(dataset))} 个样本)"
                )

    print(f"\n✅ 全部完成！共处理 {processed_count} 个样本")
    print(f"结果保存在: {output_file}")


if __name__ == "__main__":
    fire.Fire(inference)
