#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2}"
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node "${NPROC_PER_NODE:-2}" --master_port="${MASTER_PORT:-34171}" sft.py \
    --model_name "${BASE_MODEL:-Qwen3-1.7B}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --train_file "${TRAIN_FILE:-${DATA_DIR}/Industrial_and_Scientific_dataset/train.jsonl}" \
    --eval_file "${EVAL_FILE:-${DATA_DIR}/Industrial_and_Scientific_dataset/valid.jsonl}" \
    --logging_dir "${LOGGING_DIR:-./logs}" \
    --output_dir "${OUTPUT_DIR:-./output_sft_engram_lr=1e-4_layer=5,10,15}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --lora_lr "${LORA_LR:-1e-4}" \
    --engram_lr "${ENGRAM_LR:-1e-4}" \
    --engram_layer_ids "${ENGRAM_LAYER_IDS:-5,10,15}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}" \
    --eval_step "${EVAL_STEP:-0.1}" \
    --wandb_project "${WANDB_PROJECT:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank}" \
    --wandb_name "${WANDB_NAME:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank__with_normal_engram__sft}" \
    > "${LOG_FILE:-sft_engram_lr=1e-4_layer=5,10,15.log}" 2>&1
