#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1

torchrun --nproc_per_node "${NPROC_PER_NODE:-2}" --master_port="${MASTER_PORT:-31883}" sft.py \
    --model_name "${BASE_MODEL:-Qwen3-1.7B}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --dataset lastfm \
    --train_file "${TRAIN_FILE:-../data/Industrial_and_Scientific_dataset/train.jsonl}" \
    --eval_file "${EVAL_FILE:-../data/Industrial_and_Scientific_dataset/valid.jsonl}" \
    --logging_dir "./logs" \
    --output_dir "${OUTPUT_DIR:-./output_sft_layer=6,13,20_lr=1e-4}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --lora_lr "${LORA_LR:-1e-4}" \
    --engram_lr "${ENGRAM_LR:-1e-4}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}" \
    --eval_step "${EVAL_STEP:-0.05}" \
    --wandb_project "${WANDB_PROJECT:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank}" \
    --wandb_name "${WANDB_NAME:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank__with_item_engram__sft}" \
    > "${LOG_FILE:-sft_engram_layer=6,13,20_lr=1e-4.log}" 2>&1
