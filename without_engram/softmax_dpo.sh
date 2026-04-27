#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-./sft_checkpoint}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-11794}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"

if [[ ! -d "$SFT_CHECKPOINT" ]]; then
    echo "[Error] SFT checkpoint not found: $SFT_CHECKPOINT"
    echo "        Set SFT_CHECKPOINT to an existing LoRA checkpoint before running softmax-DPO."
    exit 1
fi

torchrun --nproc_per_node "$NPROC_PER_NODE" --master_port="$MASTER_PORT" softmax_dpo.py \
    --model_name "$BASE_MODEL" \
    --train_file "../data/Industrial_and_Scientific_dataset/train.jsonl" \
    --eval_file "../data/Industrial_and_Scientific_dataset/valid.jsonl" \
    --info_file "../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt" \
    --resume_from_checkpoint "$SFT_CHECKPOINT" \
    --batch_size "${BATCH_SIZE:-2}" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-32}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" \
    --eval_step "${EVAL_STEP:-0.2}" \
    --beta "${DPO_BETA:-0.1}" \
    --neg_num "${NEG_NUM:-3}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
    --logging_dir "${LOGGING_DIR:-./logs}" \
    --output_dir "${OUTPUT_DIR:-./softmax_dpo_output}" \
    --wandb_project "${WANDB_PROJECT:-Engram4Rec}" \
    --wandb_name "${WANDB_NAME:-without_engram_softmax_dpo}" \
    > "${LOG_FILE:-softmax_dpo.log}" 2>&1
