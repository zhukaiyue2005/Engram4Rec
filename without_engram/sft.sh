#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Position the number of processes specified after the --nproc_per_node flag
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,6}"
export PYTHONUNBUFFERED=1
    
torchrun --nproc_per_node "${NPROC_PER_NODE:-2}" --master_port="${MASTER_PORT:-21792}" sft.py \
    --model_name "${BASE_MODEL:-Qwen3-1.7B}" \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_dir "./logs" \
    --output_dir "./output_sft_lr=4e-4" \
    --learning_rate 4e-4 \
    --num_train_epochs 10 \
    --eval_step 0.1 \
    --wandb_project "${WANDB_PROJECT:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank}" \
    --wandb_name "${WANDB_NAME:-Qwen3_ReRe_1.7B__Industrial_and_Scientific_full_rank__without_engram__sft}" > sft_lr=4e-4.log 2>&1

