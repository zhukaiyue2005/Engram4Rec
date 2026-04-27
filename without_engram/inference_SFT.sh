#!/usr/bin/env bash
set -euo pipefail

# The number of processes can only be one for inference
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output_sft_lr=4e-4/final_checkpoint_sft}"
MASTER_PORT="${MASTER_PORT:-37990}"
LOG_FILE="${LOG_FILE:-eval_sft_without_engram.log}"

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "[Error] checkpoint not found: $CHECKPOINT_PATH"
    echo "        Set CHECKPOINT_PATH=./output_sft_lr=4e-4/checkpoint-XXXX or run sft.sh first."
    exit 1
fi

torchrun --nproc_per_node 1 --master_port="$MASTER_PORT" \
        inference.py \
        --batch_size 4 \
        --base_model "$BASE_MODEL" \
        --resume_from_checkpoint "$CHECKPOINT_PATH" \
        --print_batch_output True \
        --print_prompt True \
        --print_topn 10 \
        --prompt_preview_chars 0 \
        2>&1 | tee "$LOG_FILE"
