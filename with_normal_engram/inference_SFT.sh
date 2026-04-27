#!/usr/bin/env bash
set -euo pipefail

# The number of processes can only be one for inference.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output_sft_engram_lr=1e-4_layer=5,10,15/final_checkpoint_sft}"
MASTER_PORT="${MASTER_PORT:-51751}"
LOG_FILE="${LOG_FILE:-eval_sft_normal_engram.log}"

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "[Error] checkpoint not found: $CHECKPOINT_PATH"
    echo "        Set CHECKPOINT_PATH=./output_sft_engram_lr=1e-4_layer=5,10,15/checkpoint-XXXX or run sft.sh first."
    exit 1
fi

torchrun --nproc_per_node 1 --master_port="$MASTER_PORT" \
        inference.py \
        --batch_size "${BATCH_SIZE:-2}" \
        --base_model "$BASE_MODEL" \
        --resume_from_checkpoint "$CHECKPOINT_PATH" \
        --test_file "${TEST_FILE:-../data/Industrial_and_Scientific_dataset/valid.jsonl}" \
        --info_file "${INFO_FILE:-../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt}" \
        --eval_topk "${EVAL_TOPK:-1,3,5,10}" \
        --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
        --engram_layer_ids "${ENGRAM_LAYER_IDS:-5,10,15}" \
        --engram_float32 "${ENGRAM_FLOAT32:-True}" \
        --print_batch_output "${PRINT_BATCH_OUTPUT:-True}" \
        --print_prompt "${PRINT_PROMPT:-False}" \
        --print_topn "${PRINT_TOPN:-10}" \
        2>&1 | tee "$LOG_FILE"
