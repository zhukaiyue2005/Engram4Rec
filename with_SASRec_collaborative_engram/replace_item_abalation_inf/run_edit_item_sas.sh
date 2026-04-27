#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output_sft_sasrec_item_spans_layers=6,13,20/final_checkpoint_sft}"

python replace_item_abalation_inf/edit_item_sas_inference.py \
    --base_model "$BASE_MODEL" \
    --resume_from_checkpoint "$CHECKPOINT_PATH" \
    --test_file "${TEST_FILE:-../data/Industrial_and_Scientific_dataset/test.jsonl}" \
    --info_file "${INFO_FILE:-../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt}" \
    --sasrec_checkpoint_path "${SASREC_CHECKPOINT_PATH:-./SAS-checkpoints/sasrec_best.pt}" \
    --replace_plan_file "${REPLACE_PLAN_FILE:-replace_item_abalation_inf/replace_item_ids_plan.json}" \
    --output_dir "${OUTPUT_DIR:-replace_item_abalation_inf/results}" \
    --sample_idx "${SAMPLE_IDX:-0}" \
    --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --engram_layer_ids "${ENGRAM_LAYER_IDS:-6,13,20}" \
    --engram_float32 "${ENGRAM_FLOAT32:-True}"
