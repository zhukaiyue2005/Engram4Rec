#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="3"

MASTER_PORT=21320
BATCH_SIZE=2
CUTOFF_LEN=2048
CKA_TOPK=5
MAX_SAMPLES=1000
# target_item_title: 取 Response 中目标物品标题内容最后一个 token，跳过引号/换行等格式符号。
# target_item_id:    取 JSON 字段 target_item_id 这个数字编码后的最后一个 token。
HIDDEN_STATE_TARGET="target_item_title"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
TEST_FILE="${SCRIPT_DIR}/../../data/Industrial_and_Scientific_dataset/test.jsonl"

# 没有 Engram 的 checkpoint
WITHOUT_ENGRAM_CHECKPOINT="${SCRIPT_DIR}/../../without_engram/output_sft_lr=3e-4/checkpoint-1981"

# 带 Engram 的 checkpoint
WITH_ENGRAM_CHECKPOINT="$SCRIPT_DIR/../output_sft_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4/checkpoint-1988"
ENGRAM_LAYER_IDS="6,13,20"

mkdir -p cka_results

LOG_FILE="CKA_evaluate_with_engram_vs_without_engram.log"
RESULT_JSON="cka_results/CKA_evaluate_with_engram_vs_without_engram.json"
PLOT_PATH="cka_results/CKA_evaluate_with_engram_vs_without_engram.png"

torchrun --nproc_per_node 1 --master_port="$MASTER_PORT" \
    cka_evaluate.py \
    --batch_size "$BATCH_SIZE" \
    --base_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --cutoff_len "$CUTOFF_LEN" \
    --resume_from_checkpoint_with_engram "$WITH_ENGRAM_CHECKPOINT" \
    --resume_from_checkpoint_without_engram "$WITHOUT_ENGRAM_CHECKPOINT" \
    --engram_layer_ids "$ENGRAM_LAYER_IDS" \
    --engram_float32 True \
    --k "$CKA_TOPK" \
    --max_samples "$MAX_SAMPLES" \
    --hidden_state_target "$HIDDEN_STATE_TARGET" \
    --result_json "$RESULT_JSON" \
    --plot_path "$PLOT_PATH" \
    > "$LOG_FILE" 2>&1
