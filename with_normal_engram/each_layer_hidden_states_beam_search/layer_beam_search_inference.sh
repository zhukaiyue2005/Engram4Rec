#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

TARGET_LAYERS="${TARGET_LAYERS:-27 26 24 22 20 18 16}"
LAYER_INDEX_BASE="${LAYER_INDEX_BASE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
EVAL_TOPK="${EVAL_TOPK:-1,3,5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-21752}"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output_sft_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4/checkpoint-1988}"
TEST_FILE="${TEST_FILE:-${PROJECT_DIR}/../data/Industrial_and_Scientific_dataset/valid.jsonl}"
INFO_FILE="${INFO_FILE:-${PROJECT_DIR}/../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt}"
METRICS_FILE="${METRICS_FILE:-${SCRIPT_DIR}/layer_metrics.jsonl}"

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "[Error] checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

mkdir -p "$SCRIPT_DIR"

LOG_SAFE_TOPK="${EVAL_TOPK//,/+}"

print_topn="$(python - "$EVAL_TOPK" <<'PY'
import sys

vals = []
for part in sys.argv[1].replace("，", ",").split(","):
    part = part.strip()
    if part:
        vals.append(int(part))
print(max(vals))
PY
)"

declare -a LAYERS=()
# shellcheck disable=SC2206
INPUT_LAYERS=(${TARGET_LAYERS//,/ })
for layer in "${INPUT_LAYERS[@]}"; do
    if [[ "$layer" =~ ^[0-9]+$ ]]; then
        LAYERS+=("$layer")
    else
        echo "[Warn] invalid layer id: $layer"
    fi
done

if [[ "${#LAYERS[@]}" -eq 0 ]]; then
    echo "[Error] no valid target layers selected."
    exit 1
fi

idx=0
for layer in "${LAYERS[@]}"; do
    master_port=$((MASTER_PORT_BASE + idx))
    log_file="${SCRIPT_DIR}/layer_${layer}_beam${LOG_SAFE_TOPK}_sample${MAX_SAMPLES}.log"
    pred_file="${SCRIPT_DIR}/layer_${layer}_predictions.json"

    echo "[Run] layer=${layer}, beam_topk=${EVAL_TOPK}, max_samples=${MAX_SAMPLES} -> $log_file"

    if ! {
        echo "[Run] layer=${layer}, beam_topk=${EVAL_TOPK}, max_samples=${MAX_SAMPLES}"
        echo "[Start] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[CWD] $(pwd)"
        echo "[Log] $log_file"
        echo "[Predictions] $pred_file"
        echo "[Metrics] $METRICS_FILE"
        torchrun --nproc_per_node 1 --master_port="$master_port" \
            "${SCRIPT_DIR}/layer_beam_search_inference.py" \
            --batch_size "$BATCH_SIZE" \
            --base_model "$BASE_MODEL" \
            --test_file "$TEST_FILE" \
            --info_file "$INFO_FILE" \
            --resume_from_checkpoint "$CHECKPOINT_PATH" \
            --target_layers "$layer" \
            --layer_index_base "$LAYER_INDEX_BASE" \
            --eval_topk "$EVAL_TOPK" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --max_samples "$MAX_SAMPLES" \
            --sample_seed "$SAMPLE_SEED" \
            --engram_float32 True \
            --print_batch_output True \
            --print_prompt False \
            --print_topn "$print_topn" \
            --save_json "$pred_file" \
            --save_metrics_jsonl "$METRICS_FILE"
        status=$?
        echo "[End] $(date '+%Y-%m-%d %H:%M:%S') status=${status}"
        exit "$status"
    } > "$log_file" 2>&1; then
        echo "[Warn] layer=${layer} failed, continue. See $log_file"
    fi

    idx=$((idx + 1))
done

echo "[Done] all selected layers finished"
