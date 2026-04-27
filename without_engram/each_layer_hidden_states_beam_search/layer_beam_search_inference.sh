#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

#TARGET_LAYERS="${TARGET_LAYERS:-25,23,21,19,17,15,14,13,12,11,9,7,5,3,1}"
#TARGET_LAYERS="${TARGET_LAYERS:-27,26,24,23,22,21,20}"
TARGET_LAYERS="${TARGET_LAYERS:-18 17 16 15}"
LAYER_INDEX_BASE="${LAYER_INDEX_BASE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
EVAL_TOPK="${EVAL_TOPK:-1,3,5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-50752}"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./output_sft_lr=3e-4/checkpoint-1981}"
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

mapfile -t LAYERS < <(python - "$TARGET_LAYERS" "$LAYER_INDEX_BASE" <<'PY'
import sys

target_layers = sys.argv[1]
layer_index_base = int(sys.argv[2])
normalized = target_layers.strip().lower()

if normalized in {"all", "*"}:
    # Qwen3-1.7B has 28 decoder layers. Override TARGET_LAYERS explicitly
    # if a different model is used.
    start = layer_index_base
    stop = layer_index_base + 28
    for layer in range(start, stop):
        print(layer)
    raise SystemExit

seen = set()
for part in target_layers.replace("，", ",").replace("、", ",").replace(" ", ",").split(","):
    part = part.strip()
    if not part:
        continue
    if "-" in part:
        left, right = [x.strip() for x in part.split("-", 1)]
        start, end = int(left), int(right)
        step = 1 if end >= start else -1
        values = range(start, end + step, step)
    else:
        values = [int(part)]
    for layer in values:
        if layer not in seen:
            seen.add(layer)
            print(layer)
PY
)

if [[ "${#LAYERS[@]}" -eq 0 ]]; then
    echo "[Error] no valid target layers selected: ${TARGET_LAYERS}"
    exit 1
fi

idx=0
for layer in "${LAYERS[@]}"; do
    master_port=$((MASTER_PORT_BASE + idx))
    log_file="${SCRIPT_DIR}/layer_${layer}_beam${LOG_SAFE_TOPK}_sample${MAX_SAMPLES}.log"
    pred_file="${SCRIPT_DIR}/layer_${layer}_predictions.json"
    status=0

    echo "[Run] layer=${layer}, eval_topk=${EVAL_TOPK}, max_samples=${MAX_SAMPLES} -> $log_file"

    {
        echo "[Run] layer=${layer}, eval_topk=${EVAL_TOPK}, max_samples=${MAX_SAMPLES}"
        echo "[Start] $(date '+%Y-%m-%d %H:%M:%S')"
        echo "[CWD] $(pwd)"
        echo "[Log] $log_file"
        echo "[Predictions] $pred_file"
        echo "[Metrics] $METRICS_FILE"
        set +e
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
            --print_batch_output True \
            --print_prompt False \
            --print_topn "$print_topn" \
            --save_json "$pred_file" \
            --save_metrics_jsonl "$METRICS_FILE"
        status=$?
        set -e
        echo "[End] $(date '+%Y-%m-%d %H:%M:%S') status=${status}"
    } > "$log_file" 2>&1

    if [[ "$status" -ne 0 ]]; then
        echo "[Warn] layer=${layer} failed, continue. See $log_file"
    fi

    idx=$((idx + 1))
done

echo "[Done] all selected layers finished"
