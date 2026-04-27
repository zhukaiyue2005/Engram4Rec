#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$PROJECT_DIR")/data"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/output_sft_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4}"
ENGRAM_LAYER_IDS="6,13,20"
MASTER_PORT_BASE=27111
METHOD="exact_kl"
SELECT_CHECKPOINT_SUFFIXES="1988"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/Industrial_and_Scientific_dataset/test.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

if [[ ! -d "$CHECKPOINT_ROOT" ]]; then
    echo "[Error] checkpoint root not found: $CHECKPOINT_ROOT"
    exit 1
fi

root_tag="${CHECKPOINT_ROOT#../}"
root_tag="${root_tag#./}"
root_tag="${root_tag#output_sft_}"
sample_tag="test${MAX_SAMPLES}_seed${SAMPLE_SEED}"

declare -a SELECTED_CKPTS=()
INPUT_TOKENS=(${SELECT_CHECKPOINT_SUFFIXES//,/ })

for tok in "${INPUT_TOKENS[@]}"; do
    if [[ "$tok" =~ ^[0-9]+$ ]]; then
        ckpt_name="checkpoint-$tok"
        if [[ -d "$CHECKPOINT_ROOT/$ckpt_name" ]]; then
            SELECTED_CKPTS+=("$ckpt_name")
        else
            echo "[Warn] checkpoint not found: $ckpt_name"
        fi
    else
        echo "[Warn] invalid checkpoint suffix: $tok"
    fi
done

if [[ "${#SELECTED_CKPTS[@]}" -eq 0 ]]; then
    echo "[Error] no valid checkpoints selected."
    exit 1
fi

idx=0
for ckpt_name in "${SELECTED_CKPTS[@]}"; do
    ckpt_path="$CHECKPOINT_ROOT/$ckpt_name"
    log_file="kl_eval_${METHOD}_${sample_tag}_${root_tag}_${ckpt_name}.log"
    json_file="kl_eval_${METHOD}_${sample_tag}_${root_tag}_${ckpt_name}.json"
    master_port=$((MASTER_PORT_BASE + idx))

    if [[ -s "$json_file" ]]; then
        echo "[Skip] ckpt=${ckpt_name} (json exists: $json_file)"
        idx=$((idx + 1))
        continue
    fi

    echo "[Run] method=${METHOD}, layers=${ENGRAM_LAYER_IDS}, ckpt=${ckpt_name} -> $log_file"

    if ! torchrun --nproc_per_node 1 --master_port="$master_port" \
            kl_evaluate.py \
            --batch_size 8 \
            --base_model "$BASE_MODEL" \
            --test_file "$TEST_FILE" \
            --max_samples "$MAX_SAMPLES" \
            --sample_seed "$SAMPLE_SEED" \
            --engram_layer_ids "${ENGRAM_LAYER_IDS}" \
            --engram_float32 True \
            --method "${METHOD}" \
            --resume_from_checkpoint "$ckpt_path" \
            --save_json "$json_file" \
            > "$log_file" 2>&1; then
        echo "[Warn] $ckpt_name failed, continue. See $log_file"
    fi

    idx=$((idx + 1))
done

echo "[Done] all checkpoints finished"
