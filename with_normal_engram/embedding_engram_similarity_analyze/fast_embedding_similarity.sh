#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CURRENT_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "${MPLCONFIGDIR}"

BASE_MODEL="${BASE_MODEL:-Qwen3-1.7B}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/output_sft_layer=6,13,20_tbs=128_lora=3e-4_engram=3e-4/checkpoint-1988}"
DATA_FILE="${DATA_FILE:-${PROJECT_DIR}/../data/Industrial_and_Scientific_dataset/valid.jsonl}"
SAVE_DIR="${SAVE_DIR:-${CURRENT_DIR}/engram_analysis_fast}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
NUM_SAMPLES="${NUM_SAMPLES:-500000}"
CUTOFF_LEN="${CUTOFF_LEN:-2048}"
X_MIN="${X_MIN:--1.0}"
X_MAX="${X_MAX:-1.0}"
BINS="${BINS:-60}"
DEVICE="${DEVICE:-cuda}"
SIMILARITY_MODE="${SIMILARITY_MODE:-sampled}"
EXACT_BLOCK_SIZE="${EXACT_BLOCK_SIZE:-4096}"
DEDUPLICATE_EMBEDDINGS="${DEDUPLICATE_EMBEDDINGS:-true}"
LOG_EVERY="${LOG_EVERY:-10}"
LOG_FILE="${LOG_FILE:-${CURRENT_DIR}/embedding_similar.log}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${CURRENT_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
python fast_embedding_similarity.py \
    --base_model "${BASE_MODEL}" \
    --resume_from_checkpoint "${CHECKPOINT}" \
    --data_file "${DATA_FILE}" \
    --save_dir "${SAVE_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --num_samples "${NUM_SAMPLES}" \
    --cutoff_len "${CUTOFF_LEN}" \
    --x_min "${X_MIN}" \
    --x_max "${X_MAX}" \
    --bins "${BINS}" \
    --device "${DEVICE}" \
    --similarity_mode "${SIMILARITY_MODE}" \
    --exact_block_size "${EXACT_BLOCK_SIZE}" \
    --deduplicate_embeddings "${DEDUPLICATE_EMBEDDINGS}" \
    --log_every "${LOG_EVERY}" \
    ${EXTRA_ARGS} 2>&1 | tee "${LOG_FILE}"
