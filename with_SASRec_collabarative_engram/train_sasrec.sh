#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

python train_sasrec.py \
  --train_file "${TRAIN_FILE:-../data/Industrial_and_Scientific_dataset/train.jsonl}" \
  --valid_file "${VALID_FILE:-../data/Industrial_and_Scientific_dataset/valid.jsonl}" \
  --info_file "${INFO_FILE:-../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt}" \
  --output_dir "${OUTPUT_DIR:-./SAS-checkpoints}" \
  --max_seq_len "${MAX_SEQ_LEN:-50}" \
  --hidden_size "${HIDDEN_SIZE:-64}" \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE:-256}" \
  --lr "${LR:-1e-3}" \
  > "${LOG_FILE:-train_sasrec.log}" 2>&1
