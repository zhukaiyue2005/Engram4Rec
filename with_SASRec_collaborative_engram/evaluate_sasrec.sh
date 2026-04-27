#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1

python evaluate_sasrec.py \
  --checkpoint_path "${SASREC_CHECKPOINT_PATH:-./SAS-checkpoints/sasrec_best.pt}" \
  --test_file "${TEST_FILE:-../data/Industrial_and_Scientific_dataset/test.jsonl}" \
  --info_file "${INFO_FILE:-../data/Amazon/info/Industrial_and_Scientific_5_1996-10-2018-11.txt}" \
  --max_seq_len "${MAX_SEQ_LEN:-50}" \
  --eval_topk "${EVAL_TOPK:-1,3,5,10}" \
  --save_json "${SAVE_JSON:-./sasrec_test_metrics.json}" \
  > "${LOG_FILE:-evaluate_sasrec.log}" 2>&1
