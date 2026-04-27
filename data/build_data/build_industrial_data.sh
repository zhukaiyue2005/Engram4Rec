#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${SCRIPT_DIR}/build_industrial_data.py" \
  --amazon-root "${DATA_DIR}/Amazon" \
  --category-prefix "Industrial_and_Scientific" \
  --category-text "industrial and scientific items" \
  --output-dir "${DATA_DIR}/Industrial_and_Scientific_dataset" \
  --max-rows-per-split 0 \
  --seed 42

echo "Done. Data generated under ${DATA_DIR}/Industrial_and_Scientific_dataset"
