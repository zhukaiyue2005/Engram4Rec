#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-$(id -u)}}"

cd "$SCRIPT_DIR"
python "$SCRIPT_DIR/gate_evaluate.py" > "$SCRIPT_DIR/gate_eval.log" 2>&1
