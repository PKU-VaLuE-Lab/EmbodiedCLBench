#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
PYTHON_BIN="$(resolve_python_bin)"

TASKGEN_ROOT="${TASKGEN_ROOT:-$PROJECT_ROOT/important_results_taskgen/8.24_l2_l4_scaleup_v1}"
EVAL_INPUT_ROOT="${EVAL_INPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.24_l2_l4_scaleup_v1/input}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"

"$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/prepare_inputs.py" \
  --taskgen-root "$TASKGEN_ROOT" \
  --eval-input-root "$EVAL_INPUT_ROOT" \
  --output-root "$DATA_ROOT"
