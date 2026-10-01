#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RUNTIME="${RUNTIME:-${TONGBENCH_RUNTIME:-docker}}"
if [[ -z "${PYTHON_BIN:-}" && "$RUNTIME" == "native" && -n "${TONGBENCH_NATIVE_BUNDLE_ROOT:-}" && -x "$TONGBENCH_NATIVE_BUNDLE_ROOT/hermes/.venv/bin/python3" ]]; then
  PYTHON_BIN="$TONGBENCH_NATIVE_BUNDLE_ROOT/hermes/.venv/bin/python3"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
fi
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/For_user/output}"

DRY_RUN_ARG=()
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_ARG+=(--dry-run)

"$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/run_eval_stage.py" \
  --stage stage3_full \
  --repo-root "$REPO_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --python-bin "$PYTHON_BIN" \
  --runtime "$RUNTIME" \
  --models "${MODELS:-}" \
  --harnesses "${HARNESSES:-}" \
  --settings "${SETTINGS:-}" \
  --parallel-jobs "${PARALLEL_JOBS:-8}" \
  --self-evolution-parallel-jobs "${SELF_EVOLUTION_PARALLEL_JOBS:-4}" \
  --combo-parallel-jobs "${COMBO_PARALLEL_JOBS:-4}" \
  --timeout-sec "${TIMEOUT_SEC:-10800}" \
  --max-iterations "${MAX_ITERATIONS:-320}" \
  --extra-steps "${EXTRA_STEPS:-6}" \
  --confirm-full \
  "${DRY_RUN_ARG[@]}"
