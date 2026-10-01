#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
source "$REPO_ROOT/For_user/scripts/python_env.sh"
PYTHON_BIN="$(resolve_python_bin)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"
SUBSET_ROOT="${SUBSET_ROOT:-$DATA_ROOT/subsets/main_10pct_child}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/For_user/output/context_compaction_smoke}"
RUNTIME="${RUNTIME:-native}"

mkdir -p "$OUTPUT_ROOT"

run_mode() {
  local CONTEXT_MODE="$1"
  local MODE_ROOT="$OUTPUT_ROOT/$CONTEXT_MODE"
  echo "[$(date '+%F %T')] smoke context_mode=$CONTEXT_MODE"
  "$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/run_eval_stage.py" \
    --stage stage0_smoke \
    --repo-root "$REPO_ROOT" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$DATA_ROOT" \
    --subset-root "$SUBSET_ROOT" \
    --output-root "$MODE_ROOT" \
    --python-bin "$PYTHON_BIN" \
    --runtime "$RUNTIME" \
    --models qwen \
    --harnesses hermesagent \
    --settings "basic_l2 zero_shot_l4 in_context_l4 skill_l4" \
    --context-mode "$CONTEXT_MODE" \
    --parallel-jobs 1 \
    --self-evolution-parallel-jobs 1 \
    --combo-parallel-jobs 3 \
    --timeout-sec "${TIMEOUT_SEC:-7200}" \
    --max-iterations "${MAX_ITERATIONS:-320}" \
    --extra-steps "${EXTRA_STEPS:-6}"

  "$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/validate_stage0_smoke.py" \
    --stage-root "$MODE_ROOT/stage0_smoke" \
    --models qwen \
    --harnesses hermesagent \
    --settings "basic_l2 zero_shot_l4 in_context_l4 skill_l4"
}

if [[ -n "${CONTEXT_MODE_ONLY:-}" ]]; then
  run_mode "$CONTEXT_MODE_ONLY"
elif [[ "${PARALLEL_CONTEXT_MODES:-0}" == "1" ]]; then
  pids=()
  for CONTEXT_MODE in full task_compact; do
    run_mode "$CONTEXT_MODE" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  (( status == 0 )) || exit "$status"
else
  for CONTEXT_MODE in full task_compact; do
    run_mode "$CONTEXT_MODE"
  done
fi

echo "Smoke complete. The 10% scale run is intentionally not started."
