#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-formal}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "Usage: $0 smoke|formal" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

NATIVE_ENV_FILE="${NATIVE_ENV_FILE:-}"
if [[ -n "$NATIVE_ENV_FILE" && -f "$NATIVE_ENV_FILE" ]]; then
  source "$NATIVE_ENV_FILE"
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/For_user/output/stage3_remaining_80pct_qwen38_formal_20260907}"
PYTHON_BIN="${PYTHON_BIN:-$(resolve_python_bin)}"

MODEL="${MODEL:-qwen3.8-flash}"
THINKING="${THINKING:-medium}"
RUNTIME="${RUNTIME:-native}"
CONTEXT_MODE="${CONTEXT_MODE:-task_compact}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10800}"
MAX_ITERATIONS="${MAX_ITERATIONS:-320}"
EXTRA_STEPS="${EXTRA_STEPS:-6}"
START_WAVE="${START_WAVE:-1}"

: "${QWEN_API_KEY:?Set QWEN_API_KEY in the environment; it is never stored by this script}"

if [[ "$RUNTIME" == "native" ]]; then
  : "${TONGBENCH_NATIVE_BUNDLE_ROOT:?Native runtime requires TONGBENCH_NATIVE_BUNDLE_ROOT; set it directly or via NATIVE_ENV_FILE}"
  : "${TONGBENCH_NATIVE_MCP_PYTHON:?Native runtime requires TONGBENCH_NATIVE_MCP_PYTHON; set it directly or via NATIVE_ENV_FILE}"
fi

if [[ "$MODE" == "smoke" ]]; then
  WAVE_ROOTS=("$DATA_ROOT/subsets/stage3_remaining_80pct_qwen38_smoke")
else
  WAVE_ROOTS=()
  for wave_number in 1 2 3 4; do
    if (( wave_number >= START_WAVE )); then
      WAVE_ROOTS+=("$DATA_ROOT/subsets/stage3_remaining_80pct_qwen38_wave$(printf '%02d' "$wave_number")")
    fi
  done
fi

LOG_ROOT="$OUTPUT_ROOT/stage3_full/_run_logs/$MODE"
mkdir -p "$LOG_ROOT"

run_combo() {
  local subset_root="$1"
  local harness="$2"
  local setting="$3"
  local log_file="$4"
  local command=(
    "$PYTHON_BIN"
    "$REPO_ROOT/For_user/scripts/run_eval_stage.py"
    --stage stage3_full
    --repo-root "$REPO_ROOT"
    --project-root "$PROJECT_ROOT"
    --data-root "$DATA_ROOT"
    --subset-root "$subset_root"
    --output-root "$OUTPUT_ROOT"
    --python-bin "$PYTHON_BIN"
    --runtime "$RUNTIME"
    --models qwen
    --harnesses "$harness"
    --settings "$setting"
    --model "$MODEL"
    --thinking "$THINKING"
    --parallel-jobs "$PARALLEL_JOBS"
    --self-evolution-parallel-jobs "$SELF_EVOLUTION_PARALLEL_JOBS"
    --combo-parallel-jobs 1
    --timeout-sec "$TIMEOUT_SEC"
    --max-iterations "$MAX_ITERATIONS"
    --extra-steps "$EXTRA_STEPS"
    --context-mode "$CONTEXT_MODE"
    --confirm-full
  )
  if [[ "$setting" == "in_context_l4" || "$setting" == "skill_l4" ]]; then
    command+=(--learning-reuse-manifest "$OUTPUT_ROOT/stage3_full/_learning_archives/qwen/$harness/learning_reuse_manifest.json")
  fi
  (
    cd "$REPO_ROOT"
    "${command[@]}"
  ) >"$log_file" 2>&1
}

for subset_root in "${WAVE_ROOTS[@]}"; do
  wave_name="$(basename "$subset_root")"
  echo "Starting $MODE $wave_name: 4 harnesses x 3 L4 settings"
  pids=()
  labels=()
  for harness in hermesagent codex claudecode openclaw; do
    for setting in zero_shot_l4 in_context_l4 skill_l4; do
      log_file="$LOG_ROOT/${wave_name}_${harness}_${setting}.log"
      run_combo "$subset_root" "$harness" "$setting" "$log_file" &
      pids+=("$!")
      labels+=("$harness/$setting")
    done
  done

  failed=0
  set +e
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}"
    code=$?
    if [[ $code -ne 0 ]]; then
      echo "FAILED $wave_name ${labels[$index]} rc=$code; see $LOG_ROOT/${wave_name}_*.log" >&2
      failed=1
    fi
  done
  set -e
  if [[ $failed -ne 0 ]]; then
    exit 1
  fi
  echo "Completed $MODE $wave_name"
done

echo "Completed $MODE remaining L4 run"
