#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/.." && pwd)}"
RUNTIME="${RUNTIME:-${TONGBENCH_RUNTIME:-native}}"
PYTHON_BIN="$(resolve_python_bin)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/For_user/output}"
SMOKE_SUBSET_ROOT="${SMOKE_SUBSET_ROOT:-$DATA_ROOT/subsets/stage0_smoke}"
if [[ ! -d "$SMOKE_SUBSET_ROOT" && -d "$DATA_ROOT/subsets/stage2_20pct" ]]; then
  # The published archive may omit a duplicate one-sample directory. The
  # evaluator selects one L4 target and its source L2 tasks for smoke.
  SMOKE_SUBSET_ROOT="$DATA_ROOT/subsets/stage2_20pct"
  echo "stage0_smoke subset not found; using stage2_20pct as the smoke source."
fi
CONTEXT_MODE="${CONTEXT_MODE:-task_compact}"
THINKING="${THINKING:-}"
QWEN_THINKING="${QWEN_THINKING:-${THINKING:-medium}}"
GLM_THINKING="${GLM_THINKING:-${THINKING:-high}}"
LUNA_THINKING="${LUNA_THINKING:-${THINKING:-medium}}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.8-flash}"
export GLM_MODEL="${GLM_MODEL:-ZHIPU/GLM-5.3-Flash}"
export LUNA_MODEL="${LUNA_MODEL:-gpt-5.6-luna}"

DRY_RUN_ARG=()
[[ "${DRY_RUN:-0}" == "1" ]] && DRY_RUN_ARG+=(--dry-run)

MODELS="${MODELS:-qwen}"
HARNESSES="${HARNESSES:-hermesagent codex claudecode openclaw}"
SETTINGS="${SETTINGS:-basic_l2 zero_shot_l4 in_context_l4 skill_l4}"
SMOKE_PARALLEL_JOBS="${SMOKE_PARALLEL_JOBS:-8}"
PER_COMMAND_PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
PER_COMMAND_SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-1}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/stage0_smoke/_smoke_parallel_logs}"
mkdir -p "$LOG_DIR"

split_words() {
  local raw="${1//,/ }"
  # shellcheck disable=SC2206
  local parts=($raw)
  printf '%s\n' "${parts[@]}"
}

contains_setting() {
  local needle="$1"
  local item
  while IFS= read -r item; do
    [[ "$item" == "$needle" ]] && return 0
  done < <(split_words "$SETTINGS")
  return 1
}

wait_for_slot() {
  while (( $(jobs -rp | wc -l | tr -d ' ') >= SMOKE_PARALLEL_JOBS )); do
    sleep 5
  done
}

run_wave() {
  local wave_name="$1"
  shift
  local settings=("$@")
  local pids=()
  local labels=()
  local logs=()
  local rc_files=()
  local models=()
  local harnesses=()
  local model harness setting label log_file rc_file model_thinking

  read -r -a models <<< "${MODELS//,/ }"
  read -r -a harnesses <<< "${HARNESSES//,/ }"

  echo "[$(date '+%F %T')] Launching $wave_name with SMOKE_PARALLEL_JOBS=$SMOKE_PARALLEL_JOBS"
  for model in "${models[@]}"; do
    case "$model" in
      qwen) model_thinking="$QWEN_THINKING" ;;
      glm) model_thinking="$GLM_THINKING" ;;
      luna) model_thinking="$LUNA_THINKING" ;;
      *) model_thinking="$THINKING" ;;
    esac
    for harness in "${harnesses[@]}"; do
      for setting in "${settings[@]}"; do
        contains_setting "$setting" || continue
        wait_for_slot
        label="${model}/${harness}/${setting}"
        log_file="$LOG_DIR/${wave_name}_${model}_${harness}_${setting}.log"
        rc_file="$LOG_DIR/${wave_name}_${model}_${harness}_${setting}.rc"
        rm -f "$rc_file"
        (
          # Keep the worker itself successful so one failed API/model run cannot
          # terminate the launcher before later harnesses are queued.
          set +e
          set -uo pipefail
          echo "[$(date '+%F %T')] START $label"
          "$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/run_eval_stage.py" \
            --stage stage0_smoke \
            --repo-root "$REPO_ROOT" \
            --project-root "$PROJECT_ROOT" \
            --data-root "$DATA_ROOT" \
            --subset-root "$SMOKE_SUBSET_ROOT" \
            --output-root "$OUTPUT_ROOT" \
            --python-bin "$PYTHON_BIN" \
            --runtime "$RUNTIME" \
            --models "$model" \
            --harnesses "$harness" \
            --settings "$setting" \
            --parallel-jobs "$PER_COMMAND_PARALLEL_JOBS" \
            --self-evolution-parallel-jobs "$PER_COMMAND_SELF_EVOLUTION_PARALLEL_JOBS" \
            --timeout-sec "${TIMEOUT_SEC:-7200}" \
            --max-iterations "${MAX_ITERATIONS:-320}" \
            --extra-steps "${EXTRA_STEPS:-6}" \
            --context-mode "$CONTEXT_MODE" \
            --thinking "$model_thinking" \
            "${DRY_RUN_ARG[@]}"
          command_rc=$?
          printf '%s\n' "$command_rc" > "$rc_file"
          echo "[$(date '+%F %T')] DONE $label (rc=$command_rc)"
        ) >"$log_file" 2>&1 &
        pids+=("$!")
        labels+=("$label")
        logs+=("$log_file")
        rc_files+=("$rc_file")
        echo "  started $label -> $log_file"
      done
    done
  done

  local failed=0
  local idx
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      failed=1
      echo "  failed ${labels[$idx]} (worker exited unexpectedly; log: ${logs[$idx]})" >&2
    elif [[ ! -s "${rc_files[$idx]}" ]] || [[ "$(<"${rc_files[$idx]}")" != "0" ]]; then
      failed=1
      echo "  failed ${labels[$idx]} (log: ${logs[$idx]})" >&2
    else
      echo "  ok ${labels[$idx]}"
    fi
  done
  return "$failed"
}

# Basic L2 is the only phase that creates learning data. It must finish before
# archives are built; learning settings then consume those archives and never
# rerun Basic L2 implicitly.
wave0_failed=0
if contains_setting "basic_l2"; then
  run_wave "wave0_basic" basic_l2 || wave0_failed=1
fi

archive_failed=0
if [[ "$wave0_failed" -eq 0 ]] && (contains_setting "in_context_l4" || contains_setting "skill_l4"); then
  set +e
  "$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/run_eval_stage.py" \
    --stage stage0_smoke \
    --repo-root "$REPO_ROOT" \
    --project-root "$PROJECT_ROOT" \
    --data-root "$DATA_ROOT" \
    --subset-root "$SMOKE_SUBSET_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --python-bin "$PYTHON_BIN" \
    --runtime "$RUNTIME" \
    --models "$MODELS" \
    --harnesses "$HARNESSES" \
    --settings basic_l2 \
    --prepare-learning-archive \
    --combo-parallel-jobs "${COMBO_PARALLEL_JOBS:-4}" \
    --timeout-sec "${TIMEOUT_SEC:-7200}" \
    --max-iterations "${MAX_ITERATIONS:-320}" \
    --extra-steps "${EXTRA_STEPS:-6}" \
    --context-mode "$CONTEXT_MODE" \
    --thinking "$THINKING" \
    "${DRY_RUN_ARG[@]}"
  archive_failed=$?
  set -e
fi

wave1_failed=0
if [[ "$wave0_failed" -eq 0 && "$archive_failed" -eq 0 ]]; then
  # Zero-shot, ICL, and Skill all consume the same prepared archive and are
  # independent after Basic L2; run them in the same wave.
  run_wave "wave1" zero_shot_l4 in_context_l4 skill_l4 || wave1_failed=1
else
  echo "wave1 skipped because Basic L2 or learning archive preparation failed."
  wave1_failed=1
fi

(( wave0_failed || archive_failed || wave1_failed )) && exit 1
exit 0
