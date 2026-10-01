#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
DATE_LABEL="${DATE_LABEL:-8.16}"
FORMAL_NAME="${FORMAL_NAME:-highlevel_v2_formal_3rooms_816}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input/formal/$FORMAL_NAME}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/formal}"
REPORT_ROOT="${REPORT_ROOT:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/report/formal}"
SECRET_FILE="${SECRET_FILE:-$PROJECT_ROOT/.local_secrets/tongsim_dashscope_api_key}"

FLASH_MODEL="${FLASH_MODEL:-qwen3-vl-flash}"
STRONG_MODEL="${STRONG_MODEL:-}"
STRONG_MODEL_CANDIDATES="${STRONG_MODEL_CANDIDATES:-qwen3.7-plus qwen3-vl-plus qwen3.8-max qwen-vl-max-latest}"
MODEL_ARMS="${MODEL_ARMS:-flash strong}"
RUN_PROBE="${RUN_PROBE:-1}"
ONLY_PROBE="${ONLY_PROBE:-0}"
ALLOW_STRONG_FAILURE="${ALLOW_STRONG_FAILURE:-1}"

RUN_MODES="${RUN_MODES:-basic_l3 language_bias_l3 zero_shot_l6 in_context_l6 report}"
TASK_LIMIT_L3="${TASK_LIMIT_L3:-108}"
TASK_LIMIT_L6="${TASK_LIMIT_L6:-90}"
SELF_EVOLUTION_LIMIT="${SELF_EVOLUTION_LIMIT:-90}"
LEARNING_TASK_COUNT="${LEARNING_TASK_COUNT:-3}"
RELATED_LEARNING_MAX_COUNT="${RELATED_LEARNING_MAX_COUNT:-3}"
USE_EXPLICIT_LEARNING_TASK_IDS="${USE_EXPLICIT_LEARNING_TASK_IDS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-12}"
SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
SELF_EVOLUTION_TIMEOUT_SEC="${SELF_EVOLUTION_TIMEOUT_SEC:-3600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-180}"
SELF_EVOLUTION_MAX_ITERATIONS="${SELF_EVOLUTION_MAX_ITERATIONS:-260}"
MAX_STEPS_FACTOR="${MAX_STEPS_FACTOR:-2.0}"
L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
L6_MAX_STEPS="${L6_MAX_STEPS:-36}"
FORCE="${FORCE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9][^a-z0-9]*/_/g; s/^_//; s/_$//'
}

run_815_runner() {
  local model="$1"
  local slug="$2"
  local run_modes="$3"
  local task_limit_l3="$4"
  local task_limit_l6="$5"
  local self_limit="$6"
  local output_base="$7"
  local report_dir="$8"
  env \
    DATE_LABEL="$DATE_LABEL" \
    FORMAL_NAME="$FORMAL_NAME" \
    INPUT_DIR="$INPUT_DIR" \
    OUTPUT_BASE="$output_base" \
    REPORT_DIR="$report_dir" \
    HERMES_MODEL="$model" \
    SECRET_FILE="$SECRET_FILE" \
    RUN_MODES="$run_modes" \
    TASK_LIMIT_L3="$task_limit_l3" \
    TASK_LIMIT_L6="$task_limit_l6" \
    SELF_EVOLUTION_LIMIT="$self_limit" \
    LEARNING_TASK_COUNT="$LEARNING_TASK_COUNT" \
    RELATED_LEARNING_MAX_COUNT="$RELATED_LEARNING_MAX_COUNT" \
    USE_EXPLICIT_LEARNING_TASK_IDS="$USE_EXPLICIT_LEARNING_TASK_IDS" \
    PARALLEL_JOBS="$PARALLEL_JOBS" \
    SELF_EVOLUTION_PARALLEL_JOBS="$SELF_EVOLUTION_PARALLEL_JOBS" \
    TIMEOUT_SEC="$TIMEOUT_SEC" \
    SELF_EVOLUTION_TIMEOUT_SEC="$SELF_EVOLUTION_TIMEOUT_SEC" \
    MAX_ITERATIONS="$MAX_ITERATIONS" \
    SELF_EVOLUTION_MAX_ITERATIONS="$SELF_EVOLUTION_MAX_ITERATIONS" \
    MAX_STEPS_FACTOR="$MAX_STEPS_FACTOR" \
    L3_MAX_STEPS="$L3_MAX_STEPS" \
    L6_MAX_STEPS="$L6_MAX_STEPS" \
    FORCE="$FORCE" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    bash "$REPO_ROOT/scripts/run_highlevel_v2_formal_815_eval.sh"
}

probe_model() {
  local model="$1"
  local slug="$(slugify "$model")"
  local probe_out="$OUTPUT_ROOT/${FORMAL_NAME}_${slug}_probe"
  local probe_report="$REPORT_ROOT/${FORMAL_NAME}_${slug}_probe"
  echo "=== Probing model $model ===" >&2
  set +e
  PARALLEL_JOBS=1 SELF_EVOLUTION_PARALLEL_JOBS=1 run_815_runner \
    "$model" "$slug" "basic_l3 report" 1 1 1 "$probe_out" "$probe_report"
  local rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    echo "model_probe_ok model=$model report=$probe_report" >&2
    return 0
  fi
  echo "model_probe_failed model=$model rc=$rc output=$probe_out report=$probe_report" >&2
  return "$rc"
}

select_strong_model() {
  if [[ -n "$STRONG_MODEL" ]]; then
    if [[ "$RUN_PROBE" == "1" ]]; then
      probe_model "$STRONG_MODEL" >&2
    fi
    printf '%s' "$STRONG_MODEL"
    return 0
  fi
  local candidate
  for candidate in $STRONG_MODEL_CANDIDATES; do
    if [[ "$RUN_PROBE" == "1" ]]; then
      if probe_model "$candidate" >&2; then
        printf '%s' "$candidate"
        return 0
      fi
    else
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

run_full_arm() {
  local label="$1"
  local model="$2"
  local slug="$(slugify "$model")"
  local output_base="$OUTPUT_ROOT/${FORMAL_NAME}_${label}_${slug}"
  local report_dir="$REPORT_ROOT/${FORMAL_NAME}_${label}_${slug}"
  echo "=== Running 8.16 arm=$label model=$model ==="
  run_815_runner "$model" "$slug" "$RUN_MODES" "$TASK_LIMIT_L3" "$TASK_LIMIT_L6" "$SELF_EVOLUTION_LIMIT" "$output_base" "$report_dir"
  echo "arm_done label=$label model=$model output=$output_base report=$report_dir"
}

if [[ ! -f "$INPUT_DIR/ready_l3_tasks_with_images.json" || ! -f "$INPUT_DIR/ready_l6_tasks_with_images.json" ]]; then
  echo "Missing prepared input under $INPUT_DIR. Prepare the input with the task-generation pipeline first." >&2
  exit 2
fi

selected_strong=""
for arm in $MODEL_ARMS; do
  case "$arm" in
    flash)
      if [[ "$RUN_PROBE" == "1" ]]; then
        probe_model "$FLASH_MODEL" || exit $?
      fi
      if [[ "$ONLY_PROBE" != "1" ]]; then
        run_full_arm "flash" "$FLASH_MODEL"
      fi
      ;;
    strong)
      set +e
      selected_strong="$(select_strong_model)"
      rc=$?
      set -e
      if [[ "$rc" -ne 0 || -z "$selected_strong" ]]; then
        echo "No strong model candidate passed probe. candidates=[$STRONG_MODEL_CANDIDATES]" >&2
        if [[ "$ALLOW_STRONG_FAILURE" == "1" ]]; then
          continue
        fi
        exit 1
      fi
      echo "selected_strong_model=$selected_strong"
      if [[ "$ONLY_PROBE" != "1" ]]; then
        run_full_arm "strong" "$selected_strong"
      fi
      ;;
    *)
      echo "Unknown MODEL_ARMS item: $arm" >&2
      exit 2
      ;;
  esac
done

echo "8.16 eval complete. output_root=$OUTPUT_ROOT report_root=$REPORT_ROOT"
