#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$WORKSPACE_ROOT}"
DATE_LABEL="${DATE_LABEL:-8.11}"

INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/${RUN_SCOPE:-test}}"
BACKENDS="${BACKENDS:-hermesagent}"
TASK_LIMIT="${TASK_LIMIT:-10}"
RUN_VARIANTS="${RUN_VARIANTS:-baseline highlevel}"
RUN_LEVELS="${RUN_LEVELS:-l2}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
PASSTHROUGH_ARGS=("$@")

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}"
if [[ -z "$DASHSCOPE_API_KEY" ]]; then
  echo "Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running HermesAgent." >&2
  exit 2
fi

common_env=(
  BACKENDS="$BACKENDS"
  TASK_LIMIT="$TASK_LIMIT"
  RUN_DIALOGUE="${RUN_DIALOGUE:-1}"
  RUN_SINGLE="${RUN_SINGLE:-1}"
  RUN_SUMMARY="${RUN_SUMMARY:-1}"
  L1_MAX_STEPS="${L1_MAX_STEPS:-6}"
  L2_MAX_STEPS="${L2_MAX_STEPS:-12}"
  L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
  MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
  PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
  TEMPERATURE="${TEMPERATURE:-0}"
  SEED="${SEED:-42}"
  ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$INPUT_DIR/atomic_templates_updated_env_v3.json}"
  SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$INPUT_DIR/subtask_templates_compressed_updated_env_v3.json}"
)

failures=()

run_variant_level() {
  local variant="$1"
  local level="$2"
  local description_field="$3"
  local hide_decomposition="$4"
  local output_root="$OUTPUT_BASE/$variant/$level"

  mkdir -p "$output_root"
  if [[ "$level" == "l2" ]]; then
    env "${common_env[@]}" \
      PROJECT_ROOT="$PROJECT_ROOT" \
      IMAGE_REUSE_ROOT="$INPUT_DIR" \
      READY_JSON="$INPUT_DIR/ready_l2_tasks_with_images.json" \
      RESULTS_ROOT="$INPUT_DIR" \
      OUTPUT_ROOT="$output_root" \
      DESCRIPTION_FIELD="$description_field" \
      HIDE_DECOMPOSITION="$hide_decomposition" \
      bash "$REPO_ROOT/scripts/run_image_reuse_l2_all_frameworks.sh" \
      "${PASSTHROUGH_ARGS[@]}"
  elif [[ "$level" == "l3" ]]; then
    env "${common_env[@]}" \
      PROJECT_ROOT="$PROJECT_ROOT" \
      IMAGE_REUSE_ROOT="$INPUT_DIR" \
      READY_JSON="$INPUT_DIR/ready_l3_tasks_with_images.json" \
      RESULTS_ROOT="$INPUT_DIR" \
      OUTPUT_ROOT="$output_root" \
      DESCRIPTION_FIELD="$description_field" \
      HIDE_DECOMPOSITION="$hide_decomposition" \
      bash "$REPO_ROOT/scripts/run_image_reuse_l3_all_frameworks.sh" \
      "${PASSTHROUGH_ARGS[@]}"
  else
    echo "Unknown level: $level" >&2
    exit 2
  fi
}

for variant in $RUN_VARIANTS; do
  case "$variant" in
    baseline|baseline_nohide)
      description_field="description_original"
      hide_decomposition="0"
      output_variant="baseline"
      ;;
    baseline_hide)
      description_field="description_original"
      hide_decomposition="1"
      output_variant="baseline_hide"
      ;;
    highlevel|highlevel_hide)
      description_field="description_highlevel"
      hide_decomposition="1"
      output_variant="highlevel"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 2
      ;;
  esac

  for level in $RUN_LEVELS; do
    level="${level,,}"
    echo "Running $output_variant ${level^^}"
    if ! run_variant_level "$output_variant" "$level" "$description_field" "$hide_decomposition"; then
      failures+=("$output_variant/$level")
      if [[ "$CONTINUE_ON_FAILURE" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Completed with failed variant runs:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
