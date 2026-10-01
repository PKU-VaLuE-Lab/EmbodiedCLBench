#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$PROJECT_ROOT/important_results_eval/8.10_remote_yanan_5c2dbf78}"
IMAGE_REUSE_ROOT="${IMAGE_REUSE_ROOT:-$PACKAGE_ROOT/input/036_02/generated_from_imaged_l1/image_reuse}"
OUR_OUTPUT_ROOT="${OUR_OUTPUT_ROOT:-$PACKAGE_ROOT/our_output}"
REFERENCE_OUTPUT_ROOT="${REFERENCE_OUTPUT_ROOT:-$PACKAGE_ROOT/reference_output}"

BACKENDS="${BACKENDS:-hermesagent openclaw}"
RUN_DIALOGUE="${RUN_DIALOGUE:-1}"
RUN_SINGLE="${RUN_SINGLE:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
LOCALIZE_IMAGES="${LOCALIZE_IMAGES:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"

L2_TASK_LIMIT="${L2_TASK_LIMIT:-${TASK_LIMIT:-2}}"
L3_TASK_LIMIT="${L3_TASK_LIMIT:-${TASK_LIMIT:-2}}"

ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$REFERENCE_OUTPUT_ROOT/atomic_templates_updated_env_v3.json}"
SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$REFERENCE_OUTPUT_ROOT/subtask_templates_compressed_updated_env_v3.json}"

if [[ "$DRY_RUN" == "1" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  export DASHSCOPE_API_KEY="dry-run-placeholder"
fi
if [[ "$DRY_RUN" != "1" && -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Set DASHSCOPE_API_KEY for an actual API run, or set DRY_RUN=1 for a no-API plan check." >&2
  exit 2
fi

if [[ "$LOCALIZE_IMAGES" == "1" ]]; then
  localize_args=(
    --project-root "$PROJECT_ROOT"
    --package-root "$PACKAGE_ROOT"
    --image-reuse-root "$IMAGE_REUSE_ROOT"
    --l2-task-limit "$L2_TASK_LIMIT"
    --l3-task-limit "$L3_TASK_LIMIT"
  )
  if [[ "${LOCALIZE_FORCE:-0}" == "1" ]]; then
    localize_args+=(--force)
  fi
  python "$SCRIPT_DIR/localize_yanan_5c2dbf78_images.py" "${localize_args[@]}"
fi

run_args=()
if [[ "$DRY_RUN" == "1" ]]; then
  run_args+=(--dry-run)
fi

common_env=(
  REPO_ROOT="$REPO_ROOT"
  PROJECT_ROOT="$PROJECT_ROOT"
  IMAGE_REUSE_ROOT="$IMAGE_REUSE_ROOT"
  READY_JSON="$IMAGE_REUSE_ROOT/ready_l2_l3_tasks_with_images.json"
  RESULTS_ROOT="$IMAGE_REUSE_ROOT"
  BACKENDS="$BACKENDS"
  RUN_DIALOGUE="$RUN_DIALOGUE"
  RUN_SINGLE="$RUN_SINGLE"
  RUN_SUMMARY="$RUN_SUMMARY"
  SKIP_EXISTING="$SKIP_EXISTING"
  FORCE="$FORCE"
  ATOMIC_TEMPLATE_PATH="$ATOMIC_TEMPLATE_PATH"
  SUBTASK_TEMPLATE_PATH="$SUBTASK_TEMPLATE_PATH"
)

env "${common_env[@]}" \
  L2_TASK_LIMIT="$L2_TASK_LIMIT" \
  OUTPUT_ROOT="$OUR_OUTPUT_ROOT/outputs_l2" \
  bash "$REPO_ROOT/scripts/run_image_reuse_l2_all_frameworks.sh" \
  "${run_args[@]}" "$@"

env "${common_env[@]}" \
  L3_TASK_LIMIT="$L3_TASK_LIMIT" \
  OUTPUT_ROOT="$OUR_OUTPUT_ROOT/outputs_l3" \
  bash "$REPO_ROOT/scripts/run_image_reuse_l3_all_frameworks.sh" \
  "${run_args[@]}" "$@"

if [[ "$RUN_COMPARE" == "1" ]]; then
  python "$SCRIPT_DIR/compare_yanan_5c2dbf78_outputs.py" \
    --project-root "$PROJECT_ROOT" \
    --package-root "$PACKAGE_ROOT" \
    --backends $BACKENDS \
    --levels L2 L3
fi
