#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"

DATE_LABEL="${DATE_LABEL:-8.16_api_smoke}"
FORMAL_NAME="${FORMAL_NAME:-highlevel_v2_api_smoke_036_02}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input/formal/$FORMAL_NAME}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/smoke/${FORMAL_NAME}_qwen3_7_plus}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/report/smoke/${FORMAL_NAME}_qwen3_7_plus}"
SECRET_FILE="${SECRET_FILE:-$PROJECT_ROOT/.local_secrets/tongsim_dashscope_api_key}"

env \
  DATE_LABEL="$DATE_LABEL" \
  FORMAL_NAME="$FORMAL_NAME" \
  INPUT_DIR="$INPUT_DIR" \
  OUTPUT_BASE="$OUTPUT_BASE" \
  REPORT_DIR="$REPORT_DIR" \
  SECRET_FILE="$SECRET_FILE" \
  HERMES_MODEL="${HERMES_MODEL:-qwen3.7-plus}" \
  RUN_MODES="${RUN_MODES:-basic_l3 zero_shot_l6 report}" \
  TASK_LIMIT_L3="${TASK_LIMIT_L3:-3}" \
  TASK_LIMIT_L6="${TASK_LIMIT_L6:-2}" \
  SELF_EVOLUTION_LIMIT="${SELF_EVOLUTION_LIMIT:-2}" \
  PARALLEL_JOBS="${PARALLEL_JOBS:-2}" \
  SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-2}" \
  MAX_STEPS_FACTOR="${MAX_STEPS_FACTOR:-2.0}" \
  FORCE="${FORCE:-1}" \
  SKIP_EXISTING="${SKIP_EXISTING:-1}" \
  bash "$REPO_ROOT/scripts/run_highlevel_v2_formal_815_eval.sh"

echo "API smoke eval complete:"
echo "  output=$OUTPUT_BASE"
echo "  report=$REPORT_DIR"
