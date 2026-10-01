#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
DATE_LABEL="${DATE_LABEL:-8.14}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input/highlevel_v2_smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/highlevel_v2_safe_choice_smoke}"
SECRET_FILE="${SECRET_FILE:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Cannot find python3 or python." >&2
  exit 2
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" && -n "$SECRET_FILE" && -f "$SECRET_FILE" ]]; then
  export DASHSCOPE_API_KEY="$(tr -d '\r\n' < "$SECRET_FILE")"
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Set DASHSCOPE_API_KEY or set SECRET_FILE to a local secret file." >&2
  exit 2
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"

READY_JSON="${READY_JSON:-$INPUT_DIR/ready_highlevel_v2_tasks_with_images.json}"
OPTION_BANK_JSONL="${OPTION_BANK_JSONL:-$INPUT_DIR/option_bank_highlevel_v2_smoke.jsonl}"
BACKENDS="${BACKENDS:-hermesagent}"
HERMES_MODEL="${HERMES_MODEL:-qwen3-vl-flash}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
TASK_LIMIT="${TASK_LIMIT:-3}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
MAX_ITERATIONS="${MAX_ITERATIONS:-180}"
L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
L6_MAX_STEPS="${L6_MAX_STEPS:-36}"
MAX_STEPS_FACTOR="${MAX_STEPS_FACTOR:-2.0}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE="${FORCE:-0}"

ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$INPUT_DIR/atomic_templates_updated_env_v3.json}"
SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$INPUT_DIR/subtask_templates_compressed_updated_env_v3.json}"
if [[ ! -f "$ATOMIC_TEMPLATE_PATH" ]]; then
  ATOMIC_TEMPLATE_PATH="$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json"
fi
if [[ ! -f "$SUBTASK_TEMPLATE_PATH" ]]; then
  SUBTASK_TEMPLATE_PATH="$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json"
fi

mkdir -p "$OUTPUT_ROOT"

resume_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  resume_args+=(--skip-existing)
fi
if [[ "$FORCE" == "1" ]]; then
  resume_args+=(--force)
fi

for backend in $BACKENDS; do
  if [[ "$backend" != "hermesagent" ]]; then
    echo "safe_choice smoke currently supports hermesagent only; got $backend" >&2
    exit 2
  fi
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks \
    --ready-json "$READY_JSON" \
    --results-root "$INPUT_DIR" \
    --output-dir "$OUTPUT_ROOT/$backend" \
    --backend hermesagent \
    --model "$HERMES_MODEL" \
    --api-key "$DASHSCOPE_API_KEY" \
    --base-url "$COMPAT_BASE_URL" \
    --description-field "${DESCRIPTION_FIELD:-description_highlevel_hard}" \
    --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language \
    --protocol-surface safe_choice \
    --choice-bank-jsonl "$OPTION_BANK_JSONL" \
    --choice-count 4 \
    --choice-seed "$SEED" \
    --write-human-trace-md \
    --generic-invalid-action-feedback \
    --composite-only \
    --limit "$TASK_LIMIT" \
    --parallel-jobs "$PARALLEL_JOBS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-iterations "$MAX_ITERATIONS" \
    --l3-max-steps "$L3_MAX_STEPS" \
    --l6-max-steps "$L6_MAX_STEPS" \
    --max-steps-factor "$MAX_STEPS_FACTOR" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --atomic-template-path "$ATOMIC_TEMPLATE_PATH" \
    --subtask-template-path "$SUBTASK_TEMPLATE_PATH" \
    "${resume_args[@]}"
done
