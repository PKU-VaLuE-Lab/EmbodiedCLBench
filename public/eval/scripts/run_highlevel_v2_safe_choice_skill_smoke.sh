#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
DATE_LABEL="${DATE_LABEL:-8.14}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input/highlevel_v2_smoke_l3_only}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/highlevel_v2_skill_smoke}"
SECRET_FILE="${SECRET_FILE:-}"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_PYTHON"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" && -n "$SECRET_FILE" && -f "$SECRET_FILE" ]]; then
  export DASHSCOPE_API_KEY="$(tr -d '\r\n' < "$SECRET_FILE" | sed 's/\\n//g; s/\\r//g')"
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Set DASHSCOPE_API_KEY or set SECRET_FILE to a local secret file." >&2
  exit 2
fi
export DASHSCOPE_API_KEY="$(printf '%s' "$DASHSCOPE_API_KEY" | tr -d '\r\n' | sed 's/\\n//g; s/\\r//g')"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$DASHSCOPE_API_KEY}"

READY_JSON="${READY_JSON:-$INPUT_DIR/ready_highlevel_v2_tasks_with_images.json}"
BACKENDS="${BACKENDS:-hermesagent}"
HERMES_MODEL="${HERMES_MODEL:-qwen3-vl-flash}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
TASK_ID="${TASK_ID:-task_L3_HL_v2_003}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
L1_MAX_STEPS="${L1_MAX_STEPS:-6}"
L2_MAX_STEPS="${L2_MAX_STEPS:-12}"
L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
SKILL_WORD_LIMIT="${SKILL_WORD_LIMIT:-300}"
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

task_args=()
if [[ -n "$TASK_ID" ]]; then
  task_args+=(--task-id "$TASK_ID")
fi

for backend in $BACKENDS; do
  if [[ "$backend" != "hermesagent" ]]; then
    echo "safe_choice skill smoke currently supports hermesagent only; got $backend" >&2
    exit 2
  fi
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue \
    --ready-json "$READY_JSON" \
    --results-root "$INPUT_DIR" \
    --output-dir "$OUTPUT_ROOT/$backend" \
    --backend hermesagent \
    --model "$HERMES_MODEL" \
    --base-url "$COMPAT_BASE_URL" \
    --description-field description_highlevel \
    --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language \
    --protocol-surface safe_choice \
    --self-evolution-mode skill \
    --skill-word-limit "$SKILL_WORD_LIMIT" \
    --choice-count 4 \
    --choice-seed "$SEED" \
    --write-human-trace-md \
    --generic-invalid-action-feedback \
    --parallel-jobs "$PARALLEL_JOBS" \
    --timeout-sec "$TIMEOUT_SEC" \
    --max-iterations "$MAX_ITERATIONS" \
    --l1-max-steps "$L1_MAX_STEPS" \
    --l2-max-steps "$L2_MAX_STEPS" \
    --l3-max-steps "$L3_MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --atomic-template-path "$ATOMIC_TEMPLATE_PATH" \
    --subtask-template-path "$SUBTASK_TEMPLATE_PATH" \
    "${resume_args[@]}" \
    "${task_args[@]}" \
    "$@"
done
