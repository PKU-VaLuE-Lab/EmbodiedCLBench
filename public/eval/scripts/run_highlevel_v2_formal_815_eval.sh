#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
DATE_LABEL="${DATE_LABEL:-8.15}"
FORMAL_NAME="${FORMAL_NAME:-highlevel_v2_formal_036_02}"
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/input/formal/$FORMAL_NAME}"
OUTPUT_BASE="${OUTPUT_BASE:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/output/formal/${FORMAL_NAME}_qwen3vl_flash}"
REPORT_DIR="${REPORT_DIR:-$PROJECT_ROOT/important_results_eval/$DATE_LABEL/report/formal/${FORMAL_NAME}_qwen3vl_flash}"
SECRET_FILE="${SECRET_FILE:-}"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$DEFAULT_PYTHON" ]]; then
    PYTHON_BIN="$DEFAULT_PYTHON"
  else
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Cannot find python3 or python." >&2
  exit 2
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

READY_L3="${READY_L3:-$INPUT_DIR/ready_l3_tasks_with_images.json}"
READY_L6="${READY_L6:-$INPUT_DIR/ready_l6_tasks_with_images.json}"
OPTION_BANK_JSONL="${OPTION_BANK_JSONL:-$INPUT_DIR/option_bank_highlevel_v2_smoke.jsonl}"
ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$INPUT_DIR/atomic_templates_updated_env_v3.json}"
SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$INPUT_DIR/subtask_templates_compressed_updated_env_v3.json}"
if [[ ! -f "$ATOMIC_TEMPLATE_PATH" ]]; then
  ATOMIC_TEMPLATE_PATH="$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json"
fi
if [[ ! -f "$SUBTASK_TEMPLATE_PATH" ]]; then
  SUBTASK_TEMPLATE_PATH="$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json"
fi

HERMES_MODEL="${HERMES_MODEL:-qwen3-vl-flash}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
RUN_MODES="${RUN_MODES:-basic_l3 language_bias_l3 zero_shot_l6 in_context_l6 skill_l6 report}"
TASK_LIMIT_L3="${TASK_LIMIT_L3:-12}"
TASK_LIMIT_L6="${TASK_LIMIT_L6:-10}"
SELF_EVOLUTION_LIMIT="${SELF_EVOLUTION_LIMIT:-10}"
LEARNING_TASK_COUNT="${LEARNING_TASK_COUNT:-3}"
RELATED_LEARNING_MAX_COUNT="${RELATED_LEARNING_MAX_COUNT:-3}"
USE_EXPLICIT_LEARNING_TASK_IDS="${USE_EXPLICIT_LEARNING_TASK_IDS:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-4}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
SELF_EVOLUTION_TIMEOUT_SEC="${SELF_EVOLUTION_TIMEOUT_SEC:-3600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-180}"
SELF_EVOLUTION_MAX_ITERATIONS="${SELF_EVOLUTION_MAX_ITERATIONS:-260}"
MAX_STEPS_FACTOR="${MAX_STEPS_FACTOR:-2.0}"
L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
L6_MAX_STEPS="${L6_MAX_STEPS:-36}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"
FORCE="${FORCE:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$OUTPUT_BASE" "$REPORT_DIR"

MODELS_CONFIG="${MODELS_CONFIG:-$OUTPUT_BASE/models_dashscope_env.json}"
cat > "$MODELS_CONFIG" <<JSON
{
  "providers": {
    "dashscope": {
      "apiKey": "\${DASHSCOPE_API_KEY}",
      "baseUrl": "$COMPAT_BASE_URL",
      "models": [
        {
          "id": "$HERMES_MODEL",
          "name": "$HERMES_MODEL"
        }
      ]
    }
  }
}
JSON
chmod 600 "$MODELS_CONFIG" || true

resume_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  resume_args+=(--skip-existing)
fi
if [[ "$FORCE" == "1" ]]; then
  resume_args+=(--force)
fi

image_args=()

L3_TASK_IDS=()
if [[ "$USE_EXPLICIT_LEARNING_TASK_IDS" == "1" ]]; then
mapfile -t L3_TASK_IDS < <("$PYTHON_BIN" - "$READY_L3" "$LEARNING_TASK_COUNT" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
limit = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
items = payload if isinstance(payload, list) else payload.get("tasks", [])
count = 0
for item in items:
    task_id = str(item.get("task_id") or "").strip()
    if not task_id:
        continue
    print(task_id)
    count += 1
    if count >= limit:
        break
PY
)
fi
learning_args=()
for task_id in "${L3_TASK_IDS[@]}"; do
  learning_args+=(--learning-task-id "$task_id")
done

run_single_mode() {
  local mode="$1"
  local ready_json="$2"
  local output_dir="$3"
  local limit="$4"
  shift 4
  echo "Running $mode -> $output_dir"
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks \
    --ready-json "$ready_json" \
    --results-root "$INPUT_DIR" \
    --output-dir "$output_dir" \
    --backend hermesagent \
    --model "$HERMES_MODEL" \
    --models-config "$MODELS_CONFIG" \
    --base-url "$COMPAT_BASE_URL" \
    --description-field description_highlevel_hard \
    --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language \
    --protocol-surface safe_choice \
    --choice-bank-jsonl "$OPTION_BANK_JSONL" \
    --choice-count 4 \
    --choice-seed "$SEED" \
    --write-human-trace-md \
    --generic-invalid-action-feedback \
    --composite-only \
    --limit "$limit" \
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
    "${image_args[@]}" \
    "${resume_args[@]}" \
    "$@"
}

run_dialogue_mode() {
  local mode="$1"
  local self_mode="$2"
  local output_dir="$3"
  local mode_learning_args=()
  if [[ "$mode" == "in_context_l6" && "$USE_EXPLICIT_LEARNING_TASK_IDS" != "1" ]]; then
    mode_learning_args+=(--related-learning-ready-json "$READY_L3" --related-learning-max-count "$RELATED_LEARNING_MAX_COUNT")
  else
    mode_learning_args+=("${learning_args[@]}")
  fi
  echo "Running $mode -> $output_dir"
  PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue \
    --ready-json "$READY_L6" \
    --results-root "$INPUT_DIR" \
    --output-dir "$output_dir" \
    --backend hermesagent \
    --model "$HERMES_MODEL" \
    --models-config "$MODELS_CONFIG" \
    --base-url "$COMPAT_BASE_URL" \
    --description-field description_highlevel_hard \
    --hide-decomposition \
    --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language \
    --protocol-surface safe_choice \
    --self-evolution-mode "$self_mode" \
    --skill-word-limit 300 \
    --choice-bank-jsonl "$OPTION_BANK_JSONL" \
    --choice-count 4 \
    --choice-seed "$SEED" \
    --write-human-trace-md \
    --generic-invalid-action-feedback \
    --limit "$SELF_EVOLUTION_LIMIT" \
    --parallel-jobs "$SELF_EVOLUTION_PARALLEL_JOBS" \
    --timeout-sec "$SELF_EVOLUTION_TIMEOUT_SEC" \
    --max-iterations "$SELF_EVOLUTION_MAX_ITERATIONS" \
    --l3-max-steps "$L3_MAX_STEPS" \
    --l6-max-steps "$L6_MAX_STEPS" \
    --max-steps-factor "$MAX_STEPS_FACTOR" \
    --temperature "$TEMPERATURE" \
    --seed "$SEED" \
    --atomic-template-path "$ATOMIC_TEMPLATE_PATH" \
    --subtask-template-path "$SUBTASK_TEMPLATE_PATH" \
    "${image_args[@]}" \
    "${mode_learning_args[@]}" \
    "${resume_args[@]}"
}

MODE_FAILURES=()
run_eval_mode() {
  local mode="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "Mode $mode exited with status $rc; continuing so report generation can still run." >&2
    MODE_FAILURES+=("$mode:$rc")
  fi
}

for mode in $RUN_MODES; do
  case "$mode" in
    basic_l3)
      run_eval_mode basic_l3 run_single_mode basic_l3 "$READY_L3" "$OUTPUT_BASE/basic_l3/hermesagent" "$TASK_LIMIT_L3"
      ;;
    language_bias_l3)
      run_eval_mode language_bias_l3 run_single_mode language_bias_l3 "$READY_L3" "$OUTPUT_BASE/language_bias_l3/hermesagent" "$TASK_LIMIT_L3" --no-image-input
      ;;
    zero_shot_l6)
      run_eval_mode zero_shot_l6 run_single_mode zero_shot_l6 "$READY_L6" "$OUTPUT_BASE/zero_shot_l6/hermesagent" "$TASK_LIMIT_L6"
      ;;
    in_context_l6)
      run_eval_mode in_context_l6 run_dialogue_mode in_context_l6 in_context "$OUTPUT_BASE/in_context_l6/hermesagent"
      ;;
    skill_l6)
      run_eval_mode skill_l6 run_dialogue_mode skill_l6 skill "$OUTPUT_BASE/skill_l6/hermesagent"
      ;;
    report)
      PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.reports.summarize_highlevel_v2_formal_815 \
        --output-base "$OUTPUT_BASE" \
        --report-dir "$REPORT_DIR"
      ;;
    *)
      echo "Unknown RUN_MODES item: $mode" >&2
      exit 2
      ;;
  esac
done

echo "8.15 formal eval outputs: $OUTPUT_BASE"
echo "8.15 formal report: $REPORT_DIR"
if [[ "${#MODE_FAILURES[@]}" -gt 0 ]]; then
  printf 'Eval modes with non-zero exits: %s\n' "${MODE_FAILURES[*]}" >&2
  exit 1
fi
