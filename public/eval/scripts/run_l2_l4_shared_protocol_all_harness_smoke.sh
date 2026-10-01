#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/output/qwen3_7_plus_all_harness_checkpoint_v2}"
LEARNING_PLAN="${LEARNING_PLAN:-$PROJECT_ROOT/important_results_taskgen/8.23_l2_l4_pilot_v1/design/learning_plan.json}"
LEARNING_CHECKPOINT_ROOT="${LEARNING_CHECKPOINT_ROOT:-$OUTPUT_ROOT/_shared_learning_checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
MODEL="${MODEL:-qwen3.7-plus}"
TASK_ID="${TASK_ID:-task_L4_HL_v4_child_0002}"
ROOM="${ROOM:-005}"
BACKENDS="${BACKENDS:-hermesagent codex claudecode openclaw}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
CLAUDECODE_BASE_URL="${CLAUDECODE_BASE_URL:-https://dashscope.aliyuncs.com/apps/anthropic}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-260}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"

API_KEY="${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$API_KEY" ]] || { echo "DASHSCOPE_API_KEY or OPENAI_API_KEY is required" >&2; exit 2; }
export DASHSCOPE_API_KEY="$API_KEY"
export OPENAI_API_KEY="$API_KEY"
export OPENROUTER_API_KEY="$API_KEY"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"

base_args=(
  --results-root "$INPUT_ROOT/scene_$ROOM"
  --model "$MODEL"
  --description-field description_highlevel_hard
  --framework-interaction-mode semantic_tool
  --public-interface-mode natural_language
  --protocol-surface safe_choice
  --choice-bank-jsonl "$INPUT_ROOT/scene_$ROOM/option_bank_highlevel_v2_smoke.jsonl"
  --choice-count 4
  --choice-seed "$SEED"
  --write-human-trace-md
  --generic-invalid-action-feedback
  --temperature "$TEMPERATURE"
  --seed "$SEED"
  --l2-max-steps 12
  --l4-max-steps 24
  --l2-max-steps-factor 1.5
  --l4-max-steps-factor 2.0
  --atomic-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json"
  --subtask-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json"
  --timeout-sec "$TIMEOUT_SEC"
  --max-iterations "$MAX_ITERATIONS"
  --skip-existing
)

backend_args() {
  local backend="$1"
  printf '%s\n' --backend "$backend" --api-key "$API_KEY"
  case "$backend" in
    hermesagent)
      printf '%s\n' --base-url "$COMPAT_BASE_URL"
      ;;
    codex)
      printf '%s\n' --base-url "$COMPAT_BASE_URL" --codex-direct-image-input --codex-mcp-only-tools
      ;;
    claudecode)
      printf '%s\n' --base-url "$CLAUDECODE_BASE_URL"
      ;;
    openclaw)
      printf '%s\n' --base-url "$COMPAT_BASE_URL" --codex-direct-image-input
      ;;
    *)
      echo "Unsupported backend: $backend" >&2
      exit 2
      ;;
  esac
}

run_zero_shot() {
  local backend="$1"
  local args=()
  while IFS= read -r value; do args+=("$value"); done < <(backend_args "$backend")
  echo "[$backend] zero_shot_l4 $TASK_ID"
  "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks \
    --ready-json "$INPUT_ROOT/scene_$ROOM/ready_l4_tasks_with_images.json" \
    --task-id "$TASK_ID" \
    --output-dir "$OUTPUT_ROOT/$backend/zero_shot_l4" \
    --composite-only \
    --parallel-jobs 1 \
    "${base_args[@]}" \
    "${args[@]}"
}

run_dialogue_mode() {
  local backend="$1"
  local mode="$2"
  local args=()
  while IFS= read -r value; do args+=("$value"); done < <(backend_args "$backend")
  echo "[$backend] ${mode}_l4 $TASK_ID"
  "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue \
    --ready-json "$INPUT_ROOT/scene_$ROOM/ready_l4_tasks_with_images.json" \
    --task-id "$TASK_ID" \
    --output-dir "$OUTPUT_ROOT/$backend/${mode}_l4" \
    --hide-decomposition \
    --self-evolution-mode "$mode" \
    --learning-plan-json "$LEARNING_PLAN" \
    --learning-checkpoint-root "$LEARNING_CHECKPOINT_ROOT" \
    --parallel-jobs 1 \
    "${base_args[@]}" \
    "${args[@]}"
}

for backend in $BACKENDS; do
  run_zero_shot "$backend"
  run_dialogue_mode "$backend" in_context
  run_dialogue_mode "$backend" skill
done

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$TASK_ID" $BACKENDS <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
task_id = sys.argv[2]
backends = sys.argv[3:]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _first_task_summary(path: Path) -> dict:
    payload = _load_json(path)
    tasks = payload.get("tasks") or []
    if not tasks:
        raise ValueError(f"No tasks in {path}")
    return tasks[0]


for backend in backends:
    icl_task = _first_task_summary(root / backend / "in_context_l4" / "summary.json")
    skill_task = _first_task_summary(root / backend / "skill_l4" / "summary.json")
    icl_checkpoint = icl_task.get("learning_checkpoint") or {}
    skill_checkpoint = skill_task.get("learning_checkpoint") or {}
    icl_payload = icl_checkpoint.get("payload_sha256")
    skill_payload = skill_checkpoint.get("payload_sha256")
    if not icl_payload or not skill_payload:
        raise SystemExit(f"{backend}: missing learning checkpoint payload in dialogue summary")
    if icl_payload != skill_payload:
        raise SystemExit(f"{backend}: ICL and Skill use different base learning checkpoints")

    icl_application_checkpoint = icl_task.get("in_context_checkpoint") or {}
    source_payload = (icl_application_checkpoint.get("metadata") or {}).get("source_checkpoint_payload_sha256")
    if source_payload and source_payload != icl_payload:
        raise SystemExit(f"{backend}: ICL derived checkpoint does not reference the base learning checkpoint")

    for label, task in (("in_context", icl_task), ("skill", skill_task)):
        validation = None
        if label == "in_context":
            validation = task.get("checkpoint_branch_validation")
        else:
            validation = (task.get("skill_summary_call") or {}).get("checkpoint_branch_validation")
        if isinstance(validation, dict) and validation.get("equivalent") is False:
            raise SystemExit(f"{backend}: {label} checkpoint branch validation failed")

    print(f"{backend}: shared base checkpoint {icl_payload}")
PY

echo "All-harness checkpoint smoke complete: $OUTPUT_ROOT"
