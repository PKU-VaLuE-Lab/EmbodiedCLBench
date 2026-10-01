#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/output/qwen3_7_plus}"
LEARNING_PLAN="${LEARNING_PLAN:-$PROJECT_ROOT/important_results_taskgen/8.23_l2_l4_pilot_v1/design/learning_plan.json}"
LEARNING_CHECKPOINT_ROOT="${LEARNING_CHECKPOINT_ROOT:-$OUTPUT_ROOT/shared_learning_checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
HERMES_MODEL="${HERMES_MODEL:-qwen3.7-plus}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
RUN_MODES="${RUN_MODES:-basic_l2 zero_shot_l4}"
PARALLEL_JOBS="${PARALLEL_JOBS:-5}"
SELF_EVOLUTION_PARALLEL_JOBS="${SELF_EVOLUTION_PARALLEL_JOBS:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
SELF_EVOLUTION_TIMEOUT_SEC="${SELF_EVOLUTION_TIMEOUT_SEC:-3600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-180}"
SELF_EVOLUTION_MAX_ITERATIONS="${SELF_EVOLUTION_MAX_ITERATIONS:-260}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a ROOMS <<< "${ROOM_LIST:-005 036_02}"

[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
[[ -n "${DASHSCOPE_API_KEY:-${OPENAI_API_KEY:-}}" ]] || {
  echo "DASHSCOPE_API_KEY or OPENAI_API_KEY is required" >&2
  exit 2
}
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-$OPENAI_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
MODELS_CONFIG="$OUTPUT_ROOT/models_dashscope_env.json"
"$PYTHON_BIN" - "$MODELS_CONFIG" "$COMPAT_BASE_URL" "$HERMES_MODEL" <<'PY'
import json, os, sys
path, base_url, model = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"providers": {"dashscope": {"apiKey": os.environ["DASHSCOPE_API_KEY"], "baseUrl": base_url, "models": [{"id": model, "name": model}]}}}, handle, indent=2)
PY
chmod 600 "$MODELS_CONFIG" || true

common_args() {
  local room="$1"
  local l2_max_steps_factor="$2"
  printf '%s\n' \
    --results-root "$INPUT_ROOT/scene_$room" \
    --backend hermesagent --model "$HERMES_MODEL" --models-config "$MODELS_CONFIG" --base-url "$COMPAT_BASE_URL" \
    --description-field description_highlevel_hard --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language --protocol-surface safe_choice \
    --choice-bank-jsonl "$INPUT_ROOT/scene_$room/option_bank_highlevel_v2_smoke.jsonl" \
    --choice-count 4 --choice-seed "$SEED" --write-human-trace-md --generic-invalid-action-feedback \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --l2-max-steps 12 --l4-max-steps 24 --l2-max-steps-factor "$l2_max_steps_factor" --l4-max-steps-factor 2.0 \
    --atomic-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json" \
    --subtask-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json"
}

run_single() {
  local room="$1" level="$2" mode="$3" output="$OUTPUT_ROOT/$mode/scene_$room"
  local args=()
  local extra_args=()
  while IFS= read -r value; do args+=("$value"); done < <(common_args "$room" 2.0)
  [[ "$DRY_RUN" == "1" ]] && extra_args+=(--dry-run)
  "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks \
    --ready-json "$INPUT_ROOT/scene_$room/ready_${level,,}_tasks_with_images.json" \
    --output-dir "$output" --composite-only --parallel-jobs "$PARALLEL_JOBS" \
    --timeout-sec "$TIMEOUT_SEC" --max-iterations "$MAX_ITERATIONS" \
    --skip-existing "${extra_args[@]}" "${args[@]}"
}

run_dialogue() {
  local room="$1" self_mode="$2" mode="$3" output="$OUTPUT_ROOT/$mode/scene_$room"
  local args=()
  local extra_args=()
  while IFS= read -r value; do args+=("$value"); done < <(common_args "$room" 1.5)
  [[ "$DRY_RUN" == "1" ]] && extra_args+=(--dry-run)
  "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue \
    --ready-json "$INPUT_ROOT/scene_$room/ready_l4_tasks_with_images.json" \
    --output-dir "$output" --hide-decomposition --self-evolution-mode "$self_mode" \
    --learning-plan-json "$LEARNING_PLAN" \
    --learning-checkpoint-root "$LEARNING_CHECKPOINT_ROOT/scene_$room" \
    --parallel-jobs "$SELF_EVOLUTION_PARALLEL_JOBS" \
    --timeout-sec "$SELF_EVOLUTION_TIMEOUT_SEC" --max-iterations "$SELF_EVOLUTION_MAX_ITERATIONS" \
    --skip-existing "${extra_args[@]}" "${args[@]}"
}

for mode in $RUN_MODES; do
  for room in "${ROOMS[@]}"; do
    case "$mode" in
      basic_l2) run_single "$room" L2 "$mode" ;;
      zero_shot_l4) run_single "$room" L4 "$mode" ;;
      in_context_l4) run_dialogue "$room" in_context "$mode" ;;
      skill_l4) run_dialogue "$room" skill "$mode" ;;
      *) echo "Unknown mode: $mode" >&2; exit 2 ;;
    esac
  done
done

echo "L2/L4 pilot eval complete: $OUTPUT_ROOT"
