#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../../.." && pwd)}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/input}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.23_l2_l4_pilot_v1/output/gpt5_5_medium}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
HERMES_MODEL="${HERMES_MODEL:-gpt-5.5}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://api.ikuncode.cc/v1}"
RUN_MODES="${RUN_MODES:-basic_l2 zero_shot_l4}"
PARALLEL_JOBS="${PARALLEL_JOBS:-12}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
MAX_ITERATIONS="${MAX_ITERATIONS:-220}"
SEED="${SEED:-42}"
THINKING="${THINKING:-medium}"
DRY_RUN="${DRY_RUN:-0}"
HERMES_API_CALL_STALE_TIMEOUT="${HERMES_API_CALL_STALE_TIMEOUT:-1200}"
export HERMES_API_CALL_STALE_TIMEOUT
read -r -a ROOMS <<< "${ROOM_LIST:-005 036_02}"

[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
EVAL_API_KEY="${EVAL_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$EVAL_API_KEY" ]] || {
  echo "EVAL_API_KEY or OPENAI_API_KEY is required" >&2
  exit 2
}
export EVAL_API_KEY OPENAI_API_KEY="$EVAL_API_KEY"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
MODELS_CONFIG="$OUTPUT_ROOT/models_ikuncode_env.json"
"$PYTHON_BIN" - "$MODELS_CONFIG" "$COMPAT_BASE_URL" "$HERMES_MODEL" <<'PY'
import json, os, sys
path, base_url, model = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"providers": {"ikuncode": {"apiKey": os.environ["EVAL_API_KEY"], "baseUrl": base_url, "models": [{"id": model, "name": model}]}}}, handle, indent=2)
PY
chmod 600 "$MODELS_CONFIG" || true

common_args() {
  local room="$1"
  printf '%s\n' \
    --results-root "$INPUT_ROOT/scene_$room" \
    --backend hermesagent --model "$HERMES_MODEL" --models-config "$MODELS_CONFIG" --base-url "$COMPAT_BASE_URL" \
    --description-field description_highlevel_hard --framework-interaction-mode semantic_tool \
    --public-interface-mode natural_language --protocol-surface safe_choice \
    --choice-bank-jsonl "$INPUT_ROOT/scene_$room/option_bank_highlevel_v2_smoke.jsonl" \
    --choice-count 4 --choice-seed "$SEED" --write-human-trace-md --generic-invalid-action-feedback \
    --no-model-temperature --no-model-seed --thinking "$THINKING" --responses-non-stream \
    --l2-max-steps 12 --l4-max-steps 24 --l2-max-steps-factor 2.0 --l4-max-steps-factor 2.0 \
    --atomic-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json" \
    --subtask-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json"
}

run_single() {
  local room="$1" level="$2" mode="$3" output="$OUTPUT_ROOT/$mode/scene_$room"
  local args=()
  local extra_args=()
  while IFS= read -r value; do args+=("$value"); done < <(common_args "$room")
  [[ "$DRY_RUN" == "1" ]] && extra_args+=(--dry-run)
  "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks \
    --ready-json "$INPUT_ROOT/scene_$room/ready_${level,,}_tasks_with_images.json" \
    --output-dir "$output" --composite-only --parallel-jobs "$PARALLEL_JOBS" \
    --timeout-sec "$TIMEOUT_SEC" --max-iterations "$MAX_ITERATIONS" \
    --skip-existing "${extra_args[@]}" "${args[@]}"
}

for mode in $RUN_MODES; do
  for room in "${ROOMS[@]}"; do
    case "$mode" in
      basic_l2) run_single "$room" L2 "$mode" ;;
      zero_shot_l4) run_single "$room" L4 "$mode" ;;
      *) echo "GPT-5.5 pilot runner only supports basic_l2 and zero_shot_l4. Unknown mode: $mode" >&2; exit 2 ;;
    esac
  done
done

echo "L2/L4 GPT-5.5 pilot eval complete: $OUTPUT_ROOT"
