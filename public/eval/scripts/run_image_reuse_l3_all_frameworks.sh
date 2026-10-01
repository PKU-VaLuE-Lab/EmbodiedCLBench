#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
IMAGE_REUSE_ROOT="${IMAGE_REUSE_ROOT:-$PROJECT_ROOT/important_results_eval/8.10/input/image_reuse_l3_partial}"
READY_JSON="${READY_JSON:-$IMAGE_REUSE_ROOT/ready_l2_l3_tasks_with_images.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$IMAGE_REUSE_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$IMAGE_REUSE_ROOT/outputs_l3}"
API_KEY="${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY to your DashScope API key.}"
export DASHSCOPE_API_KEY="$API_KEY"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$API_KEY}"

BACKENDS="${BACKENDS:-hermesagent codex claudecode openclaw}"
RUN_DIALOGUE="${RUN_DIALOGUE:-1}"
RUN_SINGLE="${RUN_SINGLE:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-${SINGLE_SKIP_EXISTING:-1}}"
FORCE="${FORCE:-0}"
L3_TASK_LIMIT="${L3_TASK_LIMIT-${TASK_LIMIT-20}}"
DESCRIPTION_FIELD="${DESCRIPTION_FIELD:-description}"
HIDE_DECOMPOSITION="${HIDE_DECOMPOSITION:-0}"

HERMES_MODEL="${HERMES_MODEL:-qwen3-vl-flash}"
CODEX_MODEL="${CODEX_MODEL:-qwen3.7-plus}"
CLAUDECODE_MODEL="${CLAUDECODE_MODEL:-qwen3.7-plus}"
OPENCLAW_MODEL="${OPENCLAW_MODEL:-qwen3-vl-flash}"
COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
CLAUDECODE_BASE_URL="${CLAUDECODE_BASE_URL:-https://dashscope.aliyuncs.com/apps/anthropic}"
OPENCLAW_BASE_URL="${OPENCLAW_BASE_URL:-$COMPAT_BASE_URL}"

L1_MAX_STEPS="${L1_MAX_STEPS:-6}"
L2_MAX_STEPS="${L2_MAX_STEPS:-12}"
L3_MAX_STEPS="${L3_MAX_STEPS:-18}"
HERMES_TIMEOUT_SEC="${HERMES_TIMEOUT_SEC:-${TIMEOUT_SEC:-900}}"
CODEX_TIMEOUT_SEC="${CODEX_TIMEOUT_SEC:-${TIMEOUT_SEC:-900}}"
CLAUDECODE_TIMEOUT_SEC="${CLAUDECODE_TIMEOUT_SEC:-${TIMEOUT_SEC:-900}}"
OPENCLAW_TIMEOUT_SEC="${OPENCLAW_TIMEOUT_SEC:-${TIMEOUT_SEC:-900}}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
PARALLEL_JOBS="${PARALLEL_JOBS:-1}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0}"

ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json}"
SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json}"

MCP_IMAGE_MAX_SIDE="${MCP_IMAGE_MAX_SIDE:-640}"
MCP_IMAGE_MAX_BASE64_CHARS="${MCP_IMAGE_MAX_BASE64_CHARS:-50000}"
MCP_IMAGE_JPEG_QUALITY="${MCP_IMAGE_JPEG_QUALITY:-35}"

mkdir -p "$OUTPUT_ROOT/ready" "$OUTPUT_ROOT/dialogue" "$OUTPUT_ROOT/single" "$OUTPUT_ROOT/summary"
FILTERED_READY="$OUTPUT_ROOT/ready/ready_l3_tasks_with_images.json"

python - "$READY_JSON" "$FILTERED_READY" "L3" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
wanted = sys.argv[3].upper()
payload = json.loads(src.read_text(encoding="utf-8"))

if isinstance(payload, list):
    items = payload
    key = None
elif isinstance(payload, dict):
    key = next((name for name in ("tasks", "ready_tasks", "items") if isinstance(payload.get(name), list)), None)
    if key is None:
        raise SystemExit(f"Unsupported ready JSON structure: {src}")
    items = payload[key]
else:
    raise SystemExit(f"Unsupported ready JSON structure: {src}")

def level_of(item):
    if not isinstance(item, dict):
        return ""
    for name in ("level", "task_level", "difficulty_level", "composite_level"):
        value = str(item.get(name) or "").upper()
        if value in {"L1", "L2", "L3"}:
            return value
    task_id = str(item.get("task_id") or item.get("id") or "")
    lowered = task_id.lower()
    for level in ("l1", "l2", "l3"):
        if lowered.startswith(f"task_{level}_") or f"/task_{level}_" in lowered or f"\\task_{level}_" in lowered:
            return level.upper()
    return ""

filtered = [item for item in items if level_of(item) == wanted]
if not filtered:
    raise SystemExit(f"No {wanted} tasks found in {src}")

if key is None:
    output = filtered
else:
    output = dict(payload)
    output[key] = filtered
    output["filtered_level"] = wanted
    output["filtered_task_count"] = len(filtered)

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(filtered)} {wanted} tasks to {dst}")
PY

common_args=(
  --ready-json "$FILTERED_READY"
  --results-root "$RESULTS_ROOT"
  --framework-interaction-mode semantic_tool
  --public-interface-mode natural_language
  --temperature "$TEMPERATURE"
  --seed "$SEED"
  --l1-max-steps "$L1_MAX_STEPS"
  --l2-max-steps "$L2_MAX_STEPS"
  --l3-max-steps "$L3_MAX_STEPS"
  --max-iterations "$MAX_ITERATIONS"
  --action-interface library_factorized
  --action-library-mode atomic_only
  --atomic-template-path "$ATOMIC_TEMPLATE_PATH"
  --subtask-template-path "$SUBTASK_TEMPLATE_PATH"
  --generic-invalid-action-feedback
  --description-field "$DESCRIPTION_FIELD"
  --parallel-jobs "$PARALLEL_JOBS"
)

resume_args=()
if [[ "$SKIP_EXISTING" == "1" ]]; then
  resume_args+=(--skip-existing)
fi
if [[ "$FORCE" == "1" ]]; then
  resume_args+=(--force)
fi

limit_args=()
if [[ -n "$L3_TASK_LIMIT" ]]; then
  limit_args+=(--limit "$L3_TASK_LIMIT")
fi

dialogue_roots=()
single_roots=()
failures=()
dialogue_extra_args=()
if [[ "$HIDE_DECOMPOSITION" == "1" ]]; then
  dialogue_extra_args+=(--hide-decomposition)
fi

for backend in $BACKENDS; do
  backend_args=()
  case "$backend" in
    hermesagent)
      backend_args=(--backend hermesagent --model "$HERMES_MODEL" --base-url "$COMPAT_BASE_URL" --timeout-sec "$HERMES_TIMEOUT_SEC")
      ;;
    codex)
      backend_args=(--backend codex --model "$CODEX_MODEL" --base-url "$COMPAT_BASE_URL" --timeout-sec "$CODEX_TIMEOUT_SEC" --codex-direct-image-input --codex-mcp-only-tools)
      ;;
    claudecode)
      backend_args=(
        --backend claudecode
        --model "$CLAUDECODE_MODEL"
        --base-url "$CLAUDECODE_BASE_URL"
        --timeout-sec "$CLAUDECODE_TIMEOUT_SEC"
      )
      ;;
    openclaw)
      backend_args=(--backend openclaw --model "$OPENCLAW_MODEL" --base-url "$OPENCLAW_BASE_URL" --timeout-sec "$OPENCLAW_TIMEOUT_SEC" --codex-direct-image-input)
      ;;
    *)
      echo "Unknown backend: $backend" >&2
      exit 2
      ;;
  esac

  if [[ "$RUN_DIALOGUE" == "1" ]]; then
    out_dir="$OUTPUT_ROOT/dialogue/$backend"
    dialogue_roots+=("$out_dir")
    if ! PYTHONPATH="$REPO_ROOT/src" python -m tongbench_eval.cli.framework.dialogue \
      "${common_args[@]}" \
	      "${resume_args[@]}" \
	      "${limit_args[@]}" \
	      "${dialogue_extra_args[@]}" \
	      --output-dir "$out_dir" \
	      "${backend_args[@]}" \
      "$@"; then
      failures+=("dialogue/$backend")
    fi
  fi

  if [[ "$RUN_SINGLE" == "1" ]]; then
    out_dir="$OUTPUT_ROOT/single/$backend"
    single_roots+=("$out_dir")
    if ! PYTHONPATH="$REPO_ROOT/src" python -m tongbench_eval.cli.framework.ready_single_tasks \
      "${common_args[@]}" \
      "${resume_args[@]}" \
      "${limit_args[@]}" \
      --output-dir "$out_dir" \
      "${backend_args[@]}" \
      "$@"; then
      failures+=("single/$backend")
    fi
  fi
done

if [[ "$RUN_SUMMARY" == "1" ]]; then
  if [[ "${#dialogue_roots[@]}" -gt 0 ]]; then
    PYTHONPATH="$REPO_ROOT/src" python -m tongbench_eval.cli.reports.summarize_compositional_steps \
      "${dialogue_roots[@]}" \
      --expected-steps 4 \
      --csv-out "$OUTPUT_ROOT/summary/dialogue_step_scores.csv" \
      --json-out "$OUTPUT_ROOT/summary/dialogue_step_scores.json"
  fi

  if [[ "${#single_roots[@]}" -gt 0 ]]; then
    PYTHONPATH="$REPO_ROOT/src" python -m tongbench_eval.cli.reports.summarize_single_levels \
      "${single_roots[@]}" \
      --levels L1 L3 \
      --csv-out "$OUTPUT_ROOT/summary/single_level_scores.csv" \
      --json-out "$OUTPUT_ROOT/summary/single_level_scores.json"
  fi
fi

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Completed with failed runs:" >&2
  printf '  %s\n' "${failures[@]}" >&2
  exit 1
fi
