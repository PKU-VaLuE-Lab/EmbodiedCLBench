#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Missing DASHSCOPE_API_KEY. Export it before running this smoke script." >&2
  exit 1
fi
export DASHSCOPE_API_KEY
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"
export DOCKER_IMAGE_CLAUDECODE="${DOCKER_IMAGE_CLAUDECODE:-wildclawbench-claudecode-ubuntu:v0.2}"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-$PROJECT_ROOT/output/036_02_evo_repro/eval_smoke/framework_level_smoke}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE_OUTPUT_ROOT/$RUN_TAG}"

L2_REUSE_ROOT="${L2_REUSE_ROOT:-$PROJECT_ROOT/output/036_02_evo_repro/outputs_compositional_2/036_02/generated_from_imaged_l1/image_reuse_l2_only}"
L3_REUSE_ROOT="${L3_REUSE_ROOT:-$PROJECT_ROOT/output/036_02_evo_repro/outputs_compositional_2/036_02/generated_from_imaged_l1/image_reuse_l3_partial}"
STAGED_ROOT="${STAGED_ROOT:-$PROJECT_ROOT/output/036_02_evo_repro/eval_smoke/staged_images}"

ATOMIC_TEMPLATE_PATH="${ATOMIC_TEMPLATE_PATH:-$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json}"
SUBTASK_TEMPLATE_PATH="${SUBTASK_TEMPLATE_PATH:-$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json}"

COMPAT_BASE_URL="${COMPAT_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
CLAUDECODE_BASE_URL="${CLAUDECODE_BASE_URL:-https://dashscope.aliyuncs.com/apps/anthropic}"

BACKENDS="${BACKENDS:-hermesagent codex claudecode openclaw}"
LEVELS="${LEVELS:-L1 L2 L3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60}"
TEMPERATURE="${TEMPERATURE:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

task_id_for_level() {
  case "$1" in
    L1) echo "task_L1_001" ;;
    L2) echo "task_L2_002" ;;
    L3) echo "task_L3_002" ;;
    *) echo "Unsupported level: $1" >&2; return 2 ;;
  esac
}

task_dir_for_level() {
  local level="$1"
  local task_id="$2"
  case "$level" in
    L1) echo "$L2_REUSE_ROOT/graph/${task_id}_atomic_expanded_graph.json" ;;
    L2) echo "$L2_REUSE_ROOT/tasks/$task_id/input/${task_id}_atomic_expanded_graph.json" ;;
    L3) echo "$L3_REUSE_ROOT/tasks/$task_id/input/${task_id}_atomic_expanded_graph.json" ;;
    *) echo "Unsupported level: $level" >&2; return 2 ;;
  esac
}

image_dir_for_level() {
  local level_lower
  level_lower="$(tr '[:upper:]' '[:lower:]' <<<"$1")"
  echo "$STAGED_ROOT/$level_lower/$2"
}

max_steps_for_level() {
  case "$1" in
    L1) echo "${L1_MAX_STEPS:-4}" ;;
    L2) echo "${L2_MAX_STEPS:-8}" ;;
    L3) echo "${L3_MAX_STEPS:-12}" ;;
    *) echo "Unsupported level: $1" >&2; return 2 ;;
  esac
}

model_for_backend() {
  case "$1" in
    hermesagent) echo "${HERMES_MODEL:-qwen3-vl-flash}" ;;
    codex) echo "${CODEX_MODEL:-qwen3.7-plus}" ;;
    claudecode) echo "${CLAUDECODE_MODEL:-qwen3.7-plus}" ;;
    openclaw) echo "${OPENCLAW_MODEL:-qwen3-vl-flash}" ;;
    *) echo "Unsupported backend: $1" >&2; return 2 ;;
  esac
}

base_url_for_backend() {
  case "$1" in
    claudecode) echo "$CLAUDECODE_BASE_URL" ;;
    *) echo "$COMPAT_BASE_URL" ;;
  esac
}

backend_extra_args() {
  case "$1" in
    codex)
      printf '%s\n' --codex-direct-image-input --codex-mcp-only-tools
      ;;
    claudecode)
      return 0
      ;;
    openclaw)
      printf '%s\n' --codex-direct-image-input
      ;;
    hermesagent)
      ;;
    *)
      echo "Unsupported backend: $1" >&2
      return 2
      ;;
  esac
}

mkdir -p "$OUTPUT_ROOT"

for backend in $BACKENDS; do
  for level in $LEVELS; do
    task_id="$(task_id_for_level "$level")"
    task_dir="$(task_dir_for_level "$level" "$task_id")"
    image_dir="$(image_dir_for_level "$level" "$task_id")"
    output_dir="$OUTPUT_ROOT/$backend/$level/$task_id"

    if [[ ! -f "$task_dir" ]]; then
      echo "Missing task graph: $task_dir" >&2
      exit 1
    fi
    if [[ ! -f "$image_dir/node_manifest.json" ]]; then
      echo "Missing staged image manifest: $image_dir/node_manifest.json" >&2
      exit 1
    fi

    mapfile -t extra_args < <(backend_extra_args "$backend")
    echo "[$backend][$level] $task_id -> $output_dir"
    PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m tongbench_eval.cli.framework.single_task \
      --task-dir "$task_dir" \
      --unique-state-image-dir "$image_dir" \
      --output-dir "$output_dir" \
      --backend "$backend" \
      --model "$(model_for_backend "$backend")" \
      --base-url "$(base_url_for_backend "$backend")" \
      --api-key "$DASHSCOPE_API_KEY" \
      --timeout-sec "$TIMEOUT_SEC" \
      --max-steps "$(max_steps_for_level "$level")" \
      --max-iterations "$MAX_ITERATIONS" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --framework-interaction-mode semantic_tool \
      --public-interface-mode natural_language \
      --action-interface library_factorized \
      --action-library-mode atomic_only \
      --atomic-template-path "$ATOMIC_TEMPLATE_PATH" \
      --subtask-template-path "$SUBTASK_TEMPLATE_PATH" \
      --generic-invalid-action-feedback \
      "${extra_args[@]}"
  done
done

echo "$OUTPUT_ROOT"
