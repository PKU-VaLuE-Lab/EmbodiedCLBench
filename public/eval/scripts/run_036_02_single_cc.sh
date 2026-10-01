#!/usr/bin/env bash
set -euo pipefail

API_KEY="${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY to your DashScope API key.}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"

PYTHONPATH="$REPO_ROOT/src" python -m tongbench_eval.cli.framework.ready_single_tasks \
  --ready-json "$PROJECT_ROOT/important_results_eval/8.10/input/ready_l2_tasks_with_images.json" \
  --output-dir "$PROJECT_ROOT/important_results_eval/8.10/output/cc_single" \
  --backend claudecode \
  --model qwen3.7-plus \
  --base-url https://dashscope.aliyuncs.com/apps/anthropic \
  --api-key "$API_KEY" \
  --framework-interaction-mode semantic_tool \
  --public-interface-mode natural_language \
  --temperature 0 \
  --seed 42 \
  --timeout-sec 300 \
  --l1-max-steps 10 \
  --l2-max-steps 20 \
  --max-iterations 180 \
  --action-interface library_factorized \
  --action-library-mode atomic_only \
  --atomic-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json" \
  --subtask-template-path "$PROJECT_ROOT/important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json" \
  --generic-invalid-action-feedback \
  "$@"
