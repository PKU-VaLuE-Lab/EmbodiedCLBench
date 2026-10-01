#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EVAL_ROOT/../../.." && pwd)"
LOCAL_VIS_ENV="$PROJECT_ROOT/.conda-visualization"

if [[ -x "$LOCAL_VIS_ENV/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$LOCAL_VIS_ENV/bin/python}"
  export PATH="$LOCAL_VIS_ENV/bin:$PATH"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

GRAPH_PATH="${GRAPH_PATH:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks/task_L2_010/input/task_L2_010_atomic_expanded_graph.json}"
SUCCESS_PATHS="${SUCCESS_PATHS:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks/task_L2_010/input/task_L2_010_success_paths.json}"
ATOMIC_TRANSITION_MANIFEST="${ATOMIC_TRANSITION_MANIFEST:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks/task_L2_010/input/task_L2_010_atomic_transition_manifest.json}"
UNIQUE_STATES="${UNIQUE_STATES:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks/task_L2_010/input/task_L2_010_unique_states.json}"
FULL_BRANCH_GRAPH="${FULL_BRANCH_GRAPH:-}"
IMAGE_MANIFEST="${IMAGE_MANIFEST:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks/task_L2_010/task_image_manifest.json}"
TRACE_PATH="${TRACE_PATH:-$PROJECT_ROOT/important_results_eval/8.11/output/formal/baseline/l2/dialogue/hermesagent/task_L2_010/task_runs/03_task_L2_010/task_output/trace.jsonl}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/important_results_eval/8.11/output/formal/visualizations/task_L2_010}"
PORT="${PORT:-8060}"
HOST="${HOST:-127.0.0.1}"

cd "$EVAL_ROOT"
EXPORT_ARGS=(
  --graph "$GRAPH_PATH"
  --success-paths "$SUCCESS_PATHS"
  --atomic-transition-manifest "$ATOMIC_TRANSITION_MANIFEST"
  --unique-states "$UNIQUE_STATES"
  --image-manifest "$IMAGE_MANIFEST"
  --trace "$TRACE_PATH"
  --out-dir "$OUT_DIR"
  --render-svg
)

if [[ -n "$FULL_BRANCH_GRAPH" ]]; then
  EXPORT_ARGS+=(--full-branch-graph "$FULL_BRANCH_GRAPH")
fi

PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.graph.visualize_task export \
  "${EXPORT_ARGS[@]}"

echo "Wrote visualization artifacts to $OUT_DIR"

if [[ "${SERVE:-0}" == "1" ]]; then
  PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.graph.visualize_task serve \
    --viz-json "$OUT_DIR/viz_graph.json" \
    --host "$HOST" \
    --port "$PORT"
fi
