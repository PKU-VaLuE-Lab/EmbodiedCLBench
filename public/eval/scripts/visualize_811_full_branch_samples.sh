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

GRAPH_ROOT="${GRAPH_ROOT:-$PROJECT_ROOT/important_results_taskgen/8.11/output/full_branch_visualization_samples/graph}"
EVAL_INPUT_ROOT="${EVAL_INPUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.11/input/tasks}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/important_results_eval/8.11/output/formal/visualizations/full_branch_samples}"
TASK_IDS="${TASK_IDS:-task_L1_053 task_L2_010 task_L3_003}"
RENDER_SVG="${RENDER_SVG:-1}"

trace_for_task() {
  local task_id="$1"
  case "$task_id" in
    task_L1_053)
      echo "$PROJECT_ROOT/important_results_eval/8.11/output/formal/baseline/l2/dialogue/hermesagent/task_L2_010/task_runs/01_task_L1_053/task_output/trace.jsonl"
      ;;
    task_L2_010)
      echo "$PROJECT_ROOT/important_results_eval/8.11/output/formal/baseline/l2/dialogue/hermesagent/task_L2_010/task_runs/03_task_L2_010/task_output/trace.jsonl"
      ;;
    *)
      echo ""
      ;;
  esac
}

cd "$EVAL_ROOT"

for task_id in $TASK_IDS; do
  task_input_dir="$EVAL_INPUT_ROOT/$task_id/input"
  out_dir="$OUT_ROOT/$task_id"
  trace_path="$(trace_for_task "$task_id")"

  export_args=(
    --graph "$task_input_dir/${task_id}_atomic_expanded_graph.json"
    --success-paths "$task_input_dir/${task_id}_success_paths.json"
    --atomic-transition-manifest "$task_input_dir/${task_id}_atomic_transition_manifest.json"
    --unique-states "$task_input_dir/${task_id}_unique_states.json"
    --full-branch-graph "$GRAPH_ROOT/${task_id}_full_branch_graph.json"
    --image-manifest "$EVAL_INPUT_ROOT/$task_id/task_image_manifest.json"
    --out-dir "$out_dir"
  )
  if [[ -n "$trace_path" && -f "$trace_path" ]]; then
    export_args+=(--trace "$trace_path")
  fi
  if [[ "$RENDER_SVG" == "1" ]]; then
    export_args+=(--render-svg)
  fi

  echo "Exporting visualization for $task_id"
  test -f "$task_input_dir/${task_id}_atomic_expanded_graph.json"
  PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.graph.visualize_task export "${export_args[@]}"
done

echo "Wrote visualizations to $OUT_ROOT"
