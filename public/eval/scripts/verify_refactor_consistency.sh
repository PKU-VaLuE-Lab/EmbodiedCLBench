#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

echo "[Eval] compileall"
"$PYTHON_BIN" -m compileall src scripts >/dev/null

echo "[Eval] import compatibility"
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import importlib

checks = [
    ("tongbench_eval.graph_env.loader", "tongbench_eval.graph_env.io.loader"),
    ("tongbench_eval.graph_env.env", "tongbench_eval.graph_env.runtime.env"),
    ("tongbench_eval.graph_env.runner", "tongbench_eval.graph_env.runtime.runner"),
    ("tongbench_eval.graph_env.workspace_env", "tongbench_eval.graph_env.workspace.env"),
    ("tongbench_eval.graph_env.framework_runtime", "tongbench_eval.graph_env.runtime.framework_runtime"),
    ("tongbench_eval.cli.framework.dialogue", "tongbench_eval.cli.framework.dialogue_runtime.main"),
]

for public_name, impl_name in checks:
    public_mod = importlib.import_module(public_name)
    impl_mod = importlib.import_module(impl_name)
    if public_mod is not impl_mod:
        raise SystemExit(f"{public_name} is not aliasing {impl_name}")
    print(f"OK {public_name} -> {impl_name}")
PY

echo "[Eval] CLI help"
PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.framework.single_task --help >/dev/null
PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue --help >/dev/null
PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks --help >/dev/null
PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.graph_env.cli --help >/dev/null

TASK_GRAPH="${TASK_GRAPH:-$PROJECT_ROOT/important_results_eval/8.10/input/image_reuse_l2_only/tasks/task_L2_002/input/task_L2_002_atomic_expanded_graph.json}"
UNIQUE_STATE_IMAGE_DIR="${UNIQUE_STATE_IMAGE_DIR:-$PROJECT_ROOT/output/036_02_evo_repro/eval_smoke/staged_images/l1/task_L1_001}"
if [[ ! -f "$TASK_GRAPH" ]]; then
  echo "[Eval] missing TASK_GRAPH, skip graph_env checks: $TASK_GRAPH" >&2
else
  echo "[Eval] graph_env validate/observe/rollout"
  PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.graph_env.cli validate --task-dir "$TASK_GRAPH" >/tmp/tongbench_eval_verify_validate.json || true
  PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.graph_env.cli observe --task-dir "$TASK_GRAPH" >/tmp/tongbench_eval_verify_observe.json
  PYTHONPATH=src "$PYTHON_BIN" -m tongbench_eval.graph_env.cli rollout --task-dir "$TASK_GRAPH" \
    --actions look_at_BP_Diswasher_005_C_1 look_at_BP_Diswasher_005_C_1 >/tmp/tongbench_eval_verify_rollout.json
fi

echo "[Eval] private workspace observe"
PYTHONPATH=src "$PYTHON_BIN" - <<'PY' "$PROJECT_ROOT" "$UNIQUE_STATE_IMAGE_DIR"
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

from tongbench_eval.graph_env.runtime.runner import prepare_tongsim_workspace
from tongbench_eval.utils.python_utils import current_python

project_root = Path(sys.argv[1])
unique_state_image_dir = Path(sys.argv[2])
l1_task_graph = os.environ.get("L1_TASK_GRAPH")
if l1_task_graph:
    task_graph = Path(l1_task_graph)
else:
    task_graph = project_root / "output/036_02_evo_repro/outputs_compositional_2/036_02/generated_from_imaged_l1/image_reuse_l2_only/graph/task_L1_001_atomic_expanded_graph.json"
    if not task_graph.exists():
        task_graph = project_root / "important_results_taskgen/8.10/output/generated_from_imaged_l1_036_02/image_reuse_l2_only/graph/task_L1_001_atomic_expanded_graph.json"
if not task_graph.exists():
    raise SystemExit(f"missing L1 task graph: {task_graph}")
if not unique_state_image_dir.exists():
    raise SystemExit(f"missing unique state image dir: {unique_state_image_dir}")
atomic_template_path = Path(
    os.environ.get(
        "ATOMIC_TEMPLATE_PATH",
        project_root / "important_results_taskgen/8.10/input/atomic_templates_updated_env_v3.json",
    )
)
subtask_template_path = Path(
    os.environ.get(
        "SUBTASK_TEMPLATE_PATH",
        project_root / "important_results_taskgen/8.10/input/subtask_templates_compressed_updated_env_v3.json",
    )
)
if not atomic_template_path.exists():
    raise SystemExit(f"missing atomic template path: {atomic_template_path}")
if not subtask_template_path.exists():
    raise SystemExit(f"missing subtask template path: {subtask_template_path}")

with tempfile.TemporaryDirectory(prefix="tongbench_eval_workspace_verify.") as tmp:
    workspace = Path(tmp)
    prepare_tongsim_workspace(task_graph, workspace, unique_state_image_dir=unique_state_image_dir)
    if not (workspace / "parts" / "part_01.pyfrag").exists():
        raise SystemExit("workspace parts were not copied")
    cmd = [
        current_python(),
        "tongsim_env.py",
        "observe",
        "--action-interface",
        "library_factorized",
        "--action-library-mode",
        "atomic_only",
        "--exclusive-in-view",
        "true",
        "--atomic-template-path",
        str(atomic_template_path),
        "--subtask-template-path",
        str(subtask_template_path),
    ]
    completed = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(completed.stderr)
    payload = json.loads(completed.stdout)
    action_count = len(payload.get("model_facing_action_candidates") or payload.get("candidate_actions") or [])
    image_count = len(payload.get("image_paths") or [])
    if image_count < 1 or action_count < 1:
        raise SystemExit(f"unexpected observe payload: images={image_count}, actions={action_count}")
    print(f"OK observe images={image_count} actions={action_count}")
PY

if [[ "${RUN_API_SMOKE:-0}" == "1" ]]; then
  if [[ -n "${API_SMOKE_OUTPUT_ROOT:-}" ]]; then
    smoke_out="$API_SMOKE_OUTPUT_ROOT"
    echo "[Eval] API smoke existing output: $smoke_out"
  elif [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "RUN_API_SMOKE=1 requires DASHSCOPE_API_KEY" >&2
    exit 1
  else
    smoke_out="$(mktemp -d /tmp/tongbench_eval_verify_api_smoke.XXXXXX)"
    echo "[Eval] API smoke output: $smoke_out"
    BACKENDS=hermesagent \
    LEVELS=L1 \
    TIMEOUT_SEC="${TIMEOUT_SEC:-180}" \
    MAX_ITERATIONS="${MAX_ITERATIONS:-12}" \
    L1_MAX_STEPS="${L1_MAX_STEPS:-3}" \
    OUTPUT_ROOT="$smoke_out" \
    bash scripts/run_036_02_level_smoke_all_frameworks.sh >/tmp/tongbench_eval_verify_api_smoke.log
  fi
  "$PYTHON_BIN" - <<'PY' "$smoke_out"
from pathlib import Path
import json
import sys

smoke_out = Path(sys.argv[1])
task_out = smoke_out / "hermesagent/L1/task_L1_001"
score_path = task_out / "score.json"
usage_path = task_out / "usage.json"
trace_path = task_out / "task_output/trace.jsonl"
bridge_stdout_path = task_out / "bridge_stdout.log"
bridge_stderr_path = task_out / "bridge_stderr.log"
if not score_path.exists():
    raise SystemExit(f"missing score: {score_path}")
if not usage_path.exists():
    raise SystemExit(f"missing usage: {usage_path}")
if not trace_path.exists():
    raise SystemExit(f"missing trace: {trace_path}")
score = json.loads(score_path.read_text(encoding="utf-8"))
usage = json.loads(usage_path.read_text(encoding="utf-8"))
if int(usage.get("request_count", 0)) < 1:
    raise SystemExit(f"API smoke made no model requests: {usage}")
trace_entries = [
    json.loads(line)
    for line in trace_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not trace_entries:
    raise SystemExit("API smoke produced no trace entries")
valid_count = sum(1 for entry in trace_entries if entry.get("valid"))
if valid_count < 1:
    raise SystemExit(f"API smoke produced no valid environment actions: {trace_entries}")
for path in (bridge_stdout_path, bridge_stderr_path):
    if path.exists() and path.stat().st_size:
        raise SystemExit(f"bridge emitted unexpected output: {path}")
print(
    "OK API smoke "
    f"score={score.get('overall_score')} "
    f"reached_goal={score.get('details', {}).get('reached_goal')} "
    f"requests={usage.get('request_count')} "
    f"trace_len={len(trace_entries)} "
    f"valid_actions={valid_count}"
)
PY
fi

echo "[Eval] verification passed"
