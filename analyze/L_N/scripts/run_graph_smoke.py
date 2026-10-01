#!/usr/bin/env python3
"""Run the existing offline graph search for the L_N smoke tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


LEVEL_SETTINGS = {
    # Smoke deliberately keeps the branch fan-out small.  The target values
    # are approximate (45/75/90); the accepted caps include a modest margin
    # because the searcher's atomic expansion is not exactly linear in L1
    # count.  The formal run will use the larger, separately documented
    # search budget.
    "L3": {"max_depth": 10, "max_outgoing": 1, "max_nodes": 50, "target_states": 45, "max_states": 70},
    "L5": {"max_depth": 16, "max_outgoing": 1, "max_nodes": 50, "target_states": 75, "max_states": 90},
    "L6": {"max_depth": 18, "max_outgoing": 1, "max_nodes": 50, "target_states": 90, "max_states": 110},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    taskgen = bundle / "input/taskgen_output"
    tasks_file = taskgen / "tasks_compositional.json"
    graph_dir = taskgen / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    logs = bundle / "output/graph_logs"
    logs.mkdir(parents=True, exist_ok=True)
    sdk = external / "tongsim-python-sdk/TongBench"
    base_dir = external / "important_results/For_yanan/env_data/6.25_delivery/scenes/001"
    rules = sdk / "prompts/planner_rules.yaml"
    atomic = sdk / "outputs/atomic_templates_updated_env_v3.json"
    subtasks = sdk / "outputs/subtask_templates_compressed_updated_env_v3.json"
    tasks = [item for item in read_json(tasks_file) if str(item.get("difficulty_level") or "") in LEVEL_SETTINGS]
    if not tasks:
        raise ValueError(f"No L3/L5/L6 tasks in {tasks_file}")
    results = []
    for item in tasks:
        task_id = str(item["task_id"])
        level = str(item["difficulty_level"])
        setting = LEVEL_SETTINGS[level]
        unique_path = graph_dir / f"{task_id}_unique_states.json"
        if unique_path.exists() and not args.force:
            status = "reused"
        else:
            log_path = logs / f"{task_id}.log"
            cmd = [
                os.environ.get("PYTHON", "python3"),
                "-u",
                str(sdk / "dag_task_search_success_only_autopath_supported_detailed.py"),
                "--base-dir", str(base_dir),
                "--rules", str(rules),
                "--atomic-file", str(atomic),
                "--subtasks", str(subtasks),
                "--tasks-file", str(tasks_file),
                "--task-id", task_id,
                "--output-dir", str(graph_dir),
                "--search-mode", "full_branch",
                "--action-level", "subtask",
                "--max-depth", str(setting["max_depth"]),
                "--max-nodes", str(setting["max_nodes"]),
                "--max-outgoing-per-node", str(setting["max_outgoing"]),
                "--skip-full-branch-graph",
            ]
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    cmd,
                    cwd=sdk,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_sec,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"Graph search failed for {task_id}; see {log_path}")
            status = "searched"
        payload = read_json(unique_path) if unique_path.exists() else {}
        states = payload.get("states", []) if isinstance(payload, dict) else []
        state_count = len(states) if isinstance(states, list) else 0
        cap = int(setting["max_states"])
        results.append({"task_id": task_id, "level": level, "status": status, "unique_state_count": state_count, "target_states_approx": int(setting["target_states"]), "max_atomic_states": cap, "within_cap": state_count <= cap, "unique_states_path": str(unique_path)})
        if state_count == 0 or state_count > cap:
            raise RuntimeError(f"{task_id} produced {state_count} unique states; expected 1..{cap}")
    (bundle / "output/graph_smoke_summary.json").write_text(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
