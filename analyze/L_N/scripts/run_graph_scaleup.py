#!/usr/bin/env python3
"""Run graph search for L3, then L5, then L6 scale-up tasks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SETTINGS = {
    # max_nodes is the planner's expanded-node budget, not the final number
    # of atomic image states. L5/L6 use the task-local guided wrapper, which
    # keeps one success continuation and one additional branch per state.
    "L3": {"max_depth": 10, "max_nodes": 120, "max_outgoing": 3, "target": 45, "cap": 90},
    "L5": {"max_depth": 8, "max_nodes": 60, "max_outgoing": 2, "target": 75, "cap": 500},
    "L6": {"max_depth": 10, "max_nodes": 80, "max_outgoing": 2, "target": 90, "cap": 700},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--levels", default="L3")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    taskgen = bundle / "input/taskgen_output"
    graph_dir = taskgen / "graph"
    logs = bundle / "output/graph_logs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tasks = read_json(taskgen / "tasks_compositional.json")
    requested = {x.strip().upper() for x in args.levels.split(",") if x.strip()}
    only = set(args.task_id)
    work = [
        row for row in tasks if isinstance(row, dict)
        and str(row.get("difficulty_level") or "").upper() in requested
        and (not only or str(row.get("task_id") or "") in only)
    ]
    if not work:
        raise ValueError(f"No tasks for levels {sorted(requested)}")
    sdk = external / "tongsim-python-sdk/TongBench"
    guided_script = repo / "analyze/L_N/scripts/run_guided_graph_search.py"
    rules = sdk / "prompts/planner_rules.yaml"
    atomic = sdk / "outputs/atomic_templates_updated_env_v3.json"
    subtasks = sdk / "outputs/subtask_templates_compressed_updated_env_v3.json"
    python = os.environ.get("PYTHON", "python3")

    def run(row: dict) -> dict:
        task_id = str(row["task_id"])
        level = str(row["difficulty_level"]).upper()
        setting = SETTINGS[level]
        unique = graph_dir / f"{task_id}_unique_states.json"
        if unique.exists() and not args.force:
            status = "reused"
        else:
            scene = str(row.get("scene_id") or "")
            base_dir = external / "important_results/For_yanan/env_data/6.25_delivery/scenes" / scene
            log = logs / f"{task_id}.log"
            cmd = [
                python, "-u", str(guided_script),
                "--sdk-root", str(sdk),
                "--base-dir", str(base_dir),
                "--rules", str(rules),
                "--atomic-file", str(atomic),
                "--subtasks", str(subtasks),
                "--tasks-file", str(taskgen / "tasks_compositional.json"),
                "--task-id", task_id,
                "--output-dir", str(graph_dir),
                "--max-depth", str(setting["max_depth"]),
                "--max-nodes", str(setting["max_nodes"]),
                "--max-outgoing-per-node", str(setting["max_outgoing"]),
                "--max-success-traces", "100",
                "--skip-full-branch-graph",
            ]
            with log.open("w", encoding="utf-8") as handle:
                result = subprocess.run(cmd, cwd=sdk, stdout=handle, stderr=subprocess.STDOUT, timeout=args.timeout_sec, check=False)
            if result.returncode:
                raise RuntimeError(f"graph search failed for {task_id}; see {log}")
            status = "searched"
        payload = read_json(unique) if unique.exists() else {}
        states = payload.get("states", []) if isinstance(payload, dict) else []
        count = len(states) if isinstance(states, list) else 0
        return {
            "task_id": task_id, "level": level, "scene_id": row.get("scene_id"),
            "status": status, "unique_state_count": count,
            "target_states_approx": setting["target"], "max_states": setting["cap"],
            "within_cap": 0 < count <= setting["cap"], "unique_states_path": str(unique),
        }

    results = []
    failures = []
    workers = max(1, int(args.max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, row): row for row in work}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"task_id": row.get("task_id"), "error": str(exc)})
    results.sort(key=lambda x: str(x["task_id"]))
    summary_path = bundle / "output/graph_scaleup_summary.json"
    previous = read_json(summary_path) if summary_path.is_file() else {}
    previous_results = {
        str(item.get("task_id")): item
        for item in previous.get("results", [])
        if isinstance(item, dict) and item.get("task_id")
    }
    previous_results.update({str(item["task_id"]): item for item in results})
    previous_failures = {
        str(item.get("task_id")): item
        for item in previous.get("failures", [])
        if isinstance(item, dict) and item.get("task_id")
    }
    for item in failures:
        previous_failures[str(item.get("task_id"))] = item
    completed_ids = set(previous_results)
    previous_failures = {key: value for key, value in previous_failures.items() if key not in completed_ids}
    summary = {
        "ok": not previous_failures and all(int(item.get("unique_state_count", 0) or 0) > 0 for item in previous_results.values()),
        "levels": sorted({str(item).upper() for item in (previous.get("levels", []) or [])} | requested),
        "submitted_count": len(work),
        "cumulative_task_count": len(previous_results),
        "results": sorted(previous_results.values(), key=lambda x: str(x["task_id"])),
        "failures": sorted(previous_failures.values(), key=lambda x: str(x.get("task_id") or "")),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
