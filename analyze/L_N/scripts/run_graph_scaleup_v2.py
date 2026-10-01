#!/usr/bin/env python3
"""Orchestrate the independent V2 compact graph build for L5/L6.

The existing ``run_graph_scaleup.py`` is the V1 entry point and is not
modified by this script. Each V2 task is written to a separate graph
directory and carries a selection manifest that records the V1 reference,
state budget, selected detour paths, and validation result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SETTINGS = {
    "L5": {"max_depth": 8, "max_nodes": 60, "max_outgoing": 2},
    "L6": {"max_depth": 10, "max_nodes": 80, "max_outgoing": 2},
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_level(row: dict) -> str:
    return str(row.get("difficulty_level") or row.get("level") or "").upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--taskgen-output-dir", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--v1-graph-dir", type=Path, required=True)
    parser.add_argument("--image-manifest-root", type=Path, default=None)
    parser.add_argument("--output-graph-dir", type=Path, required=True)
    parser.add_argument("--levels", default="L5,L6")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--max-detour-paths", type=int, default=24)
    parser.add_argument("--state-budget-divisor", type=int, default=3)
    parser.add_argument("--max-state-cap", type=int, default=100)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    taskgen = (args.taskgen_output_dir or (bundle / "input/taskgen_output")).resolve()
    output_graph_dir = args.output_graph_dir.resolve()
    logs = (args.log_dir or (bundle / "output/graph_v2_logs")).resolve()
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    tasks = read_json(taskgen / "tasks_compositional.json")
    task_rows = tasks if isinstance(tasks, list) else tasks.get("tasks", [])
    requested = {value.strip().upper() for value in args.levels.split(",") if value.strip()}
    only = set(args.task_id)
    work = [
        row for row in task_rows
        if isinstance(row, dict)
        and task_level(row) in requested
        and (not only or str(row.get("task_id") or "") in only)
    ]
    if not work:
        raise ValueError(f"No tasks for levels {sorted(requested)}")

    sdk = external / "tongsim-python-sdk/TongBench"
    v2_script = repo / "analyze/L_N/scripts/run_guided_graph_search_v2.py"
    rules = sdk / "prompts/planner_rules.yaml"
    atomic = sdk / "outputs/atomic_templates_updated_env_v3.json"
    subtasks = sdk / "outputs/subtask_templates_compressed_updated_env_v3.json"
    python = os.environ.get("PYTHON", "python3")

    def run(row: dict) -> dict:
        task_id = str(row["task_id"])
        level = task_level(row)
        setting = SETTINGS[level]
        task_output_dir = output_graph_dir
        manifest = task_output_dir / f"{task_id}_v2_selection_manifest.json"
        if manifest.is_file() and not args.force:
            payload = read_json(manifest)
            validation = payload.get("validation", {})
            return {
                "task_id": task_id,
                "level": level,
                "scene_id": row.get("scene_id"),
                "status": "reused",
                "v1_unique_state_count": payload.get("v1_unique_state_count"),
                "state_budget": payload.get("state_budget"),
                "v2_unique_state_count": validation.get("node_count"),
                "v2_edge_count": validation.get("edge_count"),
                "detour_path_count": validation.get("selected_recoverable_detour_path_count"),
                "validation_ok": validation.get("ok"),
                "selection_manifest": str(manifest),
            }
        scene = str(row.get("scene_id") or "")
        base_dir = external / "important_results/For_yanan/env_data/6.25_delivery/scenes" / scene
        log = logs / f"{task_id}.log"
        cmd = [
            python, "-u", str(v2_script),
            "--sdk-root", str(sdk),
            "--base-dir", str(base_dir),
            "--rules", str(rules),
            "--atomic-file", str(atomic),
            "--subtasks", str(subtasks),
            "--tasks-file", str(taskgen / "tasks_compositional.json"),
            "--task-id", task_id,
            "--output-dir", str(task_output_dir),
            "--v1-graph-dir", str(args.v1_graph_dir.resolve()),
            "--max-depth", str(setting["max_depth"]),
            "--max-nodes", str(setting["max_nodes"]),
            "--max-outgoing-per-node", str(setting["max_outgoing"]),
            "--max-success-traces", "500",
            "--max-detour-paths", str(args.max_detour_paths),
            "--state-budget-divisor", str(args.state_budget_divisor),
            "--max-state-cap", str(args.max_state_cap),
        ]
        if args.image_manifest_root is not None:
            cmd.extend(["--image-manifest-root", str(args.image_manifest_root.resolve())])
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=sdk,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_sec,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"V2 graph search failed for {task_id}; see {log}")
        payload = read_json(manifest)
        validation = payload.get("validation", {})
        return {
            "task_id": task_id,
            "level": level,
            "scene_id": row.get("scene_id"),
            "status": "searched",
            "v1_unique_state_count": payload.get("v1_unique_state_count"),
            "state_budget": payload.get("state_budget"),
            "v2_unique_state_count": validation.get("node_count"),
            "v2_edge_count": validation.get("edge_count"),
            "detour_path_count": validation.get("selected_recoverable_detour_path_count"),
            "validation_ok": validation.get("ok"),
            "selection_manifest": str(manifest),
        }

    results: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = {executor.submit(run, row): row for row in work}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"task_id": row.get("task_id"), "error": str(exc)})
    results.sort(key=lambda item: str(item.get("task_id")))
    failures.sort(key=lambda item: str(item.get("task_id")))

    def aggregate(rows: list[dict]) -> dict:
        if not rows:
            return {"task_count": 0}
        v1_counts = [int(item.get("v1_unique_state_count") or 0) for item in rows]
        v2_counts = [int(item.get("v2_unique_state_count") or 0) for item in rows]
        return {
            "task_count": len(rows),
            "v1_state_min": min(v1_counts),
            "v1_state_max": max(v1_counts),
            "v1_state_avg": sum(v1_counts) / len(v1_counts),
            "v2_state_min": min(v2_counts),
            "v2_state_max": max(v2_counts),
            "v2_state_avg": sum(v2_counts) / len(v2_counts),
            "v2_to_v1_state_ratio": sum(v2_counts) / max(1, sum(v1_counts)),
            "state_reduction_fraction": 1.0 - (sum(v2_counts) / max(1, sum(v1_counts))),
            "detour_path_min": min(int(item.get("detour_path_count") or 0) for item in rows),
            "detour_path_max": max(int(item.get("detour_path_count") or 0) for item in rows),
        }

    summary = {
        "ok": not failures and all(item.get("validation_ok") is True for item in results),
        "version": "v2",
        "levels": sorted(requested),
        "submitted_count": len(work),
        "result_count": len(results),
        "failure_count": len(failures),
        "state_budget_divisor": int(args.state_budget_divisor),
        "max_detour_paths": int(args.max_detour_paths),
        "max_state_cap": int(args.max_state_cap),
        "v1_graph_dir": str(args.v1_graph_dir.resolve()),
        "image_manifest_root": str(args.image_manifest_root.resolve()) if args.image_manifest_root else None,
        "output_graph_dir": str(output_graph_dir),
        "statistics": {
            "overall": aggregate(results),
            "by_level": {
                level: aggregate([item for item in results if item.get("level") == level])
                for level in sorted(requested)
            },
        },
        "results": results,
        "failures": failures,
    }
    write_json(bundle / "output/graph_scaleup_v2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
