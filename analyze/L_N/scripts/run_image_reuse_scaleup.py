#!/usr/bin/env python3
"""Package scale-up graph states with reusable child-view images."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def state_key(values) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in values or []))


def shortest_path_state_ids(taskgen_output: Path, task_id: str) -> list[str]:
    graph = read_json(taskgen_output / "graph" / f"{task_id}_atomic_expanded_graph.json")
    success = read_json(taskgen_output / "graph" / f"{task_id}_success_paths.json")
    path = next(
        (item for item in success.get("success_paths", []) if item.get("is_shortest_success_path")),
        None,
    )
    if path is None:
        paths = [item for item in success.get("success_paths", []) if isinstance(item, dict)]
        path = min(paths, key=lambda item: int(item.get("atomic_len", 10**9) or 10**9), default=None)
    if path is None:
        raise ValueError(f"No successful path for {task_id}")
    by_state = {state_key(node.get("state")): str(node.get("unique_state_id") or node.get("id")) for node in graph.get("nodes", [])}
    result: list[str] = []
    for step in path.get("atomic_steps", []) or []:
        state_id = by_state.get(state_key(step.get("before_state")))
        if state_id and state_id not in result:
            result.append(state_id)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--task-name", default="LN_scaleup_v1")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    taskgen = bundle / "input/taskgen_output"
    source_reuse = bundle / "input/l1_sources"
    output_dir = bundle / "input/image_reuse"
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(taskgen / "tasks_compositional.json", output_dir / "tasks_compositional.json")

    initial_images = sorted(
        (external / "important_results/For_yanan/final_scene_results_two_heights_320x180/child/scenes").glob(
            "*/task_L1_066/output/task_L1_066_unique_states/000_state_0000/*.jpg"
        )
    )
    if not initial_images:
        initial_images = sorted(
            (external / "important_results/For_yanan/final_scene_results_two_heights_320x180/child/scenes").glob(
                "*/task_L1_*/output/*_unique_states/000_state_0000/*.jpg"
            )
        )
    if not initial_images:
        raise FileNotFoundError("No child state_0000 image found.")
    sdk = external / "tongsim-python-sdk/TongBench"
    cmd = [
        sys.executable,
        "-m",
        "tongbench_taskgen.image_reuse.packaging.generate_composites",
        "--source-reuse-dir", str(source_reuse),
        "--task-root", str(taskgen),
        "--tasks-json", str(taskgen / "tasks_compositional.json"),
        "--initial-image", str(initial_images[0]),
        "--output-dir", str(output_dir),
        "--image-output-dir", str(output_dir),
        "--graph-dir", str(taskgen / "graph"),
        "--l1-graph-dir", str(taskgen / "graph"),
        "--reuse-existing-generated-tasks",
        "--allow-existing-output",
        "--copy-images",
        "--copy-task-files",
        "--use-full-source-l1-pool-for-reuse",
        "--levels", "L3,L5,L6",
    ]
    log = bundle / "output/image_reuse_scaleup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "private/taskgen/src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(cmd, cwd=external, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"image reuse failed; see {log}")

    complete_path = output_dir / "complete_l2_l3_tasks_with_images.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"Missing image reuse output: {complete_path}")
    records = read_json(complete_path)
    target_prefixes = tuple(f"task_{level}_{args.task_name}_" for level in ("L3", "L5", "L6"))
    target_records = [
        row for row in records if isinstance(row, dict)
        and str(row.get("task_id") or "").startswith(target_prefixes)
    ]
    ready_records = []
    fully_ready_records = []
    missing_records = []
    readiness_audit = []
    for row in target_records:
        task_id = str(row.get("task_id") or "")
        task_dir = output_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        manifest_by_state = {
            str(item.get("target_state_id")): item
            for item in row.get("states", [])
            if isinstance(item, dict)
        }
        path_state_ids = shortest_path_state_ids(taskgen, task_id)
        path_missing = [
            state_id
            for state_id in path_state_ids
            if str(manifest_by_state.get(state_id, {}).get("match_status") or "") != "reused"
        ]
        all_missing = [
            state_id
            for state_id, item in manifest_by_state.items()
            if str(item.get("match_status") or "") != "reused"
        ]
        side_branch_missing = [state_id for state_id in all_missing if state_id not in path_state_ids]
        audited = dict(row)
        audited.update(
            {
                "shortest_path_state_ids": path_state_ids,
                "shortest_path_state_count": len(path_state_ids),
                "shortest_path_missing_state_count": len(path_missing),
                "side_branch_missing_state_count": len(side_branch_missing),
                "downstream_ready": not path_missing,
                "fully_image_ready": not all_missing,
            }
        )
        write_json(task_dir / "task_image_manifest.json", audited)
        if not path_missing:
            ready_records.append(audited)
        if not all_missing:
            fully_ready_records.append(audited)
        if path_missing or side_branch_missing:
            missing_records.append(audited)
        readiness_audit.append(
            {
                "task_id": task_id,
                "shortest_path_state_count": len(path_state_ids),
                "shortest_path_missing_state_count": len(path_missing),
                "side_branch_missing_state_count": len(side_branch_missing),
                "downstream_ready": not path_missing,
                "fully_image_ready": not all_missing,
            }
        )
    ready_path = output_dir / "ready_l3_l5_l6_tasks_with_images.json"
    write_json(ready_path, ready_records)
    fully_ready_path = output_dir / "fully_ready_l3_l5_l6_tasks_with_images.json"
    write_json(fully_ready_path, fully_ready_records)
    write_json(output_dir / "all_l3_l5_l6_tasks_with_images.json", target_records)
    write_json(output_dir / "image_reuse_missing_tasks.json", missing_records)
    audit_path = output_dir / "image_reuse_readiness_audit.json"
    write_json(audit_path, readiness_audit)
    write_json(bundle / "output/image_reuse_summary.json", {
        "ok": len(ready_records) == len(target_records),
        "target_count": len(target_records),
        "downstream_ready_count": len(ready_records),
        "fully_image_ready_count": len(fully_ready_records),
        "missing_image_task_count": sum(1 for row in readiness_audit if not row["downstream_ready"]),
        "side_branch_missing_task_count": sum(1 for row in readiness_audit if row["side_branch_missing_state_count"]),
        "ready_json": str(ready_path),
        "fully_ready_json": str(fully_ready_path),
        "readiness_audit": str(audit_path),
        "missing_tasks": [
            row for row in readiness_audit if not row["downstream_ready"]
        ],
    })
    print(json.dumps({
        "ok": len(ready_records) == len(target_records),
        "target_count": len(target_records),
        "downstream_ready_count": len(ready_records),
        "fully_image_ready_count": len(fully_ready_records),
        "missing_image_task_count": sum(1 for row in readiness_audit if not row["downstream_ready"]),
        "side_branch_missing_task_count": sum(1 for row in readiness_audit if row["side_branch_missing_state_count"]),
        "ready_json": str(ready_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
