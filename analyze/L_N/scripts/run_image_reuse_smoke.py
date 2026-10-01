#!/usr/bin/env python3
"""Materialize reusable child images and ready manifests after graph search."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    taskgen = bundle / "input/taskgen_output"
    source_reuse = bundle / "input/l1_sources"
    output_dir = bundle / "input/image_reuse"
    graph_dir = taskgen / "graph"
    initial_image = external / "important_results/For_yanan/final_scene_results_two_heights_320x180/child/scenes/001/task_L1_066/output/task_L1_066_unique_states/000_state_0000/state_0000_selected_orbit_v2_r_0p67_a_0_rgb.jpg"
    # The packaging tool's reuse mode expects the existing task manifest in
    # its output directory.  Seed that manifest explicitly; otherwise it
    # would interpret this as a request to regenerate L2/L3 tasks.
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(taskgen / "tasks_compositional.json", output_dir / "tasks_compositional.json")
    cmd = [
        sys.executable,
        "-m",
        "tongbench_taskgen.image_reuse.packaging.generate_composites",
        "--source-reuse-dir", str(source_reuse),
        "--task-root", str(taskgen),
        "--tasks-json", str(taskgen / "tasks_compositional.json"),
        "--initial-image", str(initial_image),
        "--output-dir", str(output_dir),
        "--image-output-dir", str(output_dir),
        "--graph-dir", str(graph_dir),
        "--l1-graph-dir", str(taskgen / "graph"),
        "--reuse-existing-generated-tasks",
        "--allow-existing-output",
        "--copy-images",
        "--copy-task-files",
        "--levels", "L3,L5,L6",
    ]
    log = bundle / "output/image_reuse.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(repo / "private/taskgen/src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=external, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Image reuse failed; see {log}")
    summary_path = output_dir / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    # The shared packager keeps the historical filename
    # `complete_l2_l3_tasks_with_images.json` even when `--levels` contains
    # L5/L6.  Expose the generic manifest name used by the API pipeline and
    # materialize its per-task manifests under eval_input_dir/tasks/.
    complete = output_dir / "complete_l2_l3_tasks_with_images.json"
    if not complete.exists():
        raise FileNotFoundError(f"Image reuse did not produce task manifests: {complete}")
    records = json.loads(complete.read_text(encoding="utf-8"))
    selected = []
    for record in records if isinstance(records, list) else []:
        level = str(record.get("difficulty_level") or "").upper()
        if level not in {"L3", "L5", "L6"}:
            continue
        task_id = str(record.get("task_id") or "")
        if not task_id:
            continue
        task_dir = output_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task_image_manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        selected.append(record)
    # L4 is the already-generated parent task.  Reuse its verified manifest
    # from the immutable N-sample input instead of regenerating or modifying
    # that task here.
    parent_ready = repo / "analyze/N-sample/input/subset_10pct_child/ready_l4_tasks_with_images.json"
    if parent_ready.is_file():
        parent_rows = json.loads(parent_ready.read_text(encoding="utf-8"))
        for record in parent_rows if isinstance(parent_rows, list) else []:
            if str(record.get("task_id") or "") != "task_L4_HL_v5_child_0002":
                continue
            record = deepcopy(record)
            for state in record.get("states", []) if isinstance(record, dict) else []:
                if (
                    isinstance(state, dict)
                    and str(state.get("match_status") or "") == "packaged"
                    and (state.get("source_image_paths") or state.get("copied_image_path") or state.get("reused_image_path"))
                ):
                    # In this local copy, `packaged` means the state already
                    # has a concrete image; normalize it to the image-safe
                    # export vocabulary without touching the source manifest.
                    state["match_status"] = "reused"
            task_dir = output_dir / "tasks" / record["task_id"]
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task_image_manifest.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            selected.append(record)
            break
    ready = output_dir / "ready_highlevel_v2_tasks_with_images.json"
    ready.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "ready_json": str(ready), "task_manifest_count": len(selected), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
