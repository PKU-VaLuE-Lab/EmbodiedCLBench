#!/usr/bin/env python3
"""Prepare an evaluation-only view of the L_N scale-up bundle."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path


LEVELS = ("L3", "L5", "L6")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    source_image_root = bundle / "input/image_reuse"
    source_api_root = bundle / "input/api_pipeline"
    source_ready = source_image_root / "ready_l3_l5_l6_tasks_with_images.json"
    repo_root = Path(__file__).resolve().parents[3]
    target_root = bundle / "input/eval_ready_scaleup"
    target_tasks = target_root / "tasks"
    target_ready = target_root / "ready_l3_l5_l6_api_descriptions.json"

    if not source_ready.is_file():
        raise FileNotFoundError(source_ready)
    if target_root.exists() and not args.force:
        summary_path = target_root / "prepare_summary.json"
        if summary_path.is_file():
            print(summary_path.read_text(encoding="utf-8"), end="")
            return 0
        raise FileExistsError(f"Evaluation root exists without summary: {target_root}")
    if target_root.exists():
        shutil.rmtree(target_root)
    target_tasks.mkdir(parents=True, exist_ok=True)

    ready_items = read_json(source_ready)
    if not isinstance(ready_items, list):
        raise ValueError(f"Expected a list in {source_ready}")
    prepared_items = []
    records = []
    for original in ready_items:
        if not isinstance(original, dict):
            continue
        task_id = str(original.get("task_id") or "").strip()
        if not task_id:
            continue
        level = str(original.get("difficulty_level") or "").strip().upper()
        if level not in LEVELS:
            level = str((original.get("task") or {}).get("difficulty_level") or "").strip().upper()
        if level not in LEVELS:
            raise ValueError(f"Cannot infer level for {task_id}")

        source_task_root = source_image_root / "tasks" / task_id
        source_input = source_task_root / "input"
        source_graph = source_input / f"{task_id}_atomic_expanded_graph.json"
        source_stage1 = source_api_root / level / task_id / "stage1_task_output.json"
        source_image_manifest = source_task_root / "task_image_manifest.json"
        required_paths = (source_graph, source_image_manifest, source_stage1)
        for path in required_paths:
            if not path.is_file():
                raise FileNotFoundError(f"{task_id}: {path}")

        stage1 = read_json(source_stage1)
        description = str(
            stage1.get("description_highlevel_hard")
            or stage1.get("description")
            or ""
        ).strip()
        if not description:
            raise ValueError(f"{task_id}: stage1 description is empty")

        target_task_root = target_tasks / task_id
        target_input = target_task_root / "input"
        target_input.mkdir(parents=True, exist_ok=True)
        graph = read_json(source_graph)
        graph["description_highlevel_hard"] = description
        graph["api_description_source"] = str(source_stage1.resolve())
        write_json(target_input / source_graph.name, graph)
        for sidecar in source_input.glob(f"{task_id}_*.json"):
            if sidecar.name == source_graph.name:
                continue
            shutil.copy2(sidecar, target_input / sidecar.name)
        shutil.copy2(source_image_manifest, target_task_root / source_image_manifest.name)

        item = copy.deepcopy(original)
        task_payload = item.get("task") if isinstance(item.get("task"), dict) else {}
        task_payload["description_highlevel_hard"] = description
        item["task"] = task_payload
        item["api_description_source"] = str(source_stage1.resolve())
        prepared_items.append(item)
        records.append(
            {
                "task_id": task_id,
                "level": level,
                "graph_path": str((target_input / source_graph.name).resolve()),
                "description_source": str(source_stage1.resolve()),
                "description_highlevel_hard": description,
                "image_manifest": str((target_task_root / source_image_manifest.name).resolve()),
            }
        )

    write_json(target_ready, prepared_items)
    summary = {
        "ok": True,
        "schema_version": "tongbench_l_n_eval_ready_v1",
        "source_ready": str(source_ready.resolve()),
        "source_api_root": str(source_api_root.resolve()),
        "target_root": str(target_root.resolve()),
        "ready_json": str(target_ready.resolve()),
        "task_count": len(records),
        "levels": {level: sum(record["level"] == level for record in records) for level in LEVELS},
        "records": records,
        "images_are_referenced_from_existing_image_reuse_bundle": True,
    }
    write_json(target_root / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
