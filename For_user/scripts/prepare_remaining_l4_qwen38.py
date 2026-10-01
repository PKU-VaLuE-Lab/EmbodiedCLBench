#!/usr/bin/env python3
"""Prepare non-overlapping Qwen3.8 L4 continuation manifests.

This script derives JSON manifests only. It never calls an API or reruns L2.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil


HARNESSES = ("hermesagent", "codex", "claudecode", "openclaw")
WAVE_SIZES = (20, 20, 20, 16)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def task_id(item: dict) -> str:
    value = str(item.get("task_id") or "").strip()
    if not value:
        raise ValueError("Task entry has no task_id")
    return value


def height_of(item: dict) -> str:
    value = item.get("height") or item.get("camera_height") or item.get("viewpoint")
    if value:
        return str(value)
    identifier = task_id(item)
    if "_adult_" in identifier:
        return "adult"
    if "_child_" in identifier:
        return "child"
    return "unknown"


def scene_of(item: dict) -> str:
    for key in ("scene_id", "scene", "room_id", "room"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    identifier = task_id(item)
    match = re.search(r"(?:scene|room)[_-]?([0-9]+(?:_[0-9]+)?)", identifier, re.I)
    return match.group(1) if match else "unknown"


def replace_generated_root(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Generated subset already exists: {path}; pass --force to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_subset(
    root: Path,
    *,
    name: str,
    full_l2: list[dict],
    l4_items: list[dict],
    plan: dict,
    targets: list[dict],
    source_name: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "ready_l2_tasks_with_images.json", full_l2)
    write_json(root / "ready_l4_tasks_with_images.json", l4_items)
    plan_copy = dict(plan)
    plan_copy["targets"] = targets
    plan_copy["target_count"] = len(targets)
    write_json(root / "learning_plan.json", plan_copy)
    write_json(
        root / "subset_summary.json",
        {
            "subset": name,
            "source_subset": source_name,
            "l2_task_count": len(full_l2),
            "l4_task_count": len(l4_items),
            "learning_target_count": len(targets),
            "height_counts_l2": dict(Counter(height_of(item) for item in full_l2)),
            "height_counts_l4": dict(Counter(height_of(item) for item in l4_items)),
            "scene_counts_l2": dict(Counter(scene_of(item) for item in full_l2)),
            "scene_counts_l4": dict(Counter(scene_of(item) for item in l4_items)),
            "l2_task_ids": [task_id(item) for item in full_l2],
            "l4_task_ids": [task_id(item) for item in l4_items],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-subset-root", type=Path, required=True)
    parser.add_argument("--completed-subset-root", type=Path, required=True)
    parser.add_argument("--subsets-root", type=Path, required=True)
    parser.add_argument("--l2-results-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    full_root = args.full_subset_root.resolve()
    completed_root = args.completed_subset_root.resolve()
    subsets_root = args.subsets_root.resolve()
    l2_results_root = args.l2_results_root.resolve()
    output_root = args.output_root.resolve()

    full_l2 = read_json(full_root / "ready_l2_tasks_with_images.json")
    full_l4 = read_json(full_root / "ready_l4_tasks_with_images.json")
    completed_l4 = read_json(completed_root / "ready_l4_tasks_with_images.json")
    plan = read_json(full_root / "learning_plan.json")
    if not isinstance(full_l2, list) or not isinstance(full_l4, list) or not isinstance(completed_l4, list):
        raise ValueError("Ready task files must contain lists")
    if not isinstance(plan, dict) or not isinstance(plan.get("targets"), list):
        raise ValueError("Full learning plan must contain a targets list")

    completed_ids = {task_id(item) for item in completed_l4}
    full_l4_ids = [task_id(item) for item in full_l4]
    if len(completed_ids) != len(completed_l4):
        raise ValueError("Completed L4 subset contains duplicate task IDs")
    if not completed_ids.issubset(set(full_l4_ids)):
        unknown = sorted(completed_ids - set(full_l4_ids))
        raise ValueError(f"Completed subset contains IDs outside full subset: {unknown}")

    remaining_l4 = [item for item in full_l4 if task_id(item) not in completed_ids]
    if len(remaining_l4) != sum(WAVE_SIZES):
        raise ValueError(
            f"Expected {sum(WAVE_SIZES)} remaining L4 tasks, found {len(remaining_l4)}"
        )

    targets_by_id = {
        str(item.get("target_task_id") or item.get("task_id") or "").strip(): item
        for item in plan["targets"]
        if isinstance(item, dict)
    }
    remaining_targets = []
    for item in remaining_l4:
        identifier = task_id(item)
        if identifier not in targets_by_id:
            raise KeyError(f"Learning plan has no target entry for {identifier}")
        remaining_targets.append(targets_by_id[identifier])

    generated_name = "stage3_remaining_80pct_qwen38"
    remaining_root = subsets_root / generated_name
    replace_generated_root(remaining_root, args.force)
    write_subset(
        remaining_root,
        name=generated_name,
        full_l2=full_l2,
        l4_items=remaining_l4,
        plan=plan,
        targets=remaining_targets,
        source_name="stage3_full",
    )

    wave_roots: list[Path] = []
    offset = 0
    for index, size in enumerate(WAVE_SIZES, start=1):
        wave_name = f"stage3_remaining_80pct_qwen38_wave{index:02d}"
        wave_root = subsets_root / wave_name
        replace_generated_root(wave_root, args.force)
        wave_items = remaining_l4[offset : offset + size]
        wave_targets = remaining_targets[offset : offset + size]
        write_subset(
            wave_root,
            name=wave_name,
            full_l2=full_l2,
            l4_items=wave_items,
            plan=plan,
            targets=wave_targets,
            source_name=generated_name,
        )
        wave_roots.append(wave_root)
        offset += size

    smoke_root = subsets_root / "stage3_remaining_80pct_qwen38_smoke"
    replace_generated_root(smoke_root, args.force)
    write_subset(
        smoke_root,
        name="stage3_remaining_80pct_qwen38_smoke",
        full_l2=full_l2,
        l4_items=remaining_l4[:1],
        plan=plan,
        targets=remaining_targets[:1],
        source_name="stage3_remaining_80pct_qwen38_wave01",
    )

    archive_source_root = output_root / "stage3_full" / "_learning_archive_sources" / "qwen"
    archive_paths = {}
    referenced_l2_ids: list[str] = []
    for target in remaining_targets:
        for identifier in target.get("learning_task_ids", []):
            identifier = str(identifier).strip()
            if identifier and identifier not in referenced_l2_ids:
                referenced_l2_ids.append(identifier)
    full_l2_by_id = {task_id(item): item for item in full_l2}
    missing_l2 = sorted(set(referenced_l2_ids) - set(full_l2_by_id))
    if missing_l2:
        raise ValueError(f"Learning plan references unknown L2 tasks: {missing_l2}")

    for harness in HARNESSES:
        source_manifest = archive_source_root / harness / "source_manifest.json"
        sources = []
        for identifier in referenced_l2_ids:
            checkpoint_dir = (
                l2_results_root
                / harness
                / "basic_l2"
                / "exported_checkpoints"
                / identifier
            )
            if not (checkpoint_dir / "agent_checkpoint.json").is_file():
                raise FileNotFoundError(
                    f"Missing completed L2 checkpoint for {harness}/{identifier}: {checkpoint_dir}"
                )
            sources.append({"task_id": identifier, "checkpoint_dir": str(checkpoint_dir)})
        write_json(
            source_manifest,
            {
                "schema_version": "tongbench_l2_zero_shot_source_manifest_v1",
                "mode": "independent_l2_zero_shot_archive",
                "backend": harness,
                "model": args.model,
                "l2_tasks": sources,
                "targets": remaining_targets,
            },
        )
        archive_paths[harness] = str(source_manifest)

    selection_manifest = output_root / "stage3_full" / "remaining_selection_manifest.json"
    write_json(
        selection_manifest,
        {
            "schema_version": "tongbench_remaining_l4_selection_v1",
            "full_subset_root": str(full_root),
            "completed_subset_root": str(completed_root),
            "remaining_subset_root": str(remaining_root),
            "completed_l4_count": len(completed_l4),
            "remaining_l4_count": len(remaining_l4),
            "remaining_l4_task_ids": [task_id(item) for item in remaining_l4],
            "wave_sizes": list(WAVE_SIZES),
            "wave_roots": [str(path) for path in wave_roots],
            "smoke_root": str(smoke_root),
            "l2_results_root": str(l2_results_root),
            "referenced_l2_count": len(referenced_l2_ids),
            "archive_source_manifests": archive_paths,
        },
    )
    print(
        json.dumps(
            {
                "remaining_l4": len(remaining_l4),
                "completed_l4": len(completed_l4),
                "referenced_l2": len(referenced_l2_ids),
                "waves": list(WAVE_SIZES),
                "selection_manifest": str(selection_manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
