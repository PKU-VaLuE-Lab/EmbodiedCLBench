#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any


LEVELS = ("L2", "L4")
HEIGHTS = ("child", "adult")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def scene_sort_value(scene_id: str) -> tuple[int, str]:
    return (0 if scene_id == "001" else 1, scene_id)


def task_sort_value(task_id: str) -> tuple[int, str]:
    for level_index, level in enumerate(LEVELS):
        if f"_{level}_" in task_id:
            return (level_index, task_id)
    return (len(LEVELS), task_id)


def task_id_for_height(task_id: str, height: str) -> str:
    if height == "child":
        return task_id.replace("_adult_", "_child_")
    if height == "adult":
        return task_id.replace("_child_", "_adult_")
    raise ValueError(f"Unsupported height: {height}")


def child_task_id(task_id: str) -> str:
    return task_id_for_height(task_id, "child")


def task_level(task_id: str, item: dict[str, Any] | None = None) -> str:
    if item and str(item.get("level") or "").strip():
        return str(item["level"]).strip()
    for level in LEVELS:
        if f"_{level}_" in task_id:
            return level
    return ""


def infer_project_root(taskgen_root: Path) -> Path:
    # taskgen_root is <project>/important_results_taskgen/8.24_l2_l4_scaleup_v1.
    return taskgen_root.parent.parent


def strip_project_prefix(value: str, project_root: Path) -> str:
    normalized = value.replace("\\", "/")
    prefix = str(project_root).replace("\\", "/").rstrip("/") + "/"
    if normalized.startswith(prefix):
        return normalized[len(prefix) :]
    return value


def rewrite_string(value: str, *, source_task_id: str, target_task_id: str, height: str, project_root: Path) -> str:
    text = strip_project_prefix(value, project_root)
    if source_task_id != target_task_id:
        text = text.replace(source_task_id, target_task_id)
    if height == "adult":
        text = text.replace("/child/", "/adult/")
        text = text.replace("_child_", "_adult_")
    elif height == "child":
        text = text.replace("/adult/", "/child/")
        text = text.replace("_adult_", "_child_")
    return text


def rewrite_payload(value: Any, *, source_task_id: str, target_task_id: str, height: str, project_root: Path) -> Any:
    if isinstance(value, dict):
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key == "height" and isinstance(item, str) and item in HEIGHTS:
                rewritten[key] = height
            else:
                rewritten[key] = rewrite_payload(
                    item,
                    source_task_id=source_task_id,
                    target_task_id=target_task_id,
                    height=height,
                    project_root=project_root,
                )
        return rewritten
    if isinstance(value, list):
        return [
            rewrite_payload(
                item,
                source_task_id=source_task_id,
                target_task_id=target_task_id,
                height=height,
                project_root=project_root,
            )
            for item in value
        ]
    if isinstance(value, str):
        if value in HEIGHTS:
            return height
        return rewrite_string(
            value,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
        )
    return value


def safe_rmtree(path: Path, *, required_leaf: str) -> None:
    if not path.exists():
        return
    if path.name != required_leaf:
        raise RuntimeError(f"Refusing to remove unexpected path: {path}")
    shutil.rmtree(path)


def clean_and_mkdir(path: Path, *, required_leaf: str) -> None:
    safe_rmtree(path, required_leaf=required_leaf)
    path.mkdir(parents=True, exist_ok=True)


def load_formal_tasks(taskgen_root: Path) -> list[dict[str, Any]]:
    all_heights = taskgen_root / "formal_l2_l4_all_heights.json"
    if all_heights.exists():
        payload = read_json(all_heights)
        tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    else:
        payload = read_json(taskgen_root / "formal_l2_l4_child.json")
        tasks = [*payload.get("l2_tasks", []), *payload.get("l4_tasks", [])]
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError(f"No formal tasks found under {taskgen_root}")
    return [item for item in tasks if isinstance(item, dict) and item.get("task_id")]


def load_learning_plan(taskgen_root: Path) -> dict[str, Any]:
    path = taskgen_root / "learning_plan.json"
    if not path.exists():
        return {
            "schema_version": "tongbench_target_learning_plan_v2",
            "easy_level": "L2",
            "hard_level": "L4",
            "learning_count_per_target": 2,
            "shared_between_modes": True,
            "targets": [],
        }
    return read_json(path)


def prepare_learning_plan(plan: dict[str, Any], available_task_ids: set[str]) -> dict[str, Any]:
    raw_targets = [target for target in plan.get("targets", []) or [] if isinstance(target, dict)]
    target_ids = [str(target.get("target_task_id") or "") for target in raw_targets]
    already_all_heights = any("_adult_" in task_id for task_id in target_ids) and any("_child_" in task_id for task_id in target_ids)
    if already_all_heights:
        targets: list[dict[str, Any]] = []
        for raw_target in raw_targets:
            target_id = str(raw_target.get("target_task_id") or "")
            learning_ids = [str(task_id) for task_id in raw_target.get("learning_task_ids", []) or []]
            if target_id in available_task_ids and all(task_id in available_task_ids for task_id in learning_ids):
                copied = deepcopy(raw_target)
                copied["height"] = "adult" if "_adult_" in target_id else "child"
                copied.setdefault("source_child_target_task_id", child_task_id(target_id))
                targets.append(copied)
        prepared = dict(plan)
        prepared["targets"] = targets
        prepared["target_count"] = len(targets)
        prepared["heights"] = list(HEIGHTS)
        prepared["filter_note"] = "Learning plan already contains child and adult targets; unavailable targets were filtered out."
        return prepared

    targets: list[dict[str, Any]] = []
    for raw_target in raw_targets:
        target_id = str(raw_target.get("target_task_id") or "")
        if not target_id:
            continue
        for height in HEIGHTS:
            copied = rewrite_payload(
                raw_target,
                source_task_id=child_task_id(target_id),
                target_task_id=task_id_for_height(target_id, height),
                height=height,
                project_root=Path("/__unused_project_root__"),
            )
            copied["target_task_id"] = task_id_for_height(target_id, height)
            for key in ("learning_task_ids", "icl_learning_task_ids", "skill_learning_task_ids"):
                values = copied.get(key)
                if isinstance(values, list):
                    copied[key] = [task_id_for_height(str(task_id), height) for task_id in values]
            learning_ids = [str(task_id) for task_id in copied.get("learning_task_ids", []) or []]
            if copied["target_task_id"] in available_task_ids and all(task_id in available_task_ids for task_id in learning_ids):
                copied["height"] = height
                copied["source_child_target_task_id"] = child_task_id(target_id)
                targets.append(copied)
    prepared = dict(plan)
    prepared["targets"] = targets
    prepared["target_count"] = len(targets)
    prepared["heights"] = list(HEIGHTS)
    prepared["mirror_policy"] = "adult duplicates the child learning plan with task ids and image paths rewritten to adult."
    return prepared


def load_child_descriptions(taskgen_root: Path) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    index_path = taskgen_root / "organized" / "manifests" / "task_index.csv"
    if index_path.exists():
        with index_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                task_id = str(row.get("task_id") or "").strip()
                desc = str(row.get("description_highlevel_hard") or "").strip()
                if task_id and desc:
                    descriptions[task_id] = desc
    pipeline_root = taskgen_root / "api_pipeline"
    if pipeline_root.exists():
        for output_path in pipeline_root.glob("scene_*/task_*/stage1_task_output.json"):
            try:
                payload = read_json(output_path)
            except Exception:
                continue
            task_id = output_path.parent.name
            desc = str(payload.get("description_highlevel_hard") or payload.get("description_highlevel") or "").strip()
            if task_id and desc:
                descriptions.setdefault(task_id, desc)
    return descriptions


def load_child_option_rows(taskgen_root: Path) -> dict[str, list[dict[str, Any]]]:
    path = taskgen_root / "organized" / "final_option_banks" / "all_scenes_final_option_bank.with_scene.jsonl"
    rows = read_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if task_id:
            grouped[task_id].append(row)
    if not grouped:
        raise RuntimeError(f"No option-bank rows found: {path}")
    return dict(grouped)


def source_task_dir(eval_input_root: Path, scene_id: str, source_task_id: str) -> Path:
    path = eval_input_root / f"scene_{scene_id}" / "tasks" / source_task_id
    if not path.exists():
        raise FileNotFoundError(f"Missing source task directory: {path}")
    return path


def image_candidates(raw: str, *, height: str, project_root: Path) -> list[Path]:
    if not raw:
        return []
    values = [raw]
    if height == "adult":
        values.append(raw.replace("/child/", "/adult/").replace("_child_", "_adult_"))
    elif height == "child":
        values.append(raw.replace("/adult/", "/child/").replace("_adult_", "_child_"))

    candidates: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(project_root / path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def first_existing_image(state: dict[str, Any], *, height: str, project_root: Path) -> Path | None:
    raw_paths: list[str] = []
    for key in ("copied_image_path", "reused_image_path"):
        value = state.get(key)
        if value:
            raw_paths.append(str(value))
    for value in state.get("source_image_paths", []) or []:
        if value:
            raw_paths.append(str(value))
    for raw in raw_paths:
        for path in image_candidates(raw, height=height, project_root=project_root):
            if path.is_file():
                return path
    return None


def load_state_to_atomic_node_map(source_task_dir: Path) -> dict[str, str]:
    matches = sorted((source_task_dir / "input").glob("*_unique_states.json"))
    if not matches:
        return {}
    try:
        payload = read_json(matches[0])
    except Exception:
        return {}
    mapping: dict[str, str] = {}
    for state in payload.get("states", []) or []:
        if not isinstance(state, dict):
            continue
        state_id = str(state.get("state_id") or state.get("target_state_id") or "").strip()
        atomic_id = str(state.get("atomic_expanded_node_id") or state.get("node_id") or "").strip()
        if state_id and atomic_id:
            mapping[state_id] = atomic_id
    return mapping


def relative_posix(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()


def package_manifest(
    *,
    source_manifest_path: Path,
    destination_manifest_path: Path,
    package_root: Path,
    destination_task_dir: Path,
    source_task_id: str,
    target_task_id: str,
    height: str,
    project_root: Path,
) -> dict[str, Any]:
    source_payload = read_json(source_manifest_path)
    state_to_atomic = load_state_to_atomic_node_map(source_manifest_path.parent)
    payload = rewrite_payload(
        source_payload,
        source_task_id=source_task_id,
        target_task_id=target_task_id,
        height=height,
        project_root=project_root,
    )
    payload["task_id"] = target_task_id
    payload["height"] = height
    payload["source_child_task_id"] = source_task_id

    packaged_states: list[dict[str, Any]] = []
    missing_states: list[dict[str, Any]] = []
    states = source_payload.get("states", []) if isinstance(source_payload, dict) else []
    for index, raw_state in enumerate(state for state in states if isinstance(state, dict)):
        state = rewrite_payload(
            raw_state,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
        )
        image = first_existing_image(raw_state, height=height, project_root=project_root)
        state_id = str(state.get("target_state_id") or state.get("state_id") or f"state_{index:04d}")
        atomic_id = state_to_atomic.get(state_id)
        if atomic_id:
            state["atomic_expanded_node_id"] = atomic_id
        if image is None:
            state.pop("copied_image_path", None)
            state.pop("reused_image_path", None)
            state["source_image_paths"] = []
            state["match_status"] = "missing_image_in_packaging"
            missing_states.append({"state_id": state_id, "index": index})
        else:
            image_dst = destination_task_dir / "images" / state_id / image.name
            image_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_dst)
            rel = relative_posix(image_dst, package_root)
            state["source_image_paths"] = [rel]
            state["reused_image_path"] = rel
            state["copied_image_path"] = rel
            state["packaged_source_image"] = strip_project_prefix(str(image), project_root)
            state["match_status"] = "packaged"
        packaged_states.append(state)

    payload["states"] = packaged_states
    payload["state_count"] = len(packaged_states)
    payload["reused_state_count"] = len(packaged_states) - len(missing_states)
    payload["missing_state_count"] = len(missing_states)
    payload["reuse_rate"] = 0.0 if not packaged_states else round((len(packaged_states) - len(missing_states)) / len(packaged_states), 6)
    payload["missing_states"] = missing_states
    write_json(destination_manifest_path, payload)
    return payload


def final_description_for_task(item: dict[str, Any], descriptions: dict[str, str]) -> str:
    task_id = str(item.get("task_id") or "")
    base_task_id = child_task_id(task_id)
    for candidate in (
        descriptions.get(base_task_id),
        descriptions.get(task_id),
        item.get("description_highlevel_hard"),
        item.get("description_highlevel_en_draft"),
        item.get("description_highlevel"),
        item.get("description"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "Complete the household task in the current room."


def patch_task_description(payload: Any, description: str) -> Any:
    if isinstance(payload, dict):
        payload["description"] = description
        payload["description_highlevel_hard"] = description
        if isinstance(payload.get("task"), dict):
            payload["task"]["description"] = description
            payload["task"]["description_highlevel_hard"] = description
    return payload


def copy_graph_files(
    *,
    source_dir: Path,
    destination_dir: Path,
    package_root: Path,
    source_task_id: str,
    target_task_id: str,
    height: str,
    project_root: Path,
    description: str,
) -> list[str]:
    destination_input = destination_dir / "input"
    destination_input.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source_path in sorted((source_dir / "input").glob("*.json")):
        destination_name = source_path.name.replace(source_task_id, target_task_id)
        destination_path = destination_input / destination_name
        payload = read_json(source_path)
        payload = rewrite_payload(
            payload,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
        )
        if destination_name.endswith("_atomic_expanded_graph.json"):
            payload = patch_task_description(payload, description)
        if isinstance(payload, dict):
            payload["task_id"] = target_task_id
        write_json(destination_path, payload)
        copied.append(relative_posix(destination_path, package_root))
    if not copied:
        raise RuntimeError(f"No graph JSON files copied from {source_dir}")
    return copied


def package_ready_item(
    *,
    formal_item: dict[str, Any],
    destination_task_dir: Path,
    package_root: Path,
    source_task_id: str,
    target_task_id: str,
    height: str,
    scene_id: str,
    manifest_payload: dict[str, Any],
    copied_graph_files: list[str],
    description: str,
    project_root: Path,
) -> dict[str, Any]:
    ready = rewrite_payload(
        deepcopy(formal_item),
        source_task_id=source_task_id,
        target_task_id=target_task_id,
        height=height,
        project_root=project_root,
    )
    ready["task_id"] = target_task_id
    ready["level"] = task_level(target_task_id, ready)
    ready["difficulty_level"] = ready["level"]
    ready["height"] = height
    ready["scene_id"] = scene_id
    ready["description"] = description
    ready["description_highlevel_hard"] = description
    ready["source_child_task_id"] = source_task_id
    ready["task_dir"] = relative_posix(destination_task_dir, package_root)
    ready["copied_graph_files"] = copied_graph_files
    ready["state_count"] = int(manifest_payload.get("state_count") or 0)
    ready["reused_state_count"] = int(manifest_payload.get("reused_state_count") or 0)
    ready["missing_state_count"] = int(manifest_payload.get("missing_state_count") or 0)
    ready["reuse_rate"] = manifest_payload.get("reuse_rate")
    ready["states"] = manifest_payload.get("states", [])
    if isinstance(ready.get("task"), dict):
        ready["task"]["task_id"] = target_task_id
        ready["task"]["description"] = description
        ready["task"]["description_highlevel_hard"] = description
        ready["task"]["height"] = height
        ready["task"]["scene_id"] = scene_id
    return ready


def package_option_rows(
    *,
    rows: list[dict[str, Any]],
    source_task_id: str,
    target_task_id: str,
    height: str,
    scene_id: str,
    project_root: Path,
) -> list[dict[str, Any]]:
    packaged: list[dict[str, Any]] = []
    for raw_row in rows:
        row = rewrite_payload(
            deepcopy(raw_row),
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
        )
        row["task_id"] = target_task_id
        row["scene_id"] = scene_id
        row["height"] = height
        row["source_child_task_id"] = source_task_id
        row["source_final_option_bank"] = f"organized/final_option_banks/by_task/scene_{scene_id}/{source_task_id}.jsonl"
        packaged.append(row)
    return packaged


def build_dataset(
    *,
    taskgen_root: Path,
    eval_input_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    project_root = infer_project_root(taskgen_root)
    prepared_input_root = output_root / "input"
    merged_root = prepared_input_root / "merged"
    tasks_root = merged_root / "tasks"
    by_scene_root = prepared_input_root / "by_scene"

    clean_and_mkdir(prepared_input_root, required_leaf="input")
    tasks_root.mkdir(parents=True, exist_ok=True)
    by_scene_root.mkdir(parents=True, exist_ok=True)

    formal_tasks = load_formal_tasks(taskgen_root)
    descriptions = load_child_descriptions(taskgen_root)
    child_option_rows = load_child_option_rows(taskgen_root)

    l2_items: list[dict[str, Any]] = []
    l4_items: list[dict[str, Any]] = []
    option_rows: list[dict[str, Any]] = []
    package_records: list[dict[str, Any]] = []
    by_scene_ready: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"L2": [], "L4": []})
    by_scene_options: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in sorted(
        formal_tasks,
        key=lambda x: (
            scene_sort_value(str(x.get("scene_id") or "")),
            str(x.get("height") or ""),
            task_sort_value(str(x.get("task_id") or "")),
        ),
    ):
        target_task_id = str(item.get("task_id") or "").strip()
        if not target_task_id:
            continue
        height = str(item.get("height") or "child").strip() or "child"
        if height not in HEIGHTS:
            raise ValueError(f"Unsupported task height for {target_task_id}: {height}")
        source_task_id = child_task_id(target_task_id)
        scene_id = str(item.get("scene_id") or "").strip()
        if not scene_id:
            raise ValueError(f"Missing scene_id for task: {target_task_id}")

        source_dir = source_task_dir(eval_input_root, scene_id, source_task_id)
        source_manifest = source_dir / "task_image_manifest.json"
        if not source_manifest.exists():
            raise FileNotFoundError(f"Missing source manifest: {source_manifest}")
        rows = child_option_rows.get(source_task_id, [])
        if not rows:
            raise RuntimeError(f"Missing option bank rows for {source_task_id}")
        description = final_description_for_task(item, descriptions)

        destination_task_dir = tasks_root / target_task_id
        destination_task_dir.mkdir(parents=True, exist_ok=True)
        copied_graph_files = copy_graph_files(
            source_dir=source_dir,
            destination_dir=destination_task_dir,
            package_root=merged_root,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
            description=description,
        )
        manifest_payload = package_manifest(
            source_manifest_path=source_manifest,
            destination_manifest_path=destination_task_dir / "task_image_manifest.json",
            package_root=merged_root,
            destination_task_dir=destination_task_dir,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            project_root=project_root,
        )
        ready = package_ready_item(
            formal_item=item,
            destination_task_dir=destination_task_dir,
            package_root=merged_root,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            scene_id=scene_id,
            manifest_payload=manifest_payload,
            copied_graph_files=copied_graph_files,
            description=description,
            project_root=project_root,
        )
        packaged_rows = package_option_rows(
            rows=rows,
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            height=height,
            scene_id=scene_id,
            project_root=project_root,
        )

        level = ready["level"]
        if level == "L2":
            l2_items.append(ready)
        elif level == "L4":
            l4_items.append(ready)
        else:
            raise ValueError(f"Unsupported task level for {target_task_id}: {level}")
        option_rows.extend(packaged_rows)
        by_scene_ready[scene_id][level].append(ready)
        by_scene_options[scene_id].extend(packaged_rows)
        package_records.append(
            {
                "task_id": target_task_id,
                "source_child_task_id": source_task_id,
                "scene_id": scene_id,
                "height": height,
                "level": level,
                "state_count": manifest_payload["state_count"],
                "missing_state_count": manifest_payload["missing_state_count"],
                "option_state_count": len(packaged_rows),
            }
        )

    write_json(merged_root / "ready_l2_tasks_with_images.json", l2_items)
    write_json(merged_root / "ready_l4_tasks_with_images.json", l4_items)
    write_json(merged_root / "ready_highlevel_v2_tasks_with_images.json", [*l2_items, *l4_items])
    write_jsonl(merged_root / "option_bank_highlevel_v2_smoke.jsonl", option_rows)

    for scene_id in sorted(by_scene_ready, key=scene_sort_value):
        scene_root = by_scene_root / f"scene_{scene_id}"
        scene_l2 = by_scene_ready[scene_id]["L2"]
        scene_l4 = by_scene_ready[scene_id]["L4"]
        write_json(scene_root / "ready_l2_tasks_with_images.json", scene_l2)
        write_json(scene_root / "ready_l4_tasks_with_images.json", scene_l4)
        write_json(scene_root / "ready_highlevel_v2_tasks_with_images.json", [*scene_l2, *scene_l4])
        write_jsonl(scene_root / "option_bank_highlevel_v2_smoke.jsonl", by_scene_options[scene_id])

    available_task_ids = {str(item["task_id"]) for item in [*l2_items, *l4_items]}
    child_plan = load_learning_plan(taskgen_root)
    learning_plan = prepare_learning_plan(child_plan, available_task_ids)
    return {
        "project_root": project_root,
        "prepared_input_root": prepared_input_root,
        "merged_root": merged_root,
        "subset_root": output_root / "subsets",
        "l2_items": l2_items,
        "l4_items": l4_items,
        "option_rows": option_rows,
        "learning_plan": learning_plan,
        "package_records": package_records,
    }


def task_ids(items: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("task_id") or "") for item in items if item.get("task_id")}


def group_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("height") or "unknown"), str(item.get("scene_id") or "unknown"))


def ratio_sample(items: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[group_key(item)].append(item)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        bucket = sorted(groups[key], key=lambda item: task_sort_value(str(item.get("task_id") or "")))
        keep = max(1, math.ceil(len(bucket) * ratio))
        selected.extend(bucket[:keep])
    return selected


def learning_target_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(target.get("target_task_id") or ""): target
        for target in plan.get("targets", []) or []
        if isinstance(target, dict) and target.get("target_task_id")
    }


def l2_items_covering_learning(
    *,
    l2_items: list[dict[str, Any]],
    base_l2_items: list[dict[str, Any]],
    l4_items: list[dict[str, Any]],
    learning_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_ids = task_ids(base_l2_items)
    target_ids = task_ids(l4_items)
    plan_by_target = learning_target_map(learning_plan)
    for target_id in target_ids:
        target = plan_by_target.get(target_id)
        if not target:
            continue
        selected_ids.update(str(task_id) for task_id in target.get("learning_task_ids", []) or [])
    return [item for item in l2_items if str(item.get("task_id") or "") in selected_ids]


def filter_learning_plan(plan: dict[str, Any], selected_l4_ids: set[str], available_l2_ids: set[str], option_task_ids: set[str]) -> dict[str, Any]:
    targets = []
    for target in plan.get("targets", []) or []:
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_task_id") or "")
        learning_ids = [str(task_id) for task_id in target.get("learning_task_ids", []) or []]
        if (
            target_id in selected_l4_ids
            and target_id in option_task_ids
            and all(task_id in option_task_ids for task_id in learning_ids)
            and all(task_id in available_l2_ids for task_id in learning_ids)
        ):
            targets.append(target)
    filtered = dict(plan)
    filtered["targets"] = targets
    filtered["target_count"] = len(targets)
    filtered["filter_note"] = "Only L4 targets whose task and L2 learning tasks are present in this subset are kept."
    return filtered


def write_subset(
    *,
    subset_root: Path,
    name: str,
    l2_items: list[dict[str, Any]],
    l4_items: list[dict[str, Any]],
    learning_plan: dict[str, Any],
    option_task_ids: set[str],
) -> dict[str, Any]:
    root = subset_root / name
    plan = filter_learning_plan(learning_plan, task_ids(l4_items), task_ids(l2_items), option_task_ids)
    selected_l4_ids = {str(target.get("target_task_id") or "") for target in plan.get("targets", [])}
    if selected_l4_ids:
        l4_items = [item for item in l4_items if str(item.get("task_id") or "") in selected_l4_ids]
    write_json(root / "ready_l2_tasks_with_images.json", l2_items)
    write_json(root / "ready_l4_tasks_with_images.json", l4_items)
    write_json(root / "ready_highlevel_v2_tasks_with_images.json", [*l2_items, *l4_items])
    write_json(root / "learning_plan.json", plan)
    summary = {
        "subset": name,
        "l2_task_count": len(l2_items),
        "l4_task_count": len(l4_items),
        "learning_target_count": len(plan.get("targets", [])),
        "height_counts_l2": dict(Counter(str(item.get("height") or "") for item in l2_items)),
        "height_counts_l4": dict(Counter(str(item.get("height") or "") for item in l4_items)),
        "scene_counts_l2": dict(Counter(str(item.get("scene_id") or "") for item in l2_items)),
        "scene_counts_l4": dict(Counter(str(item.get("scene_id") or "") for item in l4_items)),
        "l2_task_ids": sorted(task_ids(l2_items)),
        "l4_task_ids": sorted(task_ids(l4_items)),
    }
    write_json(root / "subset_summary.json", summary)
    return summary


def build_subsets(
    *,
    subset_root: Path,
    l2_items: list[dict[str, Any]],
    l4_items: list[dict[str, Any]],
    learning_plan: dict[str, Any],
    option_task_ids: set[str],
    stage2_ratio: float,
) -> list[dict[str, Any]]:
    clean_and_mkdir(subset_root, required_leaf="subsets")
    plan_by_target = learning_target_map(learning_plan)
    l4_with_learning = [
        item
        for item in l4_items
        if str(item.get("task_id") or "") in plan_by_target
    ]
    if not l4_with_learning:
        raise RuntimeError("No L4 task has complete learning coverage.")

    stage0_l4 = l4_with_learning[:1]
    stage0_learning_ids: set[str] = set()
    if stage0_l4:
        target = plan_by_target.get(str(stage0_l4[0].get("task_id") or ""))
        if target:
            stage0_learning_ids.update(str(task_id) for task_id in target.get("learning_task_ids", []) or [])
    stage0_l2 = [item for item in l2_items if str(item.get("task_id") or "") in stage0_learning_ids]
    if not stage0_l2:
        stage0_l2 = l2_items[:1]
    stage2_l4 = ratio_sample(l4_with_learning, stage2_ratio)
    stage2_l2_base = ratio_sample(l2_items, stage2_ratio)
    stage2_l2 = l2_items_covering_learning(
        l2_items=l2_items,
        base_l2_items=stage2_l2_base,
        l4_items=stage2_l4,
        learning_plan=learning_plan,
    )

    return [
        write_subset(
            subset_root=subset_root,
            name="stage0_smoke",
            l2_items=stage0_l2,
            l4_items=stage0_l4,
            learning_plan=learning_plan,
            option_task_ids=option_task_ids,
        ),
        write_subset(
            subset_root=subset_root,
            name="stage2_20pct",
            l2_items=stage2_l2,
            l4_items=stage2_l4,
            learning_plan=learning_plan,
            option_task_ids=option_task_ids,
        ),
        write_subset(
            subset_root=subset_root,
            name="stage3_full",
            l2_items=l2_items,
            l4_items=l4_with_learning,
            learning_plan=learning_plan,
            option_task_ids=option_task_ids,
        ),
    ]


def validate_packaged_dataset(merged_root: Path, l2_items: list[dict[str, Any]], l4_items: list[dict[str, Any]], option_rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[str] = []
    all_items = [*l2_items, *l4_items]
    item_ids = task_ids(all_items)
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    option_role_counts: Counter[str] = Counter()
    success_position_counts: Counter[str] = Counter()
    for row in option_rows:
        task_id = str(row.get("task_id") or "")
        rows_by_task[task_id].append(row)
        choices = row.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 4:
            issues.append(f"{task_id}/{row.get('state_id')}: expected 4 choices")
            continue
        for index, choice in enumerate(choices):
            role = str(choice.get("role") or "")
            option_role_counts[role] += 1
            if role == "success_path":
                success_position_counts["ABCD"[index]] += 1
    for item in all_items:
        task_id = str(item.get("task_id") or "")
        task_dir = merged_root / "tasks" / task_id
        if not task_dir.exists():
            issues.append(f"{task_id}: missing packaged task dir")
        manifest = task_dir / "task_image_manifest.json"
        if not manifest.exists():
            issues.append(f"{task_id}: missing packaged manifest")
            continue
        payload = read_json(manifest)
        if int(payload.get("reused_state_count") or 0) <= 0:
            issues.append(f"{task_id}: no packaged images")
        usable_node_ids: set[str] = set()
        for state in payload.get("states", []) or []:
            if not isinstance(state, dict):
                continue
            has_image = bool(state.get("reused_image_path") or state.get("copied_image_path") or state.get("source_image_paths"))
            if not has_image:
                continue
            for key in ("atomic_expanded_node_id", "node_id", "target_node_id", "state_id", "target_state_id"):
                value = str(state.get(key) or "").strip()
                if value:
                    usable_node_ids.add(value)
        for row in rows_by_task.get(task_id, []):
            state_id = str(row.get("state_id") or "").strip()
            if state_id and state_id not in usable_node_ids:
                issues.append(f"{task_id}/{state_id}: option-bank state has no packaged image")
            for choice in row.get("choices", []) or []:
                if not isinstance(choice, dict):
                    continue
                to_state = str(choice.get("to_state") or "").strip()
                if to_state and to_state not in usable_node_ids:
                    issues.append(f"{task_id}/{state_id}: choice to_state {to_state} has no packaged image")
        graph = task_dir / "input" / f"{task_id}_atomic_expanded_graph.json"
        if not graph.exists():
            issues.append(f"{task_id}: missing graph JSON")
        if task_id not in rows_by_task:
            issues.append(f"{task_id}: no option-bank rows")
    extra_rows = sorted(set(rows_by_task) - item_ids)
    if extra_rows:
        issues.append(f"option bank has rows for tasks not in ready JSON: {extra_rows[:10]}")

    symlinks = [str(path.relative_to(merged_root)) for path in merged_root.rglob("*") if path.is_symlink()]
    if symlinks:
        issues.append(f"packaged input contains symlinks: {symlinks[:10]}")

    return {
        "ok": not issues,
        "issues": issues,
        "l2_task_count": len(l2_items),
        "l4_task_count": len(l4_items),
        "total_task_count": len(all_items),
        "option_row_count": len(option_rows),
        "option_task_count": len(rows_by_task),
        "role_counts": dict(option_role_counts),
        "success_position_counts": dict(success_position_counts),
        "symlink_count": len(symlinks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare self-contained For_user L2/L4 eval inputs and subsets.")
    parser.add_argument("--taskgen-root", required=True)
    parser.add_argument("--eval-input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage2-ratio", type=float, default=0.2)
    args = parser.parse_args()

    taskgen_root = Path(args.taskgen_root).resolve()
    eval_input_root = Path(args.eval_input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        taskgen_root=taskgen_root,
        eval_input_root=eval_input_root,
        output_root=output_root,
    )
    l2_items = dataset["l2_items"]
    l4_items = dataset["l4_items"]
    option_rows = dataset["option_rows"]
    option_task_ids = {str(row.get("task_id") or "") for row in option_rows if row.get("task_id")}
    subset_summaries = build_subsets(
        subset_root=dataset["subset_root"],
        l2_items=l2_items,
        l4_items=l4_items,
        learning_plan=dataset["learning_plan"],
        option_task_ids=option_task_ids,
        stage2_ratio=float(args.stage2_ratio),
    )
    validation = validate_packaged_dataset(dataset["merged_root"], l2_items, l4_items, option_rows)

    manifest = {
        "dataset_name": "tongbench-l2-l4-scaleup-v1",
        "package_schema_version": "self_contained_v1",
        "taskgen_root": str(taskgen_root),
        "eval_input_root": str(eval_input_root),
        "prepared_input_root": str(dataset["prepared_input_root"]),
        "merged_results_root": str(dataset["merged_root"]),
        "uses_320x180_images": True,
        "height_counts": dict(Counter(str(item.get("height") or "") for item in [*l2_items, *l4_items])),
        "scene_counts": dict(Counter(str(item.get("scene_id") or "") for item in [*l2_items, *l4_items])),
        "level_counts": dict(Counter(str(item.get("level") or "") for item in [*l2_items, *l4_items])),
        "subsets": subset_summaries,
        "validation": validation,
    }
    write_json(output_root / "prepare_manifest.json", manifest)
    portable_manifest = deepcopy(manifest)
    portable_manifest["taskgen_source"] = "important_results_taskgen/8.24_l2_l4_scaleup_v1"
    portable_manifest["eval_input_source"] = "important_results_eval/8.24_l2_l4_scaleup_v1/input"
    portable_manifest["prepared_input_root"] = "input"
    portable_manifest["merged_results_root"] = "input/merged"
    portable_manifest.pop("taskgen_root", None)
    portable_manifest.pop("eval_input_root", None)
    write_json(dataset["prepared_input_root"] / "package_manifest.json", portable_manifest)
    if not validation["ok"]:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise SystemExit("Packaged dataset validation failed.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
