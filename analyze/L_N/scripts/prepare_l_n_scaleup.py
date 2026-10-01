#!/usr/bin/env python3
"""Build the L_N L3/L5/L6 scale-up composition bundle."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from tongbench_taskgen.cli.build_highlevel_v2_smoke_input import (
    _build_composite_tasks,
    _load_l1_task_from_inventory,
    _prepare_l1_sources,
)

LEVELS = ("L3", "L5", "L6")
SOURCE_ROOT = "important_results/For_yanan/final_scene_results_two_heights_320x180/child/scenes"
TASKGEN_ROOT = "important_results_taskgen/8.24_l2_l4_scaleup_v1"
PROJECT_ROOT: Path


def scoped_id(scene: str, task_id: str) -> str:
    # Keep the task_L1_ prefix expected by the existing image-reuse indexer,
    # while still making the same source task from different scenes unique.
    return f"{task_id}__scene_{scene}"


def original_ids_of(task: dict[str, Any]) -> list[str]:
    value = task.get("original_source_l1_task_ids")
    return [str(x) for x in value] if isinstance(value, list) else ids_of(task)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ids_of(task: dict[str, Any]) -> list[str]:
    for value in (
        task.get("source_l1_task_ids"),
        (task.get("component_sources") or {}).get("mapped_source_l1_task_ids"),
        (task.get("composition") or {}).get("source_l1_task_ids"),
    ):
        if isinstance(value, list) and value:
            return [str(x) for x in value]
    return []


def level_of(task: dict[str, Any]) -> str:
    return str(task.get("difficulty_level") or task.get("requested_difficulty_level") or task.get("level") or "").upper()


def scene_of(task: dict[str, Any]) -> str:
    return str(task.get("scene_id") or task.get("scene") or "").strip()


def action_of(item: dict[str, Any]) -> str:
    return str(item.get("last_action_type") or item.get("semantic_action") or "").strip().lower()


def object_type_of(item: dict[str, Any]) -> str:
    types = item.get("object_types") or []
    return str(item.get("primary_object_type") or item.get("target_object_type") or (types[0] if types else "")).lower()


def objects_of(item: dict[str, Any]) -> set[str]:
    return {str(x) for x in item.get("objects", []) or []}


def description_of(item: dict[str, Any]) -> str:
    return str(item.get("description") or item.get("task_description") or "").strip()


def merge_variant(task_id: str, level: str, source_ids: list[str], inventory: dict[str, dict[str, Any]], cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    try:
        tasks: dict[str, dict[str, Any]] = {}
        for source_id in source_ids:
            if source_id not in cache:
                cache[source_id] = _load_l1_task_from_inventory(PROJECT_ROOT, inventory[source_id])
                cache[source_id]["task_id"] = source_id
                cache[source_id]["source_original_task_id"] = inventory[source_id].get("original_task_id")
            tasks[source_id] = cache[source_id]
        rows = _build_composite_tasks(
            candidates=[{"task_id": task_id, "level": level, "source_l1_task_ids": source_ids}],
            l1_tasks=tasks,
        )
        return rows[0] if rows else None
    except Exception:
        return None


def score(source_ids: list[str], parent_ids: set[str], inventory: dict[str, dict[str, Any]]) -> tuple[int, int, int, int, str]:
    actions = {action_of(inventory[x]) for x in source_ids}
    types = {object_type_of(inventory[x]) for x in source_ids}
    objects = set().union(*(objects_of(inventory[x]) for x in source_ids))
    return (len(set(source_ids) & parent_ids), len(actions), len(types), len(objects), "|".join(source_ids))


def addition_candidates(base_ids: list[str], all_ids: list[str], inventory: dict[str, dict[str, Any]], count: int) -> list[tuple[str, ...]]:
    base_actions = {action_of(inventory[x]) for x in base_ids}
    base_objects = set().union(*(objects_of(inventory[x]) for x in base_ids))
    ranked = sorted(
        (x for x in all_ids if x not in base_ids),
        key=lambda x: (
            int(action_of(inventory[x]) in base_actions),
            int(bool(base_objects & objects_of(inventory[x]))),
            -len(objects_of(inventory[x])),
            x,
        ),
    )
    return [(x,) for x in ranked] if count == 1 else list(itertools.combinations(ranked, count))


def choose_strict_chain(
    task_id_prefix: str,
    parent: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    used: dict[str, set[tuple[str, ...]]],
) -> dict[str, tuple[list[str], str]]:
    """Choose one nested L3/L5/L6 chain for a fixed L4 parent."""
    parent_ids = ids_of(parent)
    if len(parent_ids) != 4:
        raise RuntimeError(f"strict L_N requires four L1s in parent {parent.get('task_id')}, got {parent_ids}")
    scene = scene_of(parent)
    all_ids = [x for x, item in inventory.items() if scene_of(item) == scene]

    l3_trials: list[tuple[list[str], str]] = []
    for removed in parent_ids:
        remaining = [x for x in parent_ids if x != removed]
        for permutation in itertools.permutations(remaining):
            chosen = list(permutation)
            if tuple(chosen) in used["L3"]:
                continue
            if merge_variant(f"{task_id_prefix}_L3", "L3", chosen, inventory, cache) is None:
                continue
            rationale = (
                f"严格嵌套链：L3 从同房间父 L4 移除 {removed}（{description_of(inventory[removed])}），"
                f"保留 {', '.join(chosen)} 作为三 L1 子目标；为满足真实前置关系采用该执行顺序，"
                "并通过 composite builder 合法性检查。"
            )
            l3_trials.append((chosen, rationale))
            break
    if not l3_trials:
        raise RuntimeError(f"no legal strict L3 variant for {parent.get('task_id')} in scene {scene}")
    l3_ids, l3_rationale = l3_trials[0]

    for l5_add in addition_candidates(parent_ids, all_ids, inventory, 1):
        l5_ids = parent_ids + list(l5_add)
        if tuple(l5_ids) in used["L5"]:
            continue
        if merge_variant(f"{task_id_prefix}_L5", "L5", l5_ids, inventory, cache) is None:
            continue
        l5_added = l5_add[0]
        l5_rationale = (
            f"严格嵌套链：L5 完整保留父 L4 的四个 L1，并增加 {l5_added} "
            f"（{description_of(inventory[l5_added])}）；与 L4 同房间且通过 composite builder 合法性检查。"
        )
        for l6_add in addition_candidates(l5_ids, all_ids, inventory, 1):
            l6_ids = l5_ids + list(l6_add)
            if tuple(l6_ids) in used["L6"]:
                continue
            if merge_variant(f"{task_id_prefix}_L6", "L6", l6_ids, inventory, cache) is None:
                continue
            l6_added = l6_add[0]
            l6_rationale = (
                f"严格嵌套链：L6 完整保留 L5 的五个 L1，再增加 {l6_added} "
                f"（{description_of(inventory[l6_added])}）；形成 L3 ⊂ L4 ⊂ L5 ⊂ L6，"
                f"六个 L1 均来自同一房间并通过 composite builder 合法性检查。"
            )
            return {
                "L3": (l3_ids, l3_rationale),
                "L5": (l5_ids, l5_rationale),
                "L6": (l6_ids, l6_rationale),
            }
    raise RuntimeError(f"no legal strict L5/L6 chain for {parent.get('task_id')} in scene {scene}")
def choose_variant(
    task_id: str,
    level: str,
    parent: dict[str, Any],
    bases: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    used: set[tuple[str, ...]],
) -> tuple[list[str], str]:
    parent_ids = set(ids_of(parent))
    scene = scene_of(parent)
    all_ids = [x for x, item in inventory.items() if scene_of(item) == scene]
    trials: list[tuple[tuple[int, int, int, int, str], list[str], str]] = []
    ordered_bases = sorted(bases, key=lambda x: (-len(set(ids_of(x)) & parent_ids), str(x.get("task_id") or "")))

    for base in ordered_bases:
        base_ids = ids_of(base)
        if level != "L3" and len(base_ids) != 4:
            continue
        if level == "L3" and len(base_ids) == 4:
            variants = [([x for x in base_ids if x != removed], (removed,)) for removed in base_ids]
        elif level == "L3" and len(base_ids) == 2:
            variants = [
                (base_ids + [added], (added,))
                for added in addition_candidates(base_ids, all_ids, inventory, 1)
            ]
        elif level == "L3":
            variants = []
        else:
            count = 1 if level == "L5" else 2
            variants = [(base_ids + list(added), added) for added in addition_candidates(base_ids, all_ids, inventory, count)]
        for chosen, changed in variants:
            if tuple(chosen) in used:
                continue
            if merge_variant(task_id, level, chosen, inventory, cache) is None:
                continue
            if level == "L3" and len(base_ids) == 4:
                rationale = f"以 {base.get('task_id')} 为基准移除 {changed[0]}；保留的三个真实 L1 在 Graph 中可合法合并。"
            elif level == "L3":
                rationale = f"以 {base.get('task_id')} 的两个 L1 为基础增加 {changed[0]}；三者来自同一房间并通过 Graph 合法性验证。"
            else:
                details = "、".join(f"{x}（{description_of(inventory[x])}）" for x in changed)
                rationale = f"在 {base.get('task_id')} 的合法同房间组合上增加 {details}；新增动作/对象优先与父组合形成差异，并通过 Graph 合法性验证。"
            trials.append((score(chosen, parent_ids, inventory), chosen, rationale))
        if len(trials) >= 24:
            break

    if not trials:
        raise RuntimeError(f"no legal {level} variant for {parent.get('task_id')} in scene {scene}")
    trials.sort(key=lambda x: x[0], reverse=True)
    return trials[0][1], trials[0][2]


def load_base_records(external: Path, scenes: set[str], source_map: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        path = external / TASKGEN_ROOT / "output" / f"scene_{scene}" / "tasks_compositional.json"
        if not path.is_file():
            continue
        for row in load_json(path):
            if isinstance(row, dict) and level_of(row) in {"L2", "L4"}:
                row = dict(row)
                row.setdefault("scene_id", scene)
                raw_ids = ids_of(row)
                row["source_l1_task_ids"] = [source_map[(scene, item)] for item in raw_ids]
                rows.append(row)
    return rows


def initial_image(external: Path, scene: str) -> Path:
    root = external / SOURCE_ROOT / scene
    preferred = sorted((root / "task_L1_066/output/task_L1_066_unique_states/000_state_0000").glob("*.jpg"))
    if preferred:
        return preferred[0]
    matches = sorted(root.glob("task_L1_*/output/*_unique_states/000_state_0000/*.jpg"))
    if not matches:
        raise FileNotFoundError(f"no child state_0000 image for scene {scene}")
    return matches[0]


def stage_source_tasks(external: Path, bundle: Path, scene: str, source_ids: list[str], inventory: dict[str, dict[str, Any]]) -> Path:
    """Create namespaced source-task directories without copying image bytes."""
    root = bundle / "input/source_staging" / f"scene_{scene}"
    root.mkdir(parents=True, exist_ok=True)
    original_root = external / SOURCE_ROOT / scene
    for source_id in source_ids:
        original_id = str(inventory[source_id]["original_task_id"])
        original = original_root / original_id
        staged = root / source_id
        if staged.exists() or staged.is_symlink():
            if staged.is_symlink() or staged.is_file():
                staged.unlink()
            else:
                shutil.rmtree(staged)
        (staged / "input").mkdir(parents=True, exist_ok=True)
        for path in sorted((original / "input").glob(f"{original_id}_*.json")):
            suffix = path.name[len(original_id):]
            shutil.copy2(path, staged / "input" / f"{source_id}{suffix}")
        original_images = original / "output" / f"{original_id}_unique_states"
        if original_images.is_dir():
            (staged / "output").mkdir(parents=True, exist_ok=True)
            (staged / "output" / f"{source_id}_unique_states").symlink_to(original_images)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--parent-ready", type=Path, default=Path("analyze/N-sample/input/subset_10pct_child/ready_l4_tasks_with_images.json"))
    parser.add_argument("--learning-plan", type=Path, default=Path("analyze/N-sample/input/subset_10pct_child/learning_plan_k2.json"))
    parser.add_argument("--learning-checkpoint-root", type=Path, default=Path("analyze/N-sample/output/full/_learning_checkpoints/qwen3_7_plus/hermesagent/k2/hermesagent/qwen3_7_plus"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-chain", action="store_true")
    parser.add_argument("--parent-count", type=int, default=0)
    parser.add_argument("--task-name", default="LN_scaleup_v1")
    args = parser.parse_args()

    global PROJECT_ROOT
    PROJECT_ROOT = args.external_root.resolve()
    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    if args.force and bundle.exists():
        shutil.rmtree(bundle)

    parent_path = args.parent_ready.resolve() if args.parent_ready.is_absolute() else repo / args.parent_ready
    raw_parents = load_json(parent_path)
    if not isinstance(raw_parents, list):
        raise ValueError("parent-ready JSON must contain a list")
    if args.parent_count:
        if args.parent_count < 1:
            raise ValueError("--parent-count must be positive when provided")
        if len(raw_parents) < args.parent_count:
            raise ValueError(f"requested {args.parent_count} parents, got only {len(raw_parents)}")
        raw_parents = raw_parents[:args.parent_count]
    elif len(raw_parents) != 12:
        raise ValueError(f"expected 12 parent L4 rows for legacy mode, got {len(raw_parents)}")
    scenes = {scene_of(x) for x in raw_parents}
    inventory_rows = load_json(external / TASKGEN_ROOT / "l1_inventory_child.json").get("tasks", [])
    source_map = {
        (str(x.get("scene_id") or ""), str(x.get("task_id") or "")): scoped_id(str(x.get("scene_id") or ""), str(x.get("task_id") or ""))
        for x in inventory_rows if isinstance(x, dict) and x.get("task_id")
    }
    inventory = {}
    for row in inventory_rows:
        if not isinstance(row, dict) or not row.get("task_id"):
            continue
        scene = str(row.get("scene_id") or "")
        original = str(row["task_id"])
        item = dict(row)
        item["original_task_id"] = original
        item["task_id"] = scoped_id(scene, original)
        inventory[item["task_id"]] = item
    parents = []
    for raw in raw_parents:
        parent = deepcopy(raw)
        scene = scene_of(parent)
        parent["original_source_l1_task_ids"] = ids_of(parent)
        parent["source_l1_task_ids"] = [source_map[(scene, item)] for item in ids_of(parent)]
        parents.append(parent)

    bases = load_base_records(external, scenes, source_map)
    known = {str(x.get("task_id") or "") for x in bases}
    bases.extend(x for x in parents if str(x.get("task_id") or "") not in known)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bases:
        by_scene[scene_of(row)].append(row)

    cache: dict[str, dict[str, Any]] = {}
    selected: dict[str, list[dict[str, Any]]] = {x: [] for x in LEVELS}
    used: dict[str, set[tuple[str, ...]]] = {x: set() for x in LEVELS}
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for serial, parent in enumerate(parents, 1):
        parent_id = str(parent.get("task_id") or "")
        strict_chain = choose_strict_chain(
            f"task_{args.task_name}_{serial:03d}", parent, inventory, cache, used
        ) if args.strict_chain else None
        for level in LEVELS:
            task_id = f"task_{level}_{args.task_name}_{serial:03d}"
            try:
                if strict_chain is not None:
                    source_ids, rationale = strict_chain[level]
                else:
                    source_ids, rationale = choose_variant(task_id, level, parent, by_scene[scene_of(parent)], inventory, cache, used[level])
            except RuntimeError as exc:
                if level != "L6":
                    raise
                skipped_rows.append({"task_id": task_id, "level": level, "scene_id": scene_of(parent), "source_parent_task_id": parent_id, "reason": str(exc)})
                continue
            merged = merge_variant(task_id, level, source_ids, inventory, cache)
            if merged is None:
                raise RuntimeError(f"final merge failed: {task_id}")
            merged = deepcopy(merged)
            merged.update({
                "task_id": task_id,
                "difficulty_level": level,
                "requested_difficulty_level": level,
                "scene_id": scene_of(parent),
                "height": "child",
                "source_parent_task_id": parent_id,
                "source_l4_parent_source_l1_task_ids": original_ids_of(parent),
                "source_l4_parent_scoped_l1_task_ids": ids_of(parent),
                "selection_rationale_zh": rationale,
                "source_l1_selection_notes": [
                    f"{x}: {description_of(inventory[x])}; action={action_of(inventory[x])}; object_type={object_type_of(inventory[x])}"
                    for x in source_ids
                ],
            })
            selected[level].append(merged)
            used[level].add(tuple(source_ids))
            manifest_rows.append({
                "task_id": task_id,
                "level": level,
                "scene_id": scene_of(parent),
                "source_parent_task_id": parent_id,
                "source_l1_task_ids": source_ids,
                "source_l1_original_task_ids": [inventory[x].get("original_task_id") for x in source_ids],
                "source_l1_descriptions": [description_of(inventory[x]) for x in source_ids],
                "source_l1_actions": [action_of(inventory[x]) for x in source_ids],
                "source_l1_object_types": [object_type_of(inventory[x]) for x in source_ids],
                "selection_rationale_zh": rationale,
            })

    selected_l1 = sorted({x for rows in selected.values() for task in rows for x in ids_of(task)})
    taskgen = bundle / "input/taskgen_output"
    source_root = bundle / "input/l1_sources"
    taskgen.mkdir(parents=True, exist_ok=True)
    for scene in sorted(scenes):
        scene_ids = [x for x in selected_l1 if scene_of(inventory[x]) == scene]
        if scene_ids:
            staged_scene_root = stage_source_tasks(external, bundle, scene, scene_ids, inventory)
            _prepare_l1_sources(
                project_root=external,
                source_scene_root=staged_scene_root,
                source_l1_dir=source_root / f"scene_{scene}",
                taskgen_output_dir=taskgen,
                initial_image=initial_image(external, scene),
                l1_task_ids=scene_ids,
                l1_tasks={x: cache[x] for x in scene_ids},
                force=False,
            )
            # The copied graph payloads originate from the unscoped source
            # files.  Align their embedded task_id with the scoped directory
            # name so the existing image indexer can join states to images.
            for task_dir in sorted((source_root / f"scene_{scene}" / "tasks").glob("task_L1_*")):
                for unique_path in sorted((task_dir / "input").glob("*_unique_states.json")):
                    payload = load_json(unique_path)
                    if isinstance(payload, dict) and payload.get("task_id") != task_dir.name:
                        payload["task_id"] = task_dir.name
                        write_json(unique_path, payload)

    # The shared packager expects one source-reuse/tasks directory. Keep the
    # per-scene organization above, and expose task directories as symlinks so
    # images are not duplicated before image reuse is run.
    source_tasks = source_root / "tasks"
    source_tasks.mkdir(parents=True, exist_ok=True)
    for scene in sorted(scenes):
        scene_tasks = source_root / f"scene_{scene}" / "tasks"
        for task_dir in sorted(scene_tasks.glob("task_L1_*")):
            link = source_tasks / task_dir.name
            if not link.exists():
                link.symlink_to(task_dir)

    composites = [task for level in LEVELS for task in selected[level]]
    write_json(taskgen / "tasks_compositional.json", [cache[x] for x in selected_l1] + composites)
    # The composite task ids are intentionally scene-scoped to avoid collisions
    # between rooms.  Export the same scoped ids in the inventory consumed by
    # the API adapter; the source graph paths remain the original external paths.
    write_json(taskgen / "l1_inventory_child_scoped.json", {
        "schema_version": "tongbench_l1_inventory_scoped_v1",
        "source_inventory": str(external / TASKGEN_ROOT / "l1_inventory_child.json"),
        "tasks": [inventory[x] for x in selected_l1],
    })
    write_json(taskgen / "task_selection_manifest.json", {
        "schema_version": "tongbench_l_n_scaleup_selection_v1",
        "parent_source": str(parent_path),
        "parent_count": len(parents),
        "parent_task_ids": [str(x.get("task_id") or "") for x in parents],
        "target_count_by_level": {x: len(selected[x]) for x in LEVELS},
        "selected_l1_count": len(selected_l1),
        "selected_l1_task_ids": selected_l1,
        "selection_rows": manifest_rows,
        "skipped_rows": skipped_rows,
    })
    write_json(bundle / "input/design.json", {
        "schema_version": "tongbench_l_n_scaleup_v1",
        "height": "child",
        "scenes": sorted(scenes),
        "levels": list(LEVELS),
        "tasks_per_level": {x: len(selected[x]) for x in LEVELS},
        "tasks_total": len(composites),
        "skipped_tasks": skipped_rows,
        "l4_reused_for_learning_only": True,
        "learning_policy": "Reuse each parent L4's existing two-L2 checkpoint; do not rerun L2/L4 learning.",
        "graph_policy": {"same_scene_required": True, "preferred_state_targets": {"L3": 45, "L5": 75, "L6": 90}, "maximum_states": {"L3": 70, "L5": 100, "L6": 120}},
        "api_concurrency": {"stage1": 16, "stage2": 64, "stage3": 16},
        "evo_concurrency": 4,
    })
    learning_plan_path = args.learning_plan.resolve() if args.learning_plan.is_absolute() else repo / args.learning_plan
    learning_plan = load_json(learning_plan_path)
    plan_rows = {
        str(row.get("target_task_id") or ""): row
        for row in learning_plan.get("targets", [])
        if isinstance(row, dict)
    }
    checkpoint_root = args.learning_checkpoint_root.resolve() if args.learning_checkpoint_root.is_absolute() else repo / args.learning_checkpoint_root
    reuse_entries = {}
    missing_reuse = []
    for task in composites:
        target_id = str(task.get("task_id") or "")
        parent_id = str(task.get("source_parent_task_id") or "")
        row = plan_rows.get(parent_id)
        checkpoint_dir = checkpoint_root / parent_id
        summary_path = checkpoint_dir / "learning_run/sequence_summary.json"
        if row is None or not (checkpoint_dir / "checkpoint/agent_checkpoint.json").is_file() or not summary_path.is_file():
            missing_reuse.append({"task_id": target_id, "source_parent_task_id": parent_id, "checkpoint_dir": str(checkpoint_dir)})
            continue
        reuse_entries[target_id] = {
            "source_parent_task_id": parent_id,
            "source_checkpoint_dir": str(checkpoint_dir),
            "source_learning_summary_path": str(summary_path),
            "learning_task_ids": list(row.get("learning_task_ids") or []),
            "shared_between_icl_and_skill": bool(row.get("shared_between_icl_and_skill", True)),
        }
    write_json(bundle / "input/learning_plan_reuse.json", {
        "schema_version": "tongbench_learning_plan_reuse_v1",
        "source_learning_plan": str(learning_plan_path),
        "targets": [
            {
                "target_task_id": target_id,
                "learning_task_ids": list(entry.get("learning_task_ids") or []),
                "icl_learning_task_ids": list(entry.get("learning_task_ids") or []),
                "skill_learning_task_ids": list(entry.get("learning_task_ids") or []),
                "learning_group_id": str(entry.get("source_parent_task_id") or target_id),
                "shared_learning_group_id": str(entry.get("source_parent_task_id") or target_id),
                "shared_between_icl_and_skill": True,
                "reuse_source_parent_task_id": str(entry.get("source_parent_task_id") or ""),
            }
            for target_id, entry in sorted(reuse_entries.items())
        ],
    })
    write_json(bundle / "input/learning_reuse_manifest.json", {
        "schema_version": "tongbench_learning_reuse_manifest_v1",
        "backend": "hermesagent",
        "model": "qwen3.7-plus",
        "source_learning_plan": str(learning_plan_path),
        "source_checkpoint_root": str(checkpoint_root),
        "entries": reuse_entries,
        "missing_entries": missing_reuse,
    })
    write_json(bundle / "output/selection_summary.json", {
        "ok": True,
        "parent_count": len(parents),
        "scene_ids": sorted(scenes),
        "selected_l1_count": len(selected_l1),
        "selected_composite_count": len(composites),
        "learning_reuse_entry_count": len(reuse_entries),
        "learning_reuse_missing_count": len(missing_reuse),
        "counts_by_level": {x: len(selected[x]) for x in LEVELS},
    })
    print(json.dumps({"ok": True, "bundle_root": str(bundle), "selected_l1_count": len(selected_l1), "selected_composite_count": len(composites), "counts_by_level": {x: len(selected[x]) for x in LEVELS}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
