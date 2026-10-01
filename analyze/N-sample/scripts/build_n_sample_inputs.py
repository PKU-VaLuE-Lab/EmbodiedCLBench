#!/usr/bin/env python3
"""Build the child-only N-sample input bundle without touching For_user."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_SOURCE_SUBSET = PROJECT_ROOT / "For_user/data/subsets/stage2_20pct"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analyze/N-sample/input/subset_10pct_child"
DEFAULT_SOURCE_RESULTS = PROJECT_ROOT / "For_user/data/input/merged"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def task_id(item: dict[str, Any]) -> str:
    return str(item.get("task_id") or "").strip()


def height(item: dict[str, Any]) -> str:
    return str(item.get("height") or "").strip().lower()


def categories(item: dict[str, Any]) -> set[str]:
    values = item.get("ability_categories")
    if not isinstance(values, list):
        values = [item.get("ability_category")]
    return {str(value).strip() for value in values if str(value or "").strip()}


def actions(item: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("action_combo_fine", "action_combo_coarse"):
        raw = item.get(key)
        if isinstance(raw, list):
            values.update(str(value).strip() for value in raw if str(value or "").strip())
    summaries = item.get("source_l1_summaries")
    if isinstance(summaries, list):
        for summary in summaries:
            if not isinstance(summary, dict):
                continue
            for key in ("semantic_action", "last_action_type"):
                value = str(summary.get(key) or "").strip()
                if value:
                    values.add(value)
    return values


def scene_id(item: dict[str, Any]) -> str:
    return str(item.get("scene_id") or "").strip()


def target_plan_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for target in plan.get("targets", []) or []:
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_task_id") or target.get("task_id") or "").strip()
        if target_id:
            result[target_id] = target
    return result


def choose_extra_learning_task(
    *,
    target: dict[str, Any],
    selected_ids: list[str],
    candidates: list[dict[str, Any]],
    usage: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    target_categories = categories(target)
    target_actions = actions(target)
    selected_items = [item for item in candidates if task_id(item) in selected_ids]
    covered_categories = set().union(*(categories(item) for item in selected_items))
    covered_actions = set().union(*(actions(item) for item in selected_items))
    target_scene = scene_id(target)

    available = [item for item in candidates if task_id(item) not in set(selected_ids)]
    if not available:
        raise ValueError(f"No unused child L2 candidate remains for {task_id(target)}")

    def rank(item: dict[str, Any]) -> tuple[Any, ...]:
        item_categories = categories(item)
        item_actions = actions(item)
        new_target_actions = len((item_actions & target_actions) - covered_actions)
        target_action_overlap = len(item_actions & target_actions)
        new_target_categories = len((item_categories & target_categories) - covered_categories)
        category_overlap = len(item_categories & target_categories)
        new_actions = len(item_actions - covered_actions)
        different_scene = int(scene_id(item) != target_scene)
        return (
            -new_target_actions,
            -target_action_overlap,
            -new_target_categories,
            -category_overlap,
            -new_actions,
            -different_scene,
            usage[task_id(item)],
            scene_id(item),
            task_id(item),
        )

    selected = min(available, key=rank)
    selected_categories = categories(selected)
    selected_actions = actions(selected)
    reason = {
        "selected_task_id": task_id(selected),
        "target_scene_id": target_scene,
        "selected_scene_id": scene_id(selected),
        "target_category_overlap": sorted(selected_categories & target_categories),
        "new_target_action_types": sorted((selected_actions & target_actions) - covered_actions),
        "new_categories_after_prefix": sorted((selected_categories & target_categories) - covered_categories),
        "new_action_types_after_prefix": sorted(selected_actions - covered_actions),
        "selection_policy": "target-category relevance, action coverage, prefix diversity, low reuse, deterministic id tie-break",
    }
    usage[task_id(selected)] += 1
    return selected, reason


def make_plan(
    *,
    source_plan: dict[str, Any],
    target_items: list[dict[str, Any]],
    count: int,
    sequences: dict[str, list[str]],
    selection_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    source_targets = target_plan_map(source_plan)
    targets: list[dict[str, Any]] = []
    for target_item in target_items:
        target_id = task_id(target_item)
        raw_target = source_targets.get(target_id)
        if raw_target is None:
            raise KeyError(f"Missing learning-plan target for {target_id}")
        selected_ids = list(sequences[target_id][:count])
        base_ids = list(sequences[target_id][:2])
        target = deepcopy(raw_target)
        target["learning_task_ids"] = selected_ids
        target["icl_learning_task_ids"] = list(selected_ids)
        target["skill_learning_task_ids"] = list(selected_ids)
        target["learning_count"] = count
        target["n_sample_prefix"] = {
            "base_k2_task_ids": base_ids,
            "added_task_ids": [record["selected_task_id"] for record in selection_records[target_id][: max(0, count - 2)]],
            "prefix_property": True,
        }
        targets.append(target)

    output = deepcopy(source_plan)
    output["schema_version"] = "tongbench_n_sample_learning_plan_v1"
    output["easy_level"] = "L2"
    output["hard_level"] = "L4"
    output["learning_count_per_target"] = count
    output["shared_between_modes"] = True
    output["same_room_between_learning_and_target_required"] = False
    output["n_sample_axis"] = [0, 1, 2, 3, 4]
    output["targets"] = targets
    output["target_count"] = len(targets)
    output["selection_note"] = (
        "k=1 is the first task from the existing k=2 prefix; k=2 preserves the existing plan; "
        "k=3 and k=4 append deterministic related child L2 tasks."
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build child-only N-sample input manifests.")
    parser.add_argument("--source-subset-root", type=Path, default=DEFAULT_SOURCE_SUBSET)
    parser.add_argument("--source-results-root", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--height", default="child", choices=("child", "adult"))
    parser.add_argument("--target-count", type=int, default=0)
    args = parser.parse_args()

    source_subset = args.source_subset_root.resolve()
    source_results = args.source_results_root.resolve()
    output_root = args.output_root.resolve()
    l2_items = read_json(source_subset / "ready_l2_tasks_with_images.json")
    l4_items = read_json(source_subset / "ready_l4_tasks_with_images.json")
    source_plan = read_json(source_subset / "learning_plan.json")
    if not isinstance(l2_items, list) or not isinstance(l4_items, list):
        raise ValueError("Ready task JSON files must contain lists")

    selected_height = args.height.strip().lower()
    child_l2 = [item for item in l2_items if isinstance(item, dict) and height(item) == selected_height]
    child_l4 = [item for item in l4_items if isinstance(item, dict) and height(item) == selected_height]
    if args.target_count:
        if args.target_count < 1:
            raise ValueError("--target-count must be positive when provided")
        if len(child_l4) < args.target_count:
            raise ValueError(f"Requested {args.target_count} {selected_height} L4 tasks, but source subset contains only {len(child_l4)}")
        child_l4 = child_l4[:args.target_count]
    if not child_l4:
        raise ValueError(f"No {selected_height} L4 tasks found in {source_subset}")
    if not child_l2:
        raise ValueError(f"No {selected_height} L2 tasks found in {source_subset}")

    child_l2_ids = {task_id(item) for item in child_l2}
    if len(child_l2_ids) != len(child_l2):
        raise ValueError("Child L2 task ids are not unique")
    child_l4_ids = [task_id(item) for item in child_l4]
    if len(set(child_l4_ids)) != len(child_l4_ids):
        raise ValueError("Child L4 task ids are not unique")

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "ready_l2_tasks_with_images.json", child_l2)
    write_json(output_root / "ready_l4_tasks_with_images.json", child_l4)

    usage: Counter[str] = Counter()
    selection_records: dict[str, list[dict[str, Any]]] = {}
    for target in target_plan_map(source_plan).values():
        if task_id(target) in set(child_l4_ids):
            for learning_id in target.get("learning_task_ids", []) or []:
                usage[str(learning_id)] += 1

    sequences: dict[str, list[str]] = {}
    for target_item in child_l4:
        target_id = task_id(target_item)
        raw_target = target_plan_map(source_plan).get(target_id)
        if raw_target is None:
            raise KeyError(f"Missing learning-plan target for {target_id}")
        base_ids = [str(value).strip() for value in raw_target.get("learning_task_ids", []) or []]
        if len(base_ids) != 2:
            raise ValueError(f"Expected exactly two base learning tasks for {target_id}, got {base_ids}")
        child_l2_ids = {task_id(item) for item in child_l2}
        if not all(value in child_l2_ids for value in base_ids):
            raise ValueError(f"Base learning task is absent from child L2 pool for {target_id}: {base_ids}")
        sequence = list(base_ids)
        additions: list[dict[str, Any]] = []
        while len(sequence) < 4:
            selected, reason = choose_extra_learning_task(
                target=target_item,
                selected_ids=sequence,
                candidates=child_l2,
                usage=usage,
            )
            sequence.append(task_id(selected))
            additions.append({"slot": len(sequence), **reason})
        sequences[target_id] = sequence
        selection_records[target_id] = additions

    for count in range(0, 5):
        if count == 0:
            plan = deepcopy(source_plan)
            targets = []
            source_targets = target_plan_map(source_plan)
            for target_item in child_l4:
                target_id = task_id(target_item)
                if target_id not in source_targets:
                    raise KeyError(f"Missing learning-plan target for {target_id}")
                target = deepcopy(source_targets[target_id])
                target["learning_task_ids"] = []
                target["icl_learning_task_ids"] = []
                target["skill_learning_task_ids"] = []
                target["learning_count"] = 0
                targets.append(target)
            plan["schema_version"] = "tongbench_n_sample_learning_plan_v1"
            plan["learning_count_per_target"] = 0
            plan["shared_between_modes"] = True
            plan["same_room_between_learning_and_target_required"] = False
            plan["n_sample_axis"] = [0, 1, 2, 3, 4]
            plan["targets"] = targets
            plan["target_count"] = len(targets)
            plan["selection_note"] = "k=0 is evaluated through the zero-shot runner; this file is metadata only."
        else:
            plan = make_plan(
                source_plan=source_plan,
                target_items=child_l4,
                count=count,
                sequences=sequences,
                selection_records=selection_records,
            )
        write_json(output_root / f"learning_plan_k{count}.json", plan)

    scene_counts = Counter(scene_id(item) for item in child_l4)
    selection_manifest = {
        "schema_version": "tongbench_n_sample_selection_v1",
        "height": selected_height,
        "source_subset": str(source_subset),
        "source_results_root": str(source_results),
        "target_level": "L4",
        "learning_level": "L2",
        "target_count": len(child_l4),
        "learning_pool_count": len(child_l2),
        "scene_counts": dict(sorted(scene_counts.items())),
        "target_task_ids": child_l4_ids,
        "learning_prefix_rule": "k=1..4 are prefixes of the same per-target sequence; k=2 is unchanged from the source plan.",
        "zero_shot_policy": "k=0 is the main-experiment zero-shot baseline and is not executed by this runner.",
        "selection_records_for_k3_k4": selection_records,
    }
    write_json(output_root / "sample_selection_manifest.json", selection_manifest)
    write_json(
        output_root / "source_manifest.json",
        {
            "source_subset_root": str(source_subset),
            "source_results_root": str(source_results),
            "choice_bank_jsonl": str(source_results / "option_bank_highlevel_v2_smoke.jsonl"),
            "images_are_referenced_not_copied": True,
            "ready_l2_source": str(source_subset / "ready_l2_tasks_with_images.json"),
            "ready_l4_source": str(source_subset / "ready_l4_tasks_with_images.json"),
            "learning_plan_source": str(source_subset / "learning_plan.json"),
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output_root": str(output_root),
                "child_l2_count": len(child_l2),
                "child_l4_count": len(child_l4),
                "scene_counts": dict(sorted(scene_counts.items())),
                "plans": [f"learning_plan_k{k}.json" for k in range(5)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
