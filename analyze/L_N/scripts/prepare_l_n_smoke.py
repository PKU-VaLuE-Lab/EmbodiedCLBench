#!/usr/bin/env python3
"""Prepare one real child-scene smoke bundle for the L_N analysis.

The script only selects and validates compositions. It does not invent action
semantics: every L1 source is loaded from the existing scene inventory and
merged with TongBench's shared compositional merger.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate(task_id: str, level: str, source_ids: list[str], description: str, rationale: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "level": level,
        "height": "child",
        "scene_id": "001",
        "ability_categories": [],
        "source_l1_task_ids": source_ids,
        "description_original_decomposed": description,
        "description_highlevel_en_draft": description,
        "description_highlevel_hard": "A coherent household situation is unfolding. "
        "Complete the connected set of actions while keeping the room usable.",
        "rationale_zh": rationale,
        "difficulty_strategy_zh": "严格控制 L1 数量；使用真实同房间动作，并保留状态图中的自然前置条件。",
        "image_reuse_risk": "Smoke 先验证图搜索和状态图像复用；若状态缺图，任务不进入 API/EVO。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    external_root = args.external_root.resolve()
    out = args.output_root.resolve()
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    scene = "001"
    taskgen_root = external_root / "important_results_taskgen/8.24_l2_l4_scaleup_v1"
    eval_root = external_root / "important_results_eval/8.24_l2_l4_scaleup_v1"
    source_scene_root = external_root / "important_results/For_yanan/final_scene_results_two_heights_320x180/child/scenes/001"
    initial_image = source_scene_root / "task_L1_066/output/task_L1_066_unique_states/000_state_0000/state_0000_selected_orbit_v2_r_0p67_a_0_rgb.jpg"
    inventory_path = taskgen_root / "scenes/001/l1_inventory.json"
    existing_tasks_path = taskgen_root / "output/scene_001/tasks_compositional.json"
    n_sample_l4_path = repo_root / "analyze/N-sample/input/subset_10pct_child/ready_l4_tasks_with_images.json"
    # The fixed learning condition uses the same two L2 demonstrations for
    # every complexity level.  k1 is a one-shot prefix; k2 is the confirmed
    # two-example condition used by this analysis.
    n_sample_plan_path = repo_root / "analyze/N-sample/input/subset_10pct_child/learning_plan_k2.json"

    inventory_payload = read_json(inventory_path)
    inventory_by_id = {str(item["task_id"]): item for item in inventory_payload["tasks"]}
    existing_tasks = read_json(existing_tasks_path)
    parent = next(item for item in existing_tasks if item.get("task_id") == "task_L4_HL_v5_child_0002")
    base_ids = list(parent["component_sources"]["mapped_source_l1_task_ids"])
    removed_id = "task_L1_066"
    l3_ids = [item for item in base_ids if item != removed_id]
    l5_ids = [*base_ids, "task_L1_015"]
    l6_ids = [*base_ids, "task_L1_015", "task_L1_114"]
    all_ids = list(dict.fromkeys(l6_ids))
    missing = [task_id for task_id in all_ids if task_id not in inventory_by_id]
    if missing:
        raise KeyError(f"Missing L1 inventory records: {missing}")
    if not initial_image.is_file():
        raise FileNotFoundError(f"Missing initial child image: {initial_image}")

    descriptions = {
        task_id: str(inventory_by_id[task_id].get("description") or "")
        for task_id in all_ids
    }
    join = lambda ids: "; then ".join(descriptions[item] for item in ids)
    candidates = [
        candidate(
            "task_L3_LN_smoke_0001",
            "L3",
            l3_ids,
            join(l3_ids),
            "从现有 L4 去掉开门目标，保留 pick_up、place、close 三种不同的 L1 action，形成严格三步组合。",
        ),
        candidate(
            "task_L5_LN_smoke_0001",
            "L5",
            l5_ids,
            join(l5_ids),
            "在同一个 L4 组合上加入独立的 switch_on 目标，增加一个动作层级，同时不改变原四个目标的顺序。",
        ),
        candidate(
            "task_L6_LN_smoke_0001",
            "L6",
            l6_ids,
            join(l6_ids),
            "在 L4 基础上加入 switch_on 与 switch_off 两个不同设备目标，覆盖六种 L1 action，作为最高复杂度 smoke。",
        ),
    ]

    input_root = out / "input"
    highlevel_dir = input_root / "highlevel"
    taskgen_output = input_root / "taskgen_output"
    source_l1_dir = input_root / "l1_sources"
    # Do not put a wrapper-level difficulty here: the adapter reads the
    # candidate-level L3/L5/L6 values when selecting the merger spec.
    write_json(highlevel_dir / "highlevel_candidates.json", {"candidates": candidates})
    write_json(highlevel_dir / "l1_inventory.json", inventory_payload)

    # Import the shared adapter after paths are resolved. This keeps this
    # analysis directory thin and makes the composition semantics identical.
    import sys
    sys.path.insert(0, str(repo_root / "private/taskgen/src"))
    from tongbench_taskgen.cli.build_highlevel_v2_smoke_input import (  # noqa: E402
        _build_composite_tasks,
        _load_l1_task_from_inventory,
        _prepare_l1_sources,
    )

    l1_tasks = {
        task_id: _load_l1_task_from_inventory(external_root, inventory_by_id[task_id])
        for task_id in all_ids
    }
    ready_l1 = _prepare_l1_sources(
        project_root=external_root,
        source_scene_root=source_scene_root,
        source_l1_dir=source_l1_dir,
        taskgen_output_dir=taskgen_output,
        initial_image=initial_image,
        l1_task_ids=all_ids,
        l1_tasks=l1_tasks,
        force=True,
    )
    if len(ready_l1) != len(all_ids):
        raise RuntimeError("Selected L1 sources are not fully imaged")
    composites = _build_composite_tasks(candidates=candidates, l1_tasks=l1_tasks)
    # Keep the existing parent L4 in the same smoke task root so that the
    # four complexity levels can use one API/eval input layout.
    output_tasks = [l1_tasks[task_id] for task_id in all_ids] + composites + [parent]
    write_json(taskgen_output / "tasks_compositional.json", output_tasks)
    existing_graph_dir = taskgen_root / "output/scene_001/graph"
    target_graph_dir = taskgen_output / "graph"
    for source in existing_graph_dir.glob(f"{parent['task_id']}*"):
        shutil.copy2(source, target_graph_dir / source.name)

    parent_learning = []
    if n_sample_plan_path.is_file():
        for record in read_json(n_sample_plan_path).get("targets", []):
            if record.get("target_task_id") == parent["task_id"]:
                parent_learning = list(record.get("learning_task_ids") or record.get("icl_learning_task_ids") or [])
                break
    if len(parent_learning) != 2:
        raise ValueError(f"Expected two L2 learning tasks for {parent['task_id']}, got {parent_learning}")

    group_id = "ln_smoke_scene001_parent0002_learning"
    target_records = []
    for item in [*candidates[:1], parent, *candidates[1:]]:
        task_id = str(item["task_id"])
        level = str(item.get("level") or item.get("difficulty_level") or "L4")
        target_records.append({
            "target_task_id": task_id,
            "scene_id": scene,
            "height": "child",
            "target_level": level,
            "learning_level": "L2",
            "learning_task_ids": parent_learning,
            "icl_learning_task_ids": parent_learning,
            "skill_learning_task_ids": parent_learning,
            "learning_group_id": group_id,
            "shared_between_icl_and_skill": True,
            "selection_rationale": "All complexity levels reuse the same ordered two L2 demonstrations so the learning data is fixed across L3-L6.",
        })
    write_json(input_root / "learning_plan.json", {
        "schema_version": "tongbench_l_n_learning_plan_v1",
        "learning_level": "L2",
        "target_levels": ["L3", "L4", "L5", "L6"],
        "learning_count_per_target": 2,
        "shared_between_icl_and_skill": True,
        "targets": target_records,
    })
    write_json(input_root / "design.json", {
        "schema_version": "tongbench_l_n_design_v1",
        "analysis": "L_N",
        "scene_id": scene,
        "height": "child",
        "smoke": True,
        "levels": {
            "L3": {"source_l1_count": 3, "max_atomic_states": 45, "task_ids": ["task_L3_LN_smoke_0001"]},
            "L4": {"source_l1_count": 4, "reused_task_ids": [parent["task_id"]], "existing_graph_reused": True},
            "L5": {"source_l1_count": 5, "max_atomic_states": 75, "task_ids": ["task_L5_LN_smoke_0001"]},
            "L6": {"source_l1_count": 6, "max_atomic_states": 90, "task_ids": ["task_L6_LN_smoke_0001"]},
        },
        "parent_l4_task_id": parent["task_id"],
        "parent_l4_source_l1_task_ids": base_ids,
        "removed_for_l3": removed_id,
        "added_for_l5": ["task_L1_015"],
        "added_for_l6": ["task_L1_015", "task_L1_114"],
        "learning_group_id": group_id,
        "learning_task_ids": parent_learning,
        "candidate_task_ids": [item["task_id"] for item in candidates],
        "notes": "This is the low-concurrency smoke bundle. Full 12-per-level scale-up is intentionally not run here.",
    })
    write_json(out / "source_manifest.json", {
        "external_root": str(external_root),
        "taskgen_root": str(taskgen_root),
        "eval_root": str(eval_root),
        "source_scene_root": str(source_scene_root),
        "initial_image": str(initial_image),
        "parent_task_id": parent["task_id"],
        "n_sample_l4": str(n_sample_l4_path),
    })
    print(json.dumps({"ok": True, "output_root": str(out), "candidate_task_ids": [item["task_id"] for item in candidates], "learning_task_ids": parent_learning}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
