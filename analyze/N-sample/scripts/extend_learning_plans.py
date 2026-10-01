#!/usr/bin/env python3
"""Extend an existing N-sample prefix from k=4 to k=5 and k=6."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_n_sample_inputs import (  # noqa: E402
    choose_extra_learning_task,
    read_json,
    task_id,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=6)
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    if args.max_k < 5:
        raise ValueError("--max-k must be at least 5")
    base_plan_path = input_root / "learning_plan_k4.json"
    base_plan = read_json(base_plan_path)
    l2_items = read_json(input_root / "ready_l2_tasks_with_images.json")
    l4_items = read_json(input_root / "ready_l4_tasks_with_images.json")
    if not isinstance(l2_items, list) or not isinstance(l4_items, list):
        raise ValueError("Ready task inputs must be JSON lists")
    targets_by_id = {task_id(item): item for item in l4_items if isinstance(item, dict)}

    sequences: dict[str, list[str]] = {}
    additions: dict[str, list[dict[str, Any]]] = {}
    usage: Counter[str] = Counter()
    for row in base_plan.get("targets", []):
        target = str(row.get("target_task_id") or "").strip()
        prefix = [str(value).strip() for value in row.get("learning_task_ids", []) if str(value).strip()]
        if target not in targets_by_id:
            raise KeyError(f"Unknown L4 target in k=4 plan: {target}")
        if len(prefix) != 4 or len(set(prefix)) != 4:
            raise ValueError(f"Expected four unique L2 tasks for {target}, got {prefix}")
        sequences[target] = prefix
        usage.update(prefix)

    for target, sequence in sequences.items():
        target_additions: list[dict[str, Any]] = []
        while len(sequence) < args.max_k:
            selected, reason = choose_extra_learning_task(
                target=targets_by_id[target],
                selected_ids=sequence,
                candidates=l2_items,
                usage=usage,
            )
            sequence.append(task_id(selected))
            target_additions.append({"slot": len(sequence), **reason})
        additions[target] = target_additions

    base_targets = {
        str(row.get("target_task_id") or "").strip(): row
        for row in base_plan.get("targets", [])
        if isinstance(row, dict)
    }
    for k in range(5, args.max_k + 1):
        plan = deepcopy(base_plan)
        plan["learning_count_per_target"] = k
        plan["n_sample_axis"] = list(range(args.max_k + 1))
        plan["selection_note"] = (
            "k=0..4 are preserved from the confirmed formal plan; "
            "k=5 and k=6 append deterministic related L2 tasks to the same per-target prefix."
        )
        targets: list[dict[str, Any]] = []
        for target in sequences:
            row = deepcopy(base_targets[target])
            selected_ids = list(sequences[target][:k])
            row["learning_task_ids"] = selected_ids
            row["icl_learning_task_ids"] = list(selected_ids)
            row["skill_learning_task_ids"] = list(selected_ids)
            row["learning_count"] = k
            prefix = deepcopy(row.get("n_sample_prefix") or {})
            prefix["added_after_k4"] = [
                record for record in additions[target] if int(record["slot"]) <= k
            ]
            prefix["prefix_property"] = True
            row["n_sample_prefix"] = prefix
            targets.append(row)
        plan["targets"] = targets
        plan["target_count"] = len(targets)
        write_json(input_root / f"learning_plan_k{k}.json", plan)

    validation = {
        "schema_version": "tongbench_n_sample_extension_validation_v1",
        "source_plan": str(base_plan_path),
        "target_count": len(sequences),
        "axis": list(range(args.max_k + 1)),
        "strict_prefix_property": all(
            len(sequence) == args.max_k and len(set(sequence)) == args.max_k
            for sequence in sequences.values()
        ),
        "added_learning_task_ids": {
            target: sequence[4:]
            for target, sequence in sequences.items()
        },
    }
    write_json(input_root / "learning_plan_extension_validation.json", validation)
    print(json.dumps({"ok": True, **validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
