from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .runner import create_skill_memory, run_tongsim_task


def _load_sequence(sequence_file: Path) -> dict[str, Any]:
    return json.loads(Path(sequence_file).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_tongsim_sequence(
    sequence_file: Path,
    output_dir: Path,
    mode: str,
    agent_command: list[str] | str | None,
    max_steps: int = 30,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    sequence = _load_sequence(sequence_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_path = output_dir / "skill_memory.md"
    if memory_path.exists():
        memory_path.unlink()

    results: list[dict[str, Any]] = []
    for index, item in enumerate(sequence.get("tasks", []), start=1):
        task_dir = Path(item["task_dir"])
        task_output_dir = output_dir / f"{index:02d}_{task_dir.name}"
        memory_file = memory_path if mode == "memory" and memory_path.exists() else None
        result = run_tongsim_task(
            task_dir=task_dir,
            output_dir=task_output_dir,
            agent_command=agent_command,
            max_steps=max_steps,
            timeout_sec=timeout_sec,
            memory_file=memory_file,
        )
        result["level"] = item.get("level", "")
        results.append(result)
        if mode == "memory":
            existing_memory = memory_path.read_text(encoding="utf-8") if memory_path.exists() else None
            memory_text = create_skill_memory(task_dir, task_output_dir, existing_memory=existing_memory)
            memory_path.write_text(memory_text, encoding="utf-8")
            workspace_memory = task_output_dir / "workspace" / "skill_memory.md"
            if workspace_memory.exists():
                shutil.copy2(workspace_memory, task_output_dir / "task_output" / "skill_memory.md")

    avg_score = (
        sum(result["score"]["overall_score"] for result in results) / len(results)
        if results
        else 0.0
    )
    summary = {
        "sequence_id": sequence.get("sequence_id", Path(sequence_file).stem),
        "mode": mode,
        "task_count": len(results),
        "avg_score": avg_score,
        "results": [
            {
                "task_id": result["task_id"],
                "level": result.get("level", ""),
                "overall_score": result["score"]["overall_score"],
                "reached_goal": result["grade"]["reached_goal"],
                "efficiency_score": result["grade"]["efficiency_score"],
                "output_dir": result["output_dir"],
            }
            for result in results
        ],
    }
    _write_json(output_dir / "sequence_summary.json", summary)
    return summary


def compare_memory_vs_no_memory(
    sequence_file: Path,
    output_dir: Path,
    agent_command: list[str] | str | None,
    max_steps: int = 30,
    timeout_sec: int = 600,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    no_memory_dir = output_dir / "no_memory"
    memory_dir = output_dir / "memory"
    no_memory = run_tongsim_sequence(
        sequence_file=sequence_file,
        output_dir=no_memory_dir,
        mode="no_memory",
        agent_command=agent_command,
        max_steps=max_steps,
        timeout_sec=timeout_sec,
    )
    with_memory = run_tongsim_sequence(
        sequence_file=sequence_file,
        output_dir=memory_dir,
        mode="memory",
        agent_command=agent_command,
        max_steps=max_steps,
        timeout_sec=timeout_sec,
    )

    no_memory_results = {item["task_id"]: item for item in no_memory["results"]}
    with_memory_results = {item["task_id"]: item for item in with_memory["results"]}
    levels: dict[str, list[float]] = defaultdict(list)
    success_no = 0
    success_yes = 0
    eff_no = 0.0
    eff_yes = 0.0
    details_by_task: list[dict[str, Any]] = []
    for task_id, with_item in with_memory_results.items():
        no_item = no_memory_results.get(task_id, {})
        gain = with_item.get("overall_score", 0.0) - no_item.get("overall_score", 0.0)
        levels[with_item.get("level", "")].append(gain)
        success_no += int(bool(no_item.get("reached_goal", False)))
        success_yes += int(bool(with_item.get("reached_goal", False)))
        eff_no += float(no_item.get("efficiency_score", 0.0))
        eff_yes += float(with_item.get("efficiency_score", 0.0))
        details_by_task.append(
            {
                "task_id": task_id,
                "level": with_item.get("level", ""),
                "score_no_memory": no_item.get("overall_score", 0.0),
                "score_with_memory": with_item.get("overall_score", 0.0),
                "gain": gain,
                "success_no_memory": no_item.get("reached_goal", False),
                "success_with_memory": with_item.get("reached_goal", False),
            }
        )

    task_count = max(1, len(with_memory_results))
    comparison = {
        "sequence_id": _load_sequence(sequence_file).get("sequence_id", Path(sequence_file).stem),
        "avg_score_no_memory": no_memory.get("avg_score", 0.0),
        "avg_score_with_memory": with_memory.get("avg_score", 0.0),
        "self_evolution_gain": with_memory.get("avg_score", 0.0) - no_memory.get("avg_score", 0.0),
        "per_level_gain": {
            level: (sum(values) / len(values) if values else 0.0)
            for level, values in levels.items()
        },
        "success_rate_gain": (success_yes / task_count) - (success_no / task_count),
        "efficiency_gain": (eff_yes / task_count) - (eff_no / task_count),
        "details_by_task": details_by_task,
    }
    _write_json(output_dir / "sequence_score.json", comparison)
    return comparison
