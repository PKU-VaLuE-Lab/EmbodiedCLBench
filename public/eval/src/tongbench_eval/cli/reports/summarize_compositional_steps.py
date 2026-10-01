from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ORDINAL_LABELS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
}


@dataclass
class StepRecord:
    root: Path
    composite_task_id: str
    step_index: int
    task_id: str
    level: str
    score: float | None
    success_score: float | None
    progress_score: float | None
    efficiency_score: float | None
    path_quality_score: float | None
    success: bool | None
    output_dir: str
    source: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_score(score_obj: Any) -> float | None:
    if not isinstance(score_obj, dict):
        return None
    for key in ("score", "overall_score", "overall"):
        score = _as_float(score_obj.get(key))
        if score is not None:
            return score
    details = score_obj.get("details")
    if isinstance(details, dict):
        for key in ("score", "overall_score", "overall"):
            score = _as_float(details.get(key))
            if score is not None:
                return score
    return None


def _extract_metric(score_obj: Any, key: str) -> float | None:
    if not isinstance(score_obj, dict):
        return None
    for container in (score_obj, score_obj.get("details"), score_obj.get("metrics")):
        if not isinstance(container, dict):
            continue
        value = _as_float(container.get(key))
        if value is not None:
            return value
    return None


def _extract_first_metric(score_obj: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _extract_metric(score_obj, key)
        if value is not None:
            return value
    return None


def _extract_success(score_obj: Any) -> bool | None:
    if not isinstance(score_obj, dict):
        return None
    details = score_obj.get("details")
    if isinstance(details, dict):
        reached_goal = details.get("reached_goal")
        if isinstance(reached_goal, bool):
            return reached_goal
        success_score = _as_float(details.get("success_score"))
        if success_score is not None:
            return success_score >= 1.0
    reached_goal = score_obj.get("reached_goal")
    if isinstance(reached_goal, bool):
        return reached_goal
    success_score = _as_float(score_obj.get("success_score"))
    if success_score is not None:
        return success_score >= 1.0
    return None


def _score_from_output_dir(output_dir: Any) -> dict[str, Any] | None:
    if not output_dir:
        return None
    score_path = Path(str(output_dir)) / "score.json"
    if not score_path.exists():
        return None
    try:
        score = _read_json(score_path)
    except Exception:
        return None
    return score if isinstance(score, dict) else None


def _records_from_sequence_summary(root: Path, summary_path: Path) -> list[StepRecord]:
    try:
        summary = _read_json(summary_path)
    except Exception:
        return []
    if not isinstance(summary, dict):
        return []
    composite_task_id = str(summary.get("task_id") or summary_path.parent.name)
    steps = summary.get("steps")
    if not isinstance(steps, list):
        return []

    records: list[StepRecord] = []
    for fallback_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("index") or fallback_index)
        score_obj = step.get("score")
        if not isinstance(score_obj, dict) or not score_obj:
            score_obj = _score_from_output_dir(step.get("output_dir")) or {}
        records.append(
            StepRecord(
                root=root,
                composite_task_id=composite_task_id,
                step_index=step_index,
                task_id=str(step.get("task_id") or score_obj.get("task_id") or ""),
                level=str(step.get("level") or ""),
                score=_extract_score(score_obj),
                success_score=_extract_metric(score_obj, "success_score"),
                progress_score=_extract_first_metric(
                    score_obj,
                    ("progress_score", "node_progress_for_score", "node_progress", "predicate_progress"),
                ),
                efficiency_score=_extract_metric(score_obj, "efficiency_score"),
                path_quality_score=_extract_metric(score_obj, "path_quality_score"),
                success=_extract_success(score_obj),
                output_dir=str(step.get("output_dir") or ""),
                source=str(summary_path),
            )
        )
    return records


def _records_from_root_summary(root: Path, summary_path: Path) -> list[StepRecord]:
    try:
        summary = _read_json(summary_path)
    except Exception:
        return []
    if not isinstance(summary, dict):
        return []
    tasks = summary.get("tasks")
    if not isinstance(tasks, list):
        return []

    records: list[StepRecord] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        steps = task.get("steps")
        if not isinstance(steps, list):
            task_id = str(task.get("task_id") or "")
            sequence_path = root / task_id / "sequence_summary.json"
            if sequence_path.exists():
                records.extend(_records_from_sequence_summary(root, sequence_path))
            continue
        temp_path = root / str(task.get("task_id") or "") / "sequence_summary.json"
        records.extend(_records_from_sequence_summary_like(root, task, temp_path))
    return records


def _records_from_sequence_summary_like(root: Path, summary: dict[str, Any], source_path: Path) -> list[StepRecord]:
    composite_task_id = str(summary.get("task_id") or source_path.parent.name)
    steps = summary.get("steps")
    if not isinstance(steps, list):
        return []
    records: list[StepRecord] = []
    for fallback_index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        step_index = int(step.get("index") or fallback_index)
        score_obj = step.get("score")
        if not isinstance(score_obj, dict) or not score_obj:
            score_obj = _score_from_output_dir(step.get("output_dir")) or {}
        records.append(
            StepRecord(
                root=root,
                composite_task_id=composite_task_id,
                step_index=step_index,
                task_id=str(step.get("task_id") or score_obj.get("task_id") or ""),
                level=str(step.get("level") or ""),
                score=_extract_score(score_obj),
                success_score=_extract_metric(score_obj, "success_score"),
                progress_score=_extract_first_metric(
                    score_obj,
                    ("progress_score", "node_progress_for_score", "node_progress", "predicate_progress"),
                ),
                efficiency_score=_extract_metric(score_obj, "efficiency_score"),
                path_quality_score=_extract_metric(score_obj, "path_quality_score"),
                success=_extract_success(score_obj),
                output_dir=str(step.get("output_dir") or ""),
                source=str(source_path),
            )
        )
    return records


def collect_records(root: Path) -> list[StepRecord]:
    root = root.resolve()
    root_summary = root / "summary.json"
    if root_summary.exists():
        records = _records_from_root_summary(root, root_summary)
        if records:
            return records

    records: list[StepRecord] = []
    for summary_path in sorted(root.glob("*/sequence_summary.json")):
        records.extend(_records_from_sequence_summary(root, summary_path))
    return records


def _backend_label(root: Path) -> str:
    name = root.name.strip()
    return name or str(root)


def collect_composite_ids(root: Path, records: list[StepRecord]) -> set[str]:
    composite_ids = {record.composite_task_id for record in records if record.composite_task_id}
    root_summary = root / "summary.json"
    if root_summary.exists():
        try:
            summary = _read_json(root_summary)
        except Exception:
            summary = {}
        tasks = summary.get("tasks") if isinstance(summary, dict) else None
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict) and str(task.get("task_id") or "").strip():
                    composite_ids.add(str(task.get("task_id")).strip())
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if (child / "sequence_summary.json").exists() or (child / "dialogue_plan.json").exists():
            composite_ids.add(child.name)
    return composite_ids


def summarize(
    records: list[StepRecord],
    composite_ids: set[str],
    expected_steps: int | None = None,
) -> list[dict[str, Any]]:
    if expected_steps is None:
        max_observed_step = max((record.step_index for record in records), default=0)
        expected_steps = max(3, max_observed_step)
    by_step: dict[int, list[StepRecord]] = {index: [] for index in range(1, expected_steps + 1)}
    if not composite_ids:
        composite_ids = {record.composite_task_id for record in records if record.composite_task_id}
    for record in records:
        if 1 <= record.step_index <= expected_steps:
            by_step.setdefault(record.step_index, []).append(record)

    rows: list[dict[str, Any]] = []
    total_tasks = len(composite_ids)
    for step_index in range(1, expected_steps + 1):
        step_records = by_step.get(step_index, [])
        scored = [record for record in step_records if record.score is not None]
        success_known = [record for record in step_records if record.success is not None]
        success_count = sum(1 for record in success_known if record.success)
        avg_score = (
            sum(float(record.score) for record in scored) / len(scored)
            if scored
            else None
        )
        total_score_with_missing_zero = sum(float(record.score) for record in scored)
        average_score_missing_zero = (
            total_score_with_missing_zero / total_tasks
            if total_tasks
            else None
        )
        success_rate = success_count / total_tasks if total_tasks else None
        observed_success_rate = success_count / len(success_known) if success_known else None

        def average_metric(name: str) -> float | None:
            values = [
                float(value)
                for record in step_records
                for value in (getattr(record, name),)
                if value is not None
            ]
            return sum(values) / len(values) if values else None

        rows.append(
            {
                "step_index": step_index,
                "step_label": _step_label(step_index, step_records, expected_steps),
                "task_count": total_tasks,
                "present_count": len(step_records),
                "missing_count": max(total_tasks - len(step_records), 0),
                "scored_count": len(scored),
                "success_known_count": len(success_known),
                "success_count": success_count,
                "success_rate": success_rate,
                "observed_success_rate": observed_success_rate,
                "average_score": avg_score,
                "average_overall": avg_score,
                "average_score_missing_zero": average_score_missing_zero,
                "average_success_score": average_metric("success_score"),
                "average_progress_score": average_metric("progress_score"),
                "average_efficiency_score": average_metric("efficiency_score"),
                "average_path_quality_score": average_metric("path_quality_score"),
            }
        )
    return rows


def _step_label(step_index: int, records: list[StepRecord], expected_steps: int) -> str:
    ordinal = ORDINAL_LABELS.get(step_index, f"step_{step_index}")
    levels = sorted({record.level for record in records if record.level})
    if len(levels) == 1:
        return f"{ordinal}_{levels[0]}"
    if levels:
        return f"{ordinal}_mixed"
    fallback_levels = {
        3: {1: "L1", 2: "L1", 3: "L2"},
        4: {1: "L1", 2: "L1", 3: "L1", 4: "L3"},
    }.get(expected_steps, {})
    if step_index in fallback_levels:
        return f"{ordinal}_{fallback_levels[step_index]}"
    return ordinal


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "root",
        "step_index",
        "step_label",
        "task_count",
        "present_count",
        "missing_count",
        "scored_count",
        "success_known_count",
        "success_count",
        "success_rate",
        "observed_success_rate",
        "average_score",
        "average_overall",
        "average_score_missing_zero",
        "average_success_score",
        "average_progress_score",
        "average_efficiency_score",
        "average_path_quality_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "backend",
        "step",
        "tasks",
        "present",
        "missing",
        "success",
        "success_rate",
        "observed_success_rate",
        "avg_overall",
        "avg_success",
        "avg_progress",
        "avg_efficiency",
        "avg_path_quality",
        "avg_score_missing0",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["backend"]),
                str(row["step_label"]),
                str(row["task_count"]),
                str(row["present_count"]),
                str(row["missing_count"]),
                f'{row["success_count"]}/{row["success_known_count"]}',
                format_float(row["success_rate"]),
                format_float(row["observed_success_rate"]),
                format_float(row["average_overall"]),
                format_float(row["average_success_score"]),
                format_float(row["average_progress_score"]),
                format_float(row["average_efficiency_score"]),
                format_float(row["average_path_quality_score"]),
                format_float(row["average_score_missing_zero"]),
            ]
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize compositional TongSIM results by sequence position: "
            "first L1, second L1, third L2."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more output roots, e.g. .../outputs/cc .../outputs/codex.",
    )
    parser.add_argument("--csv-out", help="Optional path to write the summary CSV.")
    parser.add_argument("--json-out", help="Optional path to write detailed JSON.")
    parser.add_argument("--expected-steps", type=int, help="Expected number of dialogue steps. Defaults to the max observed step, at least 3.")
    args = parser.parse_args()

    all_summary_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for raw_root in args.roots:
        root = Path(raw_root).expanduser().resolve()
        records = collect_records(root)
        composite_ids = collect_composite_ids(root, records)
        backend = _backend_label(root)
        rows = summarize(records, composite_ids, args.expected_steps)
        for row in rows:
            row["backend"] = backend
            row["root"] = str(root)
        all_summary_rows.extend(rows)
        detailed[backend] = {
            "root": str(root),
            "composite_task_ids": sorted(composite_ids),
            "summary": rows,
            "records": [{**record.__dict__, "root": str(record.root)} for record in records],
        }

    print_table(all_summary_rows)
    if args.csv_out:
        write_csv(Path(args.csv_out), all_summary_rows)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(detailed, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
