from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LEVELS = ("L1", "L2")


@dataclass
class TaskRecord:
    root: Path
    task_id: str
    level: str
    score: float | None
    success_score: float | None
    progress_score: float | None
    efficiency_score: float | None
    path_quality_score: float | None
    success: bool | None
    output_dir: str
    status: str
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


def _task_level_from_id(task_id: str) -> str:
    lowered = task_id.lower()
    if "_l1_" in lowered or lowered.startswith("task_l1_"):
        return "L1"
    if "_l2_" in lowered or lowered.startswith("task_l2_"):
        return "L2"
    if "_l3_" in lowered or lowered.startswith("task_l3_"):
        return "L3"
    if lowered.startswith("task_"):
        parts = lowered.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return "L2"
    return ""


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


def _record_from_item(root: Path, item: dict[str, Any], source: Path) -> TaskRecord | None:
    task_id = str(item.get("task_id") or "").strip()
    output_dir = str(item.get("output_dir") or "")
    score_obj = item.get("score")
    if not isinstance(score_obj, dict) or not score_obj:
        score_obj = _score_from_output_dir(output_dir) or {}
    if not task_id:
        task_id = str(score_obj.get("task_id") or "").strip()
    if not task_id:
        return None
    level = str(item.get("level") or "").strip() or _task_level_from_id(task_id)
    return TaskRecord(
        root=root,
        task_id=task_id,
        level=level,
        score=_extract_score(score_obj),
        success_score=_extract_metric(score_obj, "success_score"),
        progress_score=_extract_first_metric(
            score_obj,
            ("progress_score", "node_progress_for_score", "node_progress", "predicate_progress"),
        ),
        efficiency_score=_extract_metric(score_obj, "efficiency_score"),
        path_quality_score=_extract_metric(score_obj, "path_quality_score"),
        success=_extract_success(score_obj),
        output_dir=output_dir,
        status=str(item.get("status") or ""),
        source=str(source),
    )


def _records_from_summary(root: Path, summary_path: Path) -> list[TaskRecord]:
    try:
        summary = _read_json(summary_path)
    except Exception:
        return []
    if not isinstance(summary, dict):
        return []
    items = summary.get("results") or summary.get("tasks")
    if not isinstance(items, list):
        return []
    records: list[TaskRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record = _record_from_item(root, item, summary_path)
        if record is not None:
            records.append(record)
    return records


def _records_from_task_runs(root: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    task_runs = root / "task_runs"
    if not task_runs.exists():
        return records
    for score_path in sorted(task_runs.glob("*/score.json")):
        task_dir = score_path.parent
        try:
            score_obj = _read_json(score_path)
        except Exception:
            score_obj = {}
        if not isinstance(score_obj, dict):
            score_obj = {}
        task_id = str(score_obj.get("task_id") or task_dir.name)
        records.append(
            TaskRecord(
                root=root,
                task_id=task_id,
                level=_task_level_from_id(task_id),
                score=_extract_score(score_obj),
                success_score=_extract_metric(score_obj, "success_score"),
                progress_score=_extract_first_metric(
                    score_obj,
                    ("progress_score", "node_progress_for_score", "node_progress", "predicate_progress"),
                ),
                efficiency_score=_extract_metric(score_obj, "efficiency_score"),
                path_quality_score=_extract_metric(score_obj, "path_quality_score"),
                success=_extract_success(score_obj),
                output_dir=str(task_dir),
                status="",
                source=str(score_path),
            )
        )
    return records


def collect_records(root: Path) -> list[TaskRecord]:
    root = root.resolve()
    for summary_name in ("single_task_summary.json", "single_task_plan.json"):
        summary_path = root / summary_name
        if summary_path.exists():
            records = _records_from_summary(root, summary_path)
            if records:
                return records
    return _records_from_task_runs(root)


def _backend_label(root: Path) -> str:
    name = root.name.strip()
    return name or str(root)


def summarize(records: list[TaskRecord], levels: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in levels:
        level_records = [record for record in records if record.level == level]
        scored = [record for record in level_records if record.score is not None]
        success_known = [record for record in level_records if record.success is not None]
        success_count = sum(1 for record in success_known if record.success)
        task_count = len(level_records)
        average_score = (
            sum(float(record.score) for record in scored) / len(scored)
            if scored
            else None
        )
        average_score_missing_zero = (
            sum(float(record.score) for record in scored) / task_count
            if task_count
            else None
        )

        def average_metric(name: str) -> float | None:
            values = [
                float(value)
                for record in level_records
                for value in (getattr(record, name),)
                if value is not None
            ]
            return sum(values) / len(values) if values else None

        rows.append(
            {
                "level": level,
                "task_count": task_count,
                "scored_count": len(scored),
                "missing_score_count": task_count - len(scored),
                "success_known_count": len(success_known),
                "success_count": success_count,
                "success_rate": success_count / task_count if task_count else None,
                "observed_success_rate": success_count / len(success_known) if success_known else None,
                "average_score": average_score,
                "average_overall": average_score,
                "average_score_missing_zero": average_score_missing_zero,
                "average_success_score": average_metric("success_score"),
                "average_progress_score": average_metric("progress_score"),
                "average_efficiency_score": average_metric("efficiency_score"),
                "average_path_quality_score": average_metric("path_quality_score"),
            }
        )
    return rows


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "backend",
        "level",
        "tasks",
        "scored",
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
                str(row["level"]),
                str(row["task_count"]),
                str(row["scored_count"]),
                str(row["missing_score_count"]),
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
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    print("  ".join(headers[index].ljust(widths[index]) for index in range(len(headers))))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in table_rows:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(headers))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "root",
        "level",
        "task_count",
        "scored_count",
        "missing_score_count",
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize independent TongSIM single-task results by L1/L2 level."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="One or more single-task output roots, e.g. .../outputs/cc_single .../outputs/codex_single.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Levels to summarize. Defaults to L1 L2.",
    )
    parser.add_argument("--csv-out", help="Optional path to write the summary CSV.")
    parser.add_argument("--json-out", help="Optional path to write detailed JSON.")
    args = parser.parse_args()

    levels = tuple(str(level).upper() for level in args.levels)
    all_rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    for raw_root in args.roots:
        root = Path(raw_root).expanduser().resolve()
        records = collect_records(root)
        backend = _backend_label(root)
        rows = summarize(records, levels)
        for row in rows:
            row["backend"] = backend
            row["root"] = str(root)
        all_rows.extend(rows)
        detailed[backend] = {
            "root": str(root),
            "summary": rows,
            "records": [{**record.__dict__, "root": str(record.root)} for record in records],
        }

    print_table(all_rows)
    if args.csv_out:
        write_csv(Path(args.csv_out), all_rows)
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(detailed, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
