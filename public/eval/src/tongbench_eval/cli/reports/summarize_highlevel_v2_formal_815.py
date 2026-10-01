from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODE_ROOTS = {
    "basic_l3": ("basic_l3", "hermesagent"),
    "language_bias_l3": ("language_bias_l3", "hermesagent"),
    "zero_shot_l6": ("zero_shot_l6", "hermesagent"),
    "in_context_l6": ("in_context_l6", "hermesagent"),
    "skill_l6": ("skill_l6", "hermesagent"),
}


@dataclass
class ScoreRecord:
    mode: str
    task_id: str
    level: str
    output_dir: str
    overall: float | None
    success_score: float | None
    progress_score: float | None
    efficiency_score: float | None
    path_quality_score: float | None
    invalid_penalty: float | None
    success: bool | None
    shortest_success_trace_length: int | None
    total_env_steps: int | None
    success_env_step: int | None
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


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metric(score: Any, *keys: str) -> float | None:
    if not isinstance(score, dict):
        return None
    containers = [score]
    for name in ("details", "metrics"):
        if isinstance(score.get(name), dict):
            containers.append(score[name])
    for container in containers:
        for key in keys:
            value = _as_float(container.get(key))
            if value is not None:
                return value
    return None


def _int_metric(score: Any, *keys: str) -> int | None:
    if not isinstance(score, dict):
        return None
    containers = [score]
    for name in ("details", "metrics"):
        if isinstance(score.get(name), dict):
            containers.append(score[name])
    for container in containers:
        for key in keys:
            value = _as_int(container.get(key))
            if value is not None:
                return value
    return None


def _success(score: Any) -> bool | None:
    if not isinstance(score, dict):
        return None
    for container in (score, score.get("details") if isinstance(score.get("details"), dict) else {}):
        reached_goal = container.get("reached_goal")
        if isinstance(reached_goal, bool):
            return reached_goal
        success_score = _as_float(container.get("success_score"))
        if success_score is not None:
            return success_score >= 1.0
    return None


def _trace_budget_stats(output_dir: str, score: Any, success: bool | None) -> tuple[int | None, int | None, int | None]:
    shortest = _int_metric(score, "shortest_success_trace_length")
    trace_path = Path(output_dir) / "task_output" / "trace.jsonl" if output_dir else Path()
    if not trace_path.is_file():
        return shortest, None, None
    entries: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            entries.append(item)
    total_steps = len(entries)
    success_step: int | None = None
    for index, item in enumerate(entries):
        if not bool(item.get("reached_goal", False)):
            continue
        step_index = _as_int(item.get("step_index"))
        success_step = (step_index + 1) if step_index is not None else index + 1
        break
    if success is True and success_step is None and total_steps:
        success_step = total_steps
    return shortest, total_steps, success_step


def _grade_from_output_dir(output_dir: Any) -> dict[str, Any]:
    if not output_dir:
        return {}
    path = Path(str(output_dir)) / "task_output" / "grade.json"
    if not path.exists():
        return {}
    try:
        grade = _read_json(path)
    except Exception:
        return {}
    return grade if isinstance(grade, dict) else {}


def _merge_score_and_grade(score: Any, output_dir: str) -> dict[str, Any]:
    grade = _grade_from_output_dir(output_dir)
    merged: dict[str, Any] = {}
    if grade:
        merged.update(grade)
    if isinstance(score, dict):
        merged.update(score)
        score_details = score.get("details") if isinstance(score.get("details"), dict) else {}
        merged["details"] = {**grade, **score_details}
    elif grade:
        merged["details"] = dict(grade)
    return merged


def _record_from_score(mode: str, task_id: str, level: str, output_dir: str, score: Any, source: Path) -> ScoreRecord:
    score = _merge_score_and_grade(score, output_dir)
    success = _success(score)
    shortest, total_steps, success_step = _trace_budget_stats(output_dir, score, success)
    return ScoreRecord(
        mode=mode,
        task_id=task_id,
        level=level,
        output_dir=output_dir,
        overall=_metric(score, "score", "overall_score", "overall"),
        success_score=_metric(score, "success_score"),
        progress_score=_metric(score, "progress_score", "node_progress_for_score", "node_progress", "predicate_progress"),
        efficiency_score=_metric(score, "efficiency_score"),
        path_quality_score=_metric(score, "path_quality_score"),
        invalid_penalty=_metric(score, "invalid_penalty"),
        success=success,
        shortest_success_trace_length=shortest,
        total_env_steps=total_steps,
        success_env_step=success_step,
        source=str(source),
    )


def _score_from_output_dir(output_dir: Any) -> dict[str, Any]:
    if not output_dir:
        return {}
    path = Path(str(output_dir)) / "score.json"
    if not path.exists():
        return {}
    try:
        score = _read_json(path)
    except Exception:
        return {}
    return score if isinstance(score, dict) else {}


def _task_level(task_id: str) -> str:
    lowered = task_id.lower()
    if "task_l3_" in lowered:
        return "L3"
    if "task_l6_" in lowered:
        return "L6"
    if "task_l1_" in lowered:
        return "L1"
    if "task_l2_" in lowered:
        return "L2"
    return ""


def _single_records(mode: str, root: Path) -> list[ScoreRecord]:
    summary_path = root / "single_task_summary.json"
    records: list[ScoreRecord] = []
    if summary_path.exists():
        payload = _read_json(summary_path)
        for item in payload.get("results", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or "").strip()
            output_dir = str(item.get("output_dir") or "")
            score = item.get("score") if isinstance(item.get("score"), dict) else _score_from_output_dir(output_dir)
            records.append(_record_from_score(mode, task_id, str(item.get("level") or _task_level(task_id)), output_dir, score, summary_path))
        return records
    for score_path in sorted((root / "task_runs").glob("*/score.json")):
        score = _read_json(score_path)
        task_id = str(score.get("task_id") or score_path.parent.name)
        records.append(_record_from_score(mode, task_id, _task_level(task_id), str(score_path.parent), score, score_path))
    return records


def _l6_step_record(mode: str, step: dict[str, Any], source: Path) -> ScoreRecord | None:
    task_id = str(step.get("task_id") or "").strip()
    level = str(step.get("level") or _task_level(task_id))
    if level != "L6":
        return None
    output_dir = str(step.get("output_dir") or "")
    score = step.get("score") if isinstance(step.get("score"), dict) else _score_from_output_dir(output_dir)
    return _record_from_score(mode, task_id, level, output_dir, score, source)


def _dialogue_records(mode: str, root: Path) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    summary_path = root / "summary.json"
    if summary_path.exists():
        payload = _read_json(summary_path)
        for item in payload.get("tasks", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            steps = item.get("steps")
            if not isinstance(steps, list):
                continue
            for step in reversed(steps):
                if isinstance(step, dict):
                    record = _l6_step_record(mode, step, summary_path)
                    if record is not None:
                        records.append(record)
                        break
        if records:
            return records
    for path in sorted(root.glob("*/sequence_summary.json")):
        payload = _read_json(path)
        steps = payload.get("steps") if isinstance(payload, dict) else []
        for step in reversed(steps if isinstance(steps, list) else []):
            if isinstance(step, dict):
                record = _l6_step_record(mode, step, path)
                if record is not None:
                    records.append(record)
                    break
    return records


def _skill_records(mode: str, root: Path) -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    summary_path = root / "summary.json"
    items: list[tuple[dict[str, Any], Path]] = []
    if summary_path.exists():
        payload = _read_json(summary_path)
        items.extend((item, summary_path) for item in payload.get("tasks", []) if isinstance(item, dict))
    for path in sorted(root.glob("*/skill_mode_summary.json")):
        try:
            payload = _read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict):
            items.append((payload, path))
    seen: set[str] = set()
    for item, source in items:
        task_id = str(item.get("task_id") or "").strip()
        if task_id in seen:
            continue
        application = item.get("application") if isinstance(item.get("application"), dict) else {}
        steps = application.get("steps") if isinstance(application, dict) else []
        for step in reversed(steps if isinstance(steps, list) else []):
            if isinstance(step, dict):
                record = _l6_step_record(mode, step, source)
                if record is not None:
                    records.append(record)
                    seen.add(task_id)
                    break
    return records


def _records_for_mode(mode: str, root: Path) -> list[ScoreRecord]:
    if mode in {"basic_l3", "language_bias_l3", "zero_shot_l6"}:
        return _single_records(mode, root)
    if mode == "skill_l6":
        return _skill_records(mode, root)
    return _dialogue_records(mode, root)


def _average(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _summarize(records: list[ScoreRecord]) -> dict[str, Any]:
    task_count = len(records)
    success_known = [record for record in records if record.success is not None]
    success_count = sum(1 for record in success_known if record.success)
    return {
        "task_count": task_count,
        "scored_count": sum(1 for record in records if record.overall is not None),
        "success_known_count": len(success_known),
        "success_count": success_count,
        "success_rate": success_count / task_count if task_count else None,
        "observed_success_rate": success_count / len(success_known) if success_known else None,
        "average_overall": _average([record.overall for record in records]),
        "average_success_score": _average([record.success_score for record in records]),
        "average_progress_score": _average([record.progress_score for record in records]),
        "average_efficiency_score": _average([record.efficiency_score for record in records]),
        "average_path_quality_score": _average([record.path_quality_score for record in records]),
        "average_invalid_penalty": _average([record.invalid_penalty for record in records]),
    }


def _budget_multiplier_rows(records_by_mode: dict[str, list[ScoreRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, records in records_by_mode.items():
        eligible = [record for record in records if record.shortest_success_trace_length and record.shortest_success_trace_length > 0]
        for multiplier in (1.0, 1.25, 1.5, 1.75, 2.0):
            success_count = 0
            known_success_step_count = 0
            for record in eligible:
                budget = int(math.ceil(float(record.shortest_success_trace_length or 0) * multiplier))
                if record.success_env_step is not None:
                    known_success_step_count += 1
                if record.success is True and record.success_env_step is not None and record.success_env_step <= budget:
                    success_count += 1
            task_count = len(eligible)
            rows.append(
                {
                    "mode": mode,
                    "budget_multiplier": multiplier,
                    "task_count": task_count,
                    "known_success_step_count": known_success_step_count,
                    "success_count": success_count,
                    "success_rate": success_count / task_count if task_count else None,
                }
            )
    return rows


def _budget_integer_rows(records_by_mode: dict[str, list[ScoreRecord]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, records in records_by_mode.items():
        eligible = [record for record in records if record.shortest_success_trace_length and record.shortest_success_trace_length > 0]
        budgets = sorted(
            {
                budget
                for record in eligible
                for budget in range(int(record.shortest_success_trace_length or 0), int(math.ceil(2.0 * float(record.shortest_success_trace_length or 0))) + 1)
            }
        )
        for budget in budgets:
            active = [record for record in eligible if budget >= int(record.shortest_success_trace_length or 0)]
            success_count = sum(
                1
                for record in active
                if record.success is True and record.success_env_step is not None and record.success_env_step <= budget
            )
            rows.append(
                {
                    "mode": mode,
                    "budget_steps": budget,
                    "task_count": len(active),
                    "success_count": success_count,
                    "success_rate": success_count / len(active) if active else None,
                }
            )
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    core_fieldnames = [
        "mode",
        "root",
        "task_count",
        "scored_count",
        "success_known_count",
        "success_count",
        "success_rate",
        "observed_success_rate",
        "average_overall",
        "average_success_score",
        "average_progress_score",
        "average_efficiency_score",
        "average_path_quality_score",
        "average_invalid_penalty",
    ]
    extra_fieldnames = sorted({key for row in rows for key in row} - set(core_fieldnames))
    fieldnames = core_fieldnames + extra_fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    output_base: Path,
    multiplier_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# 8.15 HighLevel V2 Formal Summary",
        "",
        f"- output_base: `{output_base}`",
        "",
        "| Mode | N | SR | M/Overall | Success | Progress | Efficiency | Path Quality | Invalid Penalty |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['mode']} | "
            f"{row['task_count']} | "
            f"{_fmt(row['success_rate'])} | "
            f"{_fmt(row['average_overall'])} | "
            f"{_fmt(row['average_success_score'])} | "
            f"{_fmt(row['average_progress_score'])} | "
            f"{_fmt(row['average_efficiency_score'])} | "
            f"{_fmt(row['average_path_quality_score'])} | "
            f"{_fmt(row['average_invalid_penalty'])} |"
        )
    lines.extend(
        [
            "",
            "## Budget Curve",
            "",
            "| Mode | Budget | N | SR | Success |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in multiplier_rows:
        lines.append(
            "| "
            f"{row['mode']} | "
            f"{row['budget_multiplier']:.2f}x | "
            f"{row['task_count']} | "
            f"{_fmt(row['success_rate'])} | "
            f"{row['success_count']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Budget curve is computed from the same 2x traces. A task counts as successful under a tighter budget only if the first `reached_goal=true` environment step is within that budget.",
            "- Invalid actions and premature finish both consume environment steps.",
            "- `language_bias_l3` uses the same L3 tasks/options as `basic_l3`, but no observation image is attached.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize 8.15 high-level v2 formal eval outputs.")
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    output_base = Path(args.output_base).resolve()
    report_dir = Path(args.report_dir).resolve()
    rows: list[dict[str, Any]] = []
    detailed: dict[str, Any] = {}
    records_by_mode: dict[str, list[ScoreRecord]] = {}
    for mode, parts in MODE_ROOTS.items():
        root = output_base.joinpath(*parts)
        records = _records_for_mode(mode, root) if root.exists() else []
        records_by_mode[mode] = records
        summary = {"mode": mode, "root": str(root), **_summarize(records)}
        rows.append(summary)
        detailed[mode] = {
            "root": str(root),
            "summary": summary,
            "records": [record.__dict__ for record in records],
        }

    multiplier_rows = _budget_multiplier_rows(records_by_mode)
    integer_rows = _budget_integer_rows(records_by_mode)
    detailed["budget_curve"] = {
        "multiplier_rows": multiplier_rows,
        "integer_rows": integer_rows,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "formal_summary.json").write_text(json.dumps(detailed, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(report_dir / "formal_summary.csv", rows)
    _write_csv(report_dir / "budget_curve_multipliers.csv", multiplier_rows)
    _write_csv(report_dir / "budget_curve_integer_steps.csv", integer_rows)
    _write_markdown(report_dir / "formal_summary.md", rows, output_base, multiplier_rows)
    print(json.dumps({"ok": True, "report_dir": str(report_dir), "rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
