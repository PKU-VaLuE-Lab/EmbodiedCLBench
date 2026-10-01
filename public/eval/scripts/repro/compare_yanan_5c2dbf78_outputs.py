#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_PACKAGE_REL = Path("important_results_eval/8.10_remote_yanan_5c2dbf78")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _score_payload(step: dict[str, Any]) -> dict[str, Any]:
    grade = step.get("grade") if isinstance(step.get("grade"), dict) else {}
    score = step.get("score") if isinstance(step.get("score"), dict) else {}
    details = score.get("details") if isinstance(score.get("details"), dict) else {}
    return {"grade": grade, "score": score, "details": details}


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None


def _field(step: dict[str, Any], name: str) -> Any:
    payload = _score_payload(step)
    for container in (payload["grade"], payload["details"], payload["score"]):
        if name in container:
            return container.get(name)
    return None


def _step_score(step: dict[str, Any]) -> float | None:
    payload = _score_payload(step)
    return _first_float(
        payload["grade"].get("overall"),
        payload["score"].get("overall_score"),
        payload["score"].get("score"),
    )


def _step_reached(step: dict[str, Any]) -> bool | None:
    return _first_bool(_field(step, "reached_goal"))


def _trace_action_sequence(summary_path: Path, composite_task_id: str, step: dict[str, Any]) -> list[str]:
    index = int(step.get("index") or 0)
    task_id = str(step.get("task_id") or "")
    if not index or not task_id:
        return []
    trace_path = (
        summary_path.parent
        / composite_task_id
        / "task_runs"
        / f"{index:02d}_{task_id}"
        / "task_output"
        / "trace.jsonl"
    )
    if not trace_path.exists():
        return []
    actions: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        action = record.get("selected_action_id") or record.get("canonical_action")
        actions.append(str(action) if action else "<invalid_or_unparsed>")
    return actions


def _summary_tasks(summary_path: Path) -> dict[str, dict[str, Any]]:
    if not summary_path.exists():
        return {}
    payload = _read_json(summary_path)
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    return {str(task.get("task_id") or ""): task for task in tasks if isinstance(task, dict)}


def _compare_status(ref_score: float | None, our_score: float | None, ref_reached: bool | None, our_reached: bool | None, score_tolerance: float) -> str:
    if ref_score is None and our_score is None and ref_reached is None and our_reached is None:
        return "no_step_score"
    if ref_reached != our_reached:
        return "different_reached_goal"
    if ref_score is None or our_score is None:
        return "missing_score"
    if abs(ref_score - our_score) <= score_tolerance:
        return "close"
    return "same_reached_goal_score_diff"


def _compare_one(
    *,
    reference_summary: Path,
    our_summary: Path,
    level: str,
    backend: str,
    score_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_tasks = _summary_tasks(reference_summary)
    our_tasks = _summary_tasks(our_summary)
    task_ids = sorted(set(ref_tasks) | set(our_tasks))
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        ref_task = ref_tasks.get(task_id)
        our_task = our_tasks.get(task_id)
        if ref_task is None or our_task is None:
            task_rows.append(
                {
                    "level": level,
                    "backend": backend,
                    "task_id": task_id,
                    "status": "missing_reference" if ref_task is None else "missing_our_output",
                    "ref_error": None if ref_task is None else ref_task.get("error"),
                    "our_error": None if our_task is None else our_task.get("error"),
                }
            )
            continue
        ref_steps = {int(step.get("index") or 0): step for step in ref_task.get("steps", []) if isinstance(step, dict)}
        our_steps = {int(step.get("index") or 0): step for step in our_task.get("steps", []) if isinstance(step, dict)}
        statuses: list[str] = []
        for step_index in sorted(set(ref_steps) | set(our_steps)):
            ref_step = ref_steps.get(step_index)
            our_step = our_steps.get(step_index)
            if ref_step is None or our_step is None:
                status = "missing_reference_step" if ref_step is None else "missing_our_step"
                row = {
                    "level": level,
                    "backend": backend,
                    "composite_task_id": task_id,
                    "step_index": step_index,
                    "step_task_id": None,
                    "status": status,
                }
                rows.append(row)
                statuses.append(status)
                continue
            ref_score = _step_score(ref_step)
            our_score = _step_score(our_step)
            ref_reached = _step_reached(ref_step)
            our_reached = _step_reached(our_step)
            ref_actions = _trace_action_sequence(reference_summary, task_id, ref_step)
            our_actions = _trace_action_sequence(our_summary, task_id, our_step)
            status = _compare_status(ref_score, our_score, ref_reached, our_reached, score_tolerance)
            statuses.append(status)
            rows.append(
                {
                    "level": level,
                    "backend": backend,
                    "composite_task_id": task_id,
                    "step_index": step_index,
                    "step_task_id": str(ref_step.get("task_id") or our_step.get("task_id") or ""),
                    "status": status,
                    "ref_score": ref_score,
                    "our_score": our_score,
                    "score_delta": None if ref_score is None or our_score is None else round(our_score - ref_score, 6),
                    "ref_reached_goal": ref_reached,
                    "our_reached_goal": our_reached,
                    "ref_error_type": _field(ref_step, "error_type"),
                    "our_error_type": _field(our_step, "error_type"),
                    "ref_valid_action_count": _field(ref_step, "valid_action_count"),
                    "our_valid_action_count": _field(our_step, "valid_action_count"),
                    "ref_invalid_action_count": _field(ref_step, "invalid_action_count"),
                    "our_invalid_action_count": _field(our_step, "invalid_action_count"),
                    "ref_request_count": (ref_task.get("usage") or {}).get("request_count") if isinstance(ref_task.get("usage"), dict) else None,
                    "our_request_count": (our_task.get("usage") or {}).get("request_count") if isinstance(our_task.get("usage"), dict) else None,
                    "action_sequence_equal": ref_actions == our_actions,
                    "ref_actions": ref_actions,
                    "our_actions": our_actions,
                }
            )
        task_rows.append(
            {
                "level": level,
                "backend": backend,
                "task_id": task_id,
                "status": "ok" if all(status == "close" for status in statuses) else "diff_or_warning",
                "ref_error": ref_task.get("error"),
                "our_error": our_task.get("error"),
                "step_statuses": statuses,
            }
        )
    aggregate = {
        "level": level,
        "backend": backend,
        "reference_summary": str(reference_summary),
        "our_summary": str(our_summary),
        "reference_summary_exists": reference_summary.exists(),
        "our_summary_exists": our_summary.exists(),
        "task_count_reference": len(ref_tasks),
        "task_count_our": len(our_tasks),
        "task_comparisons": task_rows,
    }
    return rows, aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "backend",
        "composite_task_id",
        "step_index",
        "step_task_id",
        "status",
        "ref_score",
        "our_score",
        "score_delta",
        "ref_reached_goal",
        "our_reached_goal",
        "ref_error_type",
        "our_error_type",
        "ref_valid_action_count",
        "our_valid_action_count",
        "ref_invalid_action_count",
        "our_invalid_action_count",
        "ref_request_count",
        "our_request_count",
        "action_sequence_equal",
        "ref_actions",
        "our_actions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["ref_actions"] = " | ".join(row.get("ref_actions") or [])
            csv_row["our_actions"] = " | ".join(row.get("our_actions") or [])
            writer.writerow({key: csv_row.get(key) for key in fieldnames})


def _markdown_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Yanan 5c2dbf78 Eval Reproduction Compare",
        "",
        f"- package_root: `{report['package_root']}`",
        f"- score_tolerance: `{report['score_tolerance']}`",
        f"- compared_backends: `{', '.join(report['backends'])}`",
        f"- compared_levels: `{', '.join(report['levels'])}`",
        "",
        "## Summary",
        "",
        "| level | backend | ref tasks | our tasks | close steps | warning/diff steps | missing steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in report["comparisons"]:
        level = aggregate["level"]
        backend = aggregate["backend"]
        subset = [row for row in rows if row.get("level") == level and row.get("backend") == backend]
        close = sum(1 for row in subset if row.get("status") == "close")
        missing = sum(1 for row in subset if str(row.get("status", "")).startswith("missing"))
        warning = len(subset) - close - missing
        lines.append(
            f"| {level} | {backend} | {aggregate['task_count_reference']} | {aggregate['task_count_our']} | {close} | {warning} | {missing} |"
        )
    lines.extend(
        [
            "",
            "## Step Details",
            "",
            "| level | backend | composite | step | task | status | ref score | our score | delta | ref reached | our reached | actions equal |",
            "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in rows:
        ref_score = "" if row.get("ref_score") is None else f"{float(row['ref_score']):.6f}"
        our_score = "" if row.get("our_score") is None else f"{float(row['our_score']):.6f}"
        delta = "" if row.get("score_delta") is None else f"{float(row['score_delta']):.6f}"
        lines.append(
            "| {level} | {backend} | {composite_task_id} | {step_index} | {step_task_id} | {status} | {ref_score} | {our_score} | {delta} | {ref_reached_goal} | {our_reached_goal} | {action_sequence_equal} |".format(
                **{
                    **row,
                    "ref_score": ref_score,
                    "our_score": our_score,
                    "delta": delta,
                }
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `close` means reached_goal is the same and score delta is within tolerance.",
            "- Action sequences are expected to differ when the model samples a different valid path, so this is reported separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare collaborator reference output against local refactor reproduction output.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--backends", nargs="+", default=["hermesagent", "openclaw"])
    parser.add_argument("--levels", nargs="+", default=["L2", "L3"])
    parser.add_argument("--score-tolerance", type=float, default=0.25)
    parser.add_argument("--report-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    package_root = (args.package_root or project_root / DEFAULT_PACKAGE_REL).resolve()
    reference_root = package_root / "reference_output"
    our_root = package_root / "our_output"
    report_dir = (args.report_dir or package_root / "compare_report").resolve()
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for level in [level.upper() for level in args.levels]:
        output_name = "outputs_l2" if level == "L2" else "outputs_l3"
        for backend in args.backends:
            reference_summary = reference_root / output_name / "dialogue" / backend / "summary.json"
            our_summary = our_root / output_name / "dialogue" / backend / "summary.json"
            new_rows, aggregate = _compare_one(
                reference_summary=reference_summary,
                our_summary=our_summary,
                level=level,
                backend=backend,
                score_tolerance=float(args.score_tolerance),
            )
            rows.extend(new_rows)
            comparisons.append(aggregate)

    report = {
        "package_root": str(package_root),
        "reference_root": str(reference_root),
        "our_root": str(our_root),
        "backends": args.backends,
        "levels": [level.upper() for level in args.levels],
        "score_tolerance": float(args.score_tolerance),
        "comparisons": comparisons,
        "step_rows": rows,
    }
    _write_json(report_dir / "comparison_summary.json", report)
    _write_csv(report_dir / "step_comparison.csv", rows)
    (report_dir / "README.md").write_text(_markdown_report(report, rows), encoding="utf-8")
    print(f"Wrote comparison report: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
