#!/usr/bin/env python3
"""Validate that Stage 0 produced real, usable harness runs.

This validator checks execution artifacts and child-process propagation.  It
does not judge whether a model chose the benchmark's correct action: a model
failure is a valid score, while a missing trace or failed harness is a Smoke
failure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FATAL_MARKERS = (
    "ModuleNotFoundError",
    "Unable to serialize unknown type",
    "command not found",
    "Traceback (most recent call last)",
)
TRACE_MARKERS = ("observe", "select_option", "choose_action", "finish")


def words(raw: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", raw.strip()) if part]


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def task_run_dirs(setting_root: Path) -> list[Path]:
    dirs: set[Path] = set()
    for result_path in setting_root.rglob("single_task_run_result.json"):
        dirs.add(result_path.parent)
    for score_path in setting_root.rglob("score.json"):
        if (score_path.parent / "agent.log").exists():
            dirs.add(score_path.parent)
    return sorted(dirs)


def validate_task_run(task_dir: Path) -> tuple[str, list[str]]:
    problems: list[str] = []
    result = read_json(task_dir / "single_task_run_result.json")
    if result is not None and result.get("returncode") not in (0, None):
        problems.append(f"outer runner returncode={result.get('returncode')}")
    if not (task_dir / "score.json").is_file():
        problems.append("score.json missing")
    agent_log = task_dir / "agent.log"
    if not agent_log.is_file() or not agent_log.stat().st_size:
        problems.append("agent.log missing or empty")

    usage_path = task_dir / "usage.json"
    usage = read_json(usage_path) if usage_path.exists() else None
    if usage is not None:
        if usage.get("error"):
            problems.append(f"usage error: {usage['error']}")
        if usage.get("agent_returncode") not in (None, 0):
            problems.append(f"agent returncode={usage.get('agent_returncode')}")

    texts: list[str] = []
    for filename in ("agent.log", "runner_stderr.txt", "bridge_stderr.log"):
        path = task_dir / filename
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    combined = "\n".join(texts)
    for marker in FATAL_MARKERS:
        if marker in combined:
            problems.append(f"runtime marker: {marker}")

    trace_paths = list(task_dir.rglob("trace.jsonl"))
    has_trace = any(path.is_file() and path.stat().st_size > 0 for path in trace_paths)
    if not has_trace and not any(marker in combined for marker in TRACE_MARKERS):
        problems.append("no non-empty trace or action-loop marker found")
    return ("failed" if problems else "ok", problems)


def validate_dialogue_run(run_dir: Path) -> tuple[str, list[str]]:
    """Validate an ICL/Skill application directory.

    Dialogue runs keep the application ``agent.log`` beside a nested
    ``task_runs`` directory, so they do not have the same layout as
    ``ready_single_tasks`` output.
    """
    problems: list[str] = []
    agent_log = run_dir / "agent.log"
    if not agent_log.is_file() or not agent_log.stat().st_size:
        problems.append("agent.log missing or empty")
    if not any(path.is_file() for path in run_dir.rglob("score.json")):
        problems.append("application score.json missing")
    usage_path = run_dir / "usage.json"
    if usage_path.exists():
        usage = read_json(usage_path)
        if usage is None:
            problems.append("usage.json is not valid JSON")
        else:
            if usage.get("error"):
                problems.append(f"usage error: {usage['error']}")
            if usage.get("agent_returncode") not in (None, 0):
                problems.append(f"agent returncode={usage.get('agent_returncode')}")

    texts: list[str] = []
    for path in (agent_log, run_dir / "runner_stderr.txt", run_dir / "bridge_stderr.log"):
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
    combined = "\n".join(texts)
    for marker in FATAL_MARKERS:
        if marker in combined:
            problems.append(f"runtime marker: {marker}")
    if not any(
        path.is_file() and path.stat().st_size > 0 for path in run_dir.rglob("trace.jsonl")
    ) and not any(marker in combined for marker in TRACE_MARKERS):
        problems.append("no non-empty trace or action-loop marker found")
    return ("failed" if problems else "ok", problems)


def validate_setting(setting_root: Path, setting: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "setting": setting,
        "path": str(setting_root),
        "status": "failed",
        "task_count": 0,
        "problems": [],
    }
    if not setting_root.is_dir():
        record["problems"] = ["setting output directory missing"]
        return record

    task_dirs = task_run_dirs(setting_root)
    record["task_count"] = len(task_dirs)
    if not task_dirs:
        # Dialogue settings place agent.log and task_runs under one
        # application directory rather than writing single_task_run_result.
        dialogue_dirs = sorted(
            {
                path.parent
                for path in setting_root.rglob("agent.log")
                if (path.parent / "task_runs").is_dir()
            }
        )
        if not dialogue_dirs:
            record["problems"] = ["no task run artifacts found"]
            return record
        record["task_count"] = len(dialogue_dirs)
        for run_dir in dialogue_dirs:
            status, problems = validate_dialogue_run(run_dir)
            if problems:
                record["problems"].append(
                    {"application": str(run_dir.relative_to(setting_root)), "problems": problems}
                )
            if status != "ok":
                continue

    else:
        for task_dir in task_dirs:
            status, problems = validate_task_run(task_dir)
            if problems:
                record["problems"].append(
                    {"task_run": str(task_dir.relative_to(setting_root)), "problems": problems}
                )
            if status != "ok":
                continue

    # Learning/application settings also need their orchestration artifacts.
    if setting == "in_context_l4" and not any(setting_root.rglob("sequence_summary.json")):
        record["problems"].append("in-context sequence_summary.json missing")
    if setting == "skill_l4":
        if not any("skill" in path.name.lower() for path in setting_root.rglob("*")):
            record["problems"].append("skill summary/checkpoint artifact missing")

    record["status"] = "ok" if not record["problems"] else "failed"
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--models", default="qwen")
    parser.add_argument("--harnesses", default="hermesagent codex claudecode openclaw")
    parser.add_argument("--settings", default="basic_l2 zero_shot_l4 in_context_l4 skill_l4")
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for model in words(args.models):
        for harness in words(args.harnesses):
            for setting in words(args.settings):
                record = validate_setting(args.stage_root / model / harness / setting, setting)
                record["model"] = model
                record["harness"] = harness
                records.append(record)

    failed = [record for record in records if record["status"] != "ok"]
    payload = {
        "stage_root": str(args.stage_root.resolve()),
        "expected_count": len(records),
        "passed_count": len(records) - len(failed),
        "failed_count": len(failed),
        "records": records,
    }
    args.stage_root.mkdir(parents=True, exist_ok=True)
    (args.stage_root / "stage0_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Stage 0 Smoke Validation",
        "",
        f"- Expected: {payload['expected_count']}",
        f"- Passed: {payload['passed_count']}",
        f"- Failed: {payload['failed_count']}",
        "",
        "| Model | Harness | Setting | Runs | Status | Problems |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        problems = record["problems"]
        problem_text = "; ".join(
            str(item) if isinstance(item, str) else str(item.get("problems", item))
            for item in problems
        )
        lines.append(
            f"| {record['model']} | {record['harness']} | {record['setting']} | "
            f"{record['task_count']} | {record['status']} | {problem_text} |"
        )
    (args.stage_root / "stage0_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in ("expected_count", "passed_count", "failed_count")}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
