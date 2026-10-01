#!/usr/bin/env python3
"""Combine main, prior Hermes, and extended N-sample evaluation results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def task_paths(job: dict[str, Any], task_id: str) -> tuple[Path, list[Path]]:
    output = Path(job["output_dir"])
    run = output / task_id
    application = run / "skill_application" if job.get("mode") == "skill" else run
    scored = application / "task_runs" / ("01_" + task_id)
    phases = [application, run / "skill_learning"] if job.get("mode") == "skill" else [application]
    return scored, phases


def read_metrics(score_path: Path) -> dict[str, Any]:
    score = read_json(score_path)
    grade = read_json(score_path.parent / "task_output" / "grade.json")
    reached = grade.get("reached_goal")
    check(isinstance(reached, bool), f"Invalid reached_goal: {score_path}")
    check(bool(score.get("SR")) == reached, f"SR/grade disagreement: {score_path}")
    shortest = int(grade["shortest_success_trace_length"])
    total = int(grade["valid_action_count"]) + int(grade["invalid_action_count"])
    check(total == int(grade["total_step_count"]), f"Step count disagreement: {score_path}")
    extra = total - shortest if reached else None
    if reached:
        check(extra == grade.get("extra_steps") == score.get("ER"), f"ER disagreement: {score_path}")
    return {
        "reached_goal": reached,
        "shortest_steps": shortest,
        "total_steps": total,
        "actual_extra_steps": extra,
    }


def plan_map(path: Path) -> dict[str, list[str]]:
    return {
        str(row["target_task_id"]): [str(value) for value in row["learning_task_ids"]]
        for row in read_json(path).get("targets", [])
    }


def validate_plans(prepared_root: Path, main_bundle: Path, target_ids: set[str]) -> dict[str, Any]:
    plans = {k: plan_map(prepared_root / f"n_sample/learning_plan_k{k}.json") for k in range(1, 7)}
    check(all(set(plan) == target_ids for plan in plans.values()), "Learning-plan target mismatch")
    for harness in ("hermesagent", "openclaw"):
        main = {
            str(row["target_task_id"]): [str(value) for value in row["learning_task_ids"]]
            for row in read_json(main_bundle / f"manifest/portable_learning_reuse/{harness}.json")["entries"]
            if str(row["target_task_id"]) in target_ids
        }
        check(set(main) == target_ids, f"Main {harness} manifest does not cover selected targets")
        for task_id in target_ids:
            full = plans[6][task_id]
            check(len(full) == len(set(full)) == 6, f"Invalid k=6 sequence: {task_id}")
            check(all(plans[k][task_id] == full[:k] for k in range(1, 7)), f"Non-prefix plans: {task_id}")
            check(main[task_id] == full[:2], f"Main k=2 mismatch for {harness}/{task_id}")
    return {"target_count": len(target_ids), "strict_prefix_axis": True, "main_k2_matches": True}


def collect_job_rows(
    manifest_path: Path,
    *,
    target_ids: set[str],
    source: str,
    allowed: set[tuple[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        if job.get("experiment") != "n_sample":
            continue
        backend = str(job.get("backend") or manifest.get("backend") or "hermesagent")
        output = Path(job["output_dir"])
        k = int(output.parent.name[1:]) if output.parent.name.startswith("k") else int(output.parent.parent.name[1:])
        if allowed is not None and (backend, k) not in allowed:
            continue
        mode = str(job["mode"])
        record = read_json(output / "job.json") if (output / "job.json").is_file() else {}
        summary = read_json(output / "summary.json") if (output / "summary.json").is_file() else {}
        infrastructure_errors = [
            {
                "task_id": str(task.get("task_id", "")),
                "error": str(task.get("error", "")),
            }
            for task in summary.get("tasks", [])
            if str(task.get("error") or "").strip()
        ]
        count = 0
        errors: list[str] = []
        for task_id in sorted(target_ids):
            scored, phases = task_paths(job, task_id)
            score_path = scored / "score.json"
            if not score_path.is_file():
                errors.append(task_id)
                continue
            metrics = read_metrics(score_path)
            rows.append(
                {
                    "harness": backend,
                    "k": k,
                    "mode": mode,
                    "task_id": task_id,
                    "source": source,
                    "score_path": str(score_path),
                    **metrics,
                }
            )
            count += 1
            for phase in phases:
                usage_path = phase / "usage.json"
                if not usage_path.is_file():
                    continue
                usage = read_json(usage_path)
                usage_rows.append(
                    {
                        "harness": backend,
                        "k": k,
                        "mode": mode,
                        "task_id": task_id,
                        "phase": "skill_summary" if phase.name == "skill_learning" else "application",
                        **{
                            key: usage.get(key, 0)
                            for key in (
                                "input_tokens",
                                "output_tokens",
                                "cache_read_tokens",
                                "uncached_input_tokens",
                                "request_count",
                            )
                        },
                        "source_path": str(usage_path),
                    }
                )
        jobs.append(
            {
                "job": str(job.get("label") or f"{backend}_k{k}_{mode}"),
                "valid_results": count,
                "expected": len(target_ids),
                "returncode": record.get("returncode"),
                "missing_task_ids": errors,
                "infrastructure_errors": infrastructure_errors,
                "complete": count == len(target_ids) and not infrastructure_errors,
            }
        )
    return rows, usage_rows, jobs


def load_main_rows(repo_root: Path, main_bundle: Path, target_ids: set[str]) -> list[dict[str, Any]]:
    with (main_bundle / "reports/full_100/task_metrics.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for harness in ("hermesagent", "openclaw"):
        selected = [
            row for row in source_rows
            if row["harness"] == harness
            and row["task_id"] in target_ids
            and row["setting"] in ("zero_shot_l4", "in_context_l4", "skill_l4")
        ]
        check(len(selected) == 150, f"Main results incomplete for {harness}: {len(selected)}/150")
        for row in selected:
            setting = row["setting"]
            mode = setting[:-3] if setting.endswith("_l4") else setting
            score_path = repo_root / row["score_path"]
            output.append(
                {
                    "harness": harness,
                    "k": 0 if mode == "zero_shot" else 2,
                    "mode": mode,
                    "task_id": row["task_id"],
                    "source": "canonical_main_eval",
                    "score_path": str(score_path),
                    **read_metrics(score_path),
                }
            )
    return output


def aggregate(rows: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["harness"]), int(row["k"]), str(row["mode"]))].append(row)
    output: list[dict[str, Any]] = []
    for (harness, k, mode), items in sorted(groups.items()):
        check(len(items) == target_count, f"Incomplete condition {harness}/k{k}/{mode}: {len(items)}")
        for budget in range(7):
            successes = [
                row for row in items
                if row["reached_goal"] and int(row["actual_extra_steps"]) <= budget
            ]
            output.append(
                {
                    "harness": harness,
                    "k": k,
                    "mode": mode,
                    "extra_budget": budget,
                    "n": target_count,
                    "successes": len(successes),
                    "sr_percent": 100.0 * len(successes) / target_count,
                    "er": (
                        sum(int(row["actual_extra_steps"]) for row in successes) / len(successes)
                        if successes else None
                    ),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--new-run-root", type=Path, required=True)
    parser.add_argument("--prior-hermes-run-root", type=Path, required=True)
    parser.add_argument("--main-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.prepared_root = args.prepared_root.resolve()
    args.new_run_root = args.new_run_root.resolve()
    args.prior_hermes_run_root = args.prior_hermes_run_root.resolve()
    args.main_bundle = args.main_bundle.resolve()
    args.output_dir = args.output_dir.resolve()

    target_ids = {
        str(row["task_id"])
        for row in read_json(args.prepared_root / "n_sample/ready_l4_tasks.json")
    }
    plan_audit = validate_plans(args.prepared_root, args.main_bundle, target_ids)
    main_rows = load_main_rows(args.repo_root, args.main_bundle, target_ids)
    prior_rows, _, prior_jobs = collect_job_rows(
        args.prior_hermes_run_root / "run_manifest.json",
        target_ids=target_ids,
        source="prior_formal_analysis",
        allowed={("hermesagent", 1), ("hermesagent", 3), ("hermesagent", 4)},
    )
    new_rows, usage_rows, new_jobs = collect_job_rows(
        args.new_run_root / "run_manifest.json",
        target_ids=target_ids,
        source="extended_formal_analysis",
    )
    rows = main_rows + prior_rows + new_rows
    expected_keys = {
        (harness, k, mode)
        for harness in ("hermesagent", "openclaw")
        for k, modes in [(0, ("zero_shot",)), *[(value, ("in_context", "skill")) for value in range(1, 7)]]
        for mode in modes
    }
    observed_keys = {(str(row["harness"]), int(row["k"]), str(row["mode"])) for row in rows}
    check(observed_keys == expected_keys, f"Condition mismatch: missing={sorted(expected_keys - observed_keys)}")
    curves = aggregate(rows, len(target_ids))
    write_csv(args.output_dir / "task_metrics.csv", rows)
    write_csv(args.output_dir / "budget_curves_extra0_to6.csv", curves)
    write_csv(args.output_dir / "extra3.csv", [row for row in curves if row["extra_budget"] == 3])
    write_csv(args.output_dir / "provider_usage_new_runs.csv", usage_rows)
    audit = {
        "schema_version": "tongbench_n_sample_extended_audit_v1",
        "complete": all(
            job["complete"]
            for job in prior_jobs + new_jobs
        ),
        "model": "qwen3.8-flash",
        "thinking": "medium",
        "context_mode": "task_compact",
        "target_count": len(target_ids),
        "condition_count": len(expected_keys),
        "result_row_count": len(rows),
        "expected_result_row_count": 1300,
        "plan_validation": plan_audit,
        "prior_jobs": prior_jobs,
        "new_jobs": new_jobs,
        "new_usage": {
            key: sum(int(row.get(key) or 0) for row in usage_rows)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "uncached_input_tokens",
                "request_count",
            )
        },
    }
    check(len(rows) == 1300, f"Unexpected result count: {len(rows)}")
    check(audit["complete"], "One or more result jobs are incomplete")
    write_json(args.output_dir / "completion_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
