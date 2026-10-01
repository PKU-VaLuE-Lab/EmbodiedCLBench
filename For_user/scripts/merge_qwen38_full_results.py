#!/usr/bin/env python3
"""Merge the saved 20% and remaining 76-task Qwen3.8 evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HARNESSES = ("hermesagent", "codex", "claudecode", "openclaw")
SETTINGS = ("basic_l2", "zero_shot_l4", "in_context_l4", "skill_l4")
L4_SETTINGS = SETTINGS[1:]


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def task_id_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith(("task_L2_", "task_L4_")):
            return parent.name
    raise ValueError(f"Cannot infer task id from {path}")


def score_paths(root: Path, harness: str, setting: str) -> list[Path]:
    base = root / harness / setting
    if setting in {"basic_l2", "zero_shot_l4"}:
        return sorted((base / "task_runs").glob("task_L*_*/score.json"))
    if setting == "in_context_l4":
        return sorted(base.glob("task_L4_*/task_runs/*/score.json"))
    if setting == "skill_l4":
        return sorted(base.glob("task_L4_*/skill_application/task_runs/*/score.json"))
    raise ValueError(f"Unsupported setting: {setting}")


def load_records(root: Path, harness: str, setting: str, source: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for score_path in score_paths(root, harness, setting):
        task_id = task_id_from_path(score_path)
        if task_id in records:
            raise ValueError(f"Duplicate {harness}/{setting}/{task_id} in {root}")
        run_dir = score_path.parent
        score = read_json(score_path)
        grade = read_json(run_dir / "task_output" / "grade.json")
        trace_path = run_dir / "task_output" / "trace.jsonl"
        if not trace_path.exists() or trace_path.stat().st_size == 0:
            raise ValueError(f"Missing or empty trace: {trace_path}")
        if bool(score.get("SR")) != bool(grade.get("reached_goal")):
            raise ValueError(f"Score/grade mismatch: {score_path}")
        shortest = int(grade["shortest_success_trace_length"])
        total_steps = int(grade["total_step_count"])
        records[task_id] = {
            "source": source,
            "score_path": str(score_path),
            "reached_goal": bool(grade["reached_goal"]),
            "shortest_steps": shortest,
            "total_steps": total_steps,
            "actual_extra_steps": total_steps - shortest if grade["reached_goal"] else None,
        }
    return records


def merge_records(
    old_root: Path,
    new_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, Any] = {"groups": {}, "ok": True}
    for harness in HARNESSES:
        for setting in SETTINGS:
            old = load_records(old_root, harness, setting, "prior_20pct")
            new = {} if setting == "basic_l2" else load_records(
                new_root, harness, setting, "remaining_76"
            )
            overlap = sorted(set(old) & set(new))
            merged = {**old, **new}
            expected_old = 168 if setting == "basic_l2" else 24
            expected_new = 0 if setting == "basic_l2" else 76
            expected_total = 168 if setting == "basic_l2" else 100
            ok = (
                len(old) == expected_old
                and len(new) == expected_new
                and len(merged) == expected_total
                and not overlap
            )
            key = f"{harness}/{setting}"
            audit["groups"][key] = {
                "prior_count": len(old),
                "remaining_count": len(new),
                "merged_count": len(merged),
                "overlap_task_ids": overlap,
                "ok": ok,
            }
            audit["ok"] = bool(audit["ok"] and ok)
            for task_id, record in sorted(merged.items()):
                row = {"harness": harness, "setting": setting, "task_id": task_id, **record}
                for extra in range(7):
                    row[f"success_extra_{extra}"] = int(
                        record["reached_goal"]
                        and record["total_steps"] <= record["shortest_steps"] + extra
                    )
                rows.append(row)
    if not audit["ok"]:
        raise ValueError(f"Completeness audit failed: {json.dumps(audit, indent=2)}")
    return rows, audit


def aggregate_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["harness"]), str(row["setting"]))].append(row)

    main_table: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for (harness, setting), items in sorted(groups.items()):
        for extra in range(7):
            successful = [row for row in items if row[f"success_extra_{extra}"]]
            curves.append(
                {
                    "harness": harness,
                    "setting": setting,
                    "extra_steps": extra,
                    "n": len(items),
                    "successes": len(successful),
                    "sr": len(successful) / len(items),
                    "er": (
                        sum(int(row["actual_extra_steps"]) for row in successful) / len(successful)
                        if successful
                        else None
                    ),
                }
            )
        at_three = curves[-4]
        main_table.append({**at_three, "metric_budget": "extra_steps=3"})
    return main_table, curves


def task_usage_paths(root: Path, harness: str, setting: str) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for score_path in score_paths(root, harness, setting):
        task_id = task_id_from_path(score_path)
        if setting in {"basic_l2", "zero_shot_l4"}:
            paths = [score_path.parent / "usage.json"]
        else:
            task_root = next(parent for parent in score_path.parents if parent.name == task_id)
            if setting == "in_context_l4":
                paths = [task_root / "usage.json"]
            else:
                paths = [
                    task_root / "skill_learning" / "usage.json",
                    task_root / "skill_application" / "usage.json",
                ]
        for path in paths:
            if not path.exists():
                raise ValueError(f"Missing usage file: {path}")
            entries.append((task_id, path))
    return entries


def normalize_usage(harness: str, payload: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(payload.get("input_tokens") or 0)
    cache_read = int(payload.get("cache_read_tokens") or 0)
    cache_write = int(payload.get("cache_write_tokens") or 0)
    output_tokens = int(payload.get("output_tokens") or 0)
    if harness == "hermesagent":
        uncached = int(payload.get("uncached_input_tokens") or max(input_tokens - cache_read - cache_write, 0))
    elif harness == "codex":
        # Codex reports cached input as a subset of input_tokens.
        uncached = max(input_tokens - cache_read - cache_write, 0)
    else:
        # Claude Code and OpenClaw report uncached and cached input separately.
        uncached = input_tokens
    return {
        "uncached_input_tokens": uncached,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tokens,
    }


def sum_usage(
    roots_and_settings: Iterable[tuple[Path, Iterable[str]]],
    *,
    excluded_task_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_task_ids = excluded_task_ids or set()
    rows: list[dict[str, Any]] = []
    total = defaultdict(int)
    for root, settings in roots_and_settings:
        for harness in HARNESSES:
            for setting in settings:
                values = defaultdict(int)
                task_ids: set[str] = set()
                usage_files = 0
                for task_id, path in task_usage_paths(root, harness, setting):
                    if task_id in excluded_task_ids:
                        continue
                    task_ids.add(task_id)
                    usage_files += 1
                    normalized = normalize_usage(harness, read_json(path))
                    for key, value in normalized.items():
                        values[key] += value
                        total[key] += value
                rows.append(
                    {
                        "harness": harness,
                        "setting": setting,
                        "tasks": len(task_ids),
                        "usage_files": usage_files,
                        **dict(values),
                    }
                )
    return rows, dict(total)


def add_cost(
    row: dict[str, Any],
    *,
    input_price: float,
    cache_read_price: float,
    cache_write_price: float,
    output_price: float,
) -> dict[str, Any]:
    cost = (
        int(row.get("uncached_input_tokens", 0)) * input_price
        + int(row.get("cache_read_tokens", 0)) * cache_read_price
        + int(row.get("cache_write_tokens", 0)) * cache_write_price
        + int(row.get("output_tokens", 0)) * output_price
    ) / 1_000_000
    return {**row, "cost_cny": cost}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-root", required=True, type=Path)
    parser.add_argument("--remaining-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--smoke-task-id", default="task_L4_HL_v5_adult_0015")
    parser.add_argument("--input-price", type=float, default=0.8)
    parser.add_argument("--cache-read-price", type=float, default=0.1)
    parser.add_argument("--cache-write-price", type=float, default=1.25)
    parser.add_argument("--output-price", type=float, default=2.7)
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    task_rows, audit = merge_records(args.prior_root, args.remaining_root)
    main_table, curves = aggregate_metrics(task_rows)
    write_csv(args.report_root / "task_metrics.csv", task_rows)
    write_csv(args.report_root / "main_table_extra3.csv", main_table)
    write_csv(args.report_root / "budget_curves_extra0_to6.csv", curves)
    (args.report_root / "completion_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    price_args = {
        "input_price": args.input_price,
        "cache_read_price": args.cache_read_price,
        "cache_write_price": args.cache_write_price,
        "output_price": args.output_price,
    }
    sections = {
        "basic_l2_saved": ([(args.prior_root, ("basic_l2",))], set()),
        "prior_20pct_l4": ([(args.prior_root, L4_SETTINGS)], set()),
        "remaining_76_l4_including_smoke": ([(args.remaining_root, L4_SETTINGS)], set()),
        "remaining_75_l4_formal_excluding_smoke": (
            [(args.remaining_root, L4_SETTINGS)],
            {args.smoke_task_id},
        ),
        "full_100_l4_saved": (
            [(args.prior_root, L4_SETTINGS), (args.remaining_root, L4_SETTINGS)],
            set(),
        ),
    }
    usage_report: dict[str, Any] = {
        "pricing": {
            "currency": "CNY",
            "unit": "per_1m_tokens",
            **price_args,
            "region": "China (Beijing)",
            "model": "qwen3.8-flash",
        },
        "sections": {},
    }
    for name, (roots_and_settings, excluded) in sections.items():
        detail, total = sum_usage(roots_and_settings, excluded_task_ids=excluded)
        detail_with_cost = [add_cost(row, **price_args) for row in detail]
        usage_report["sections"][name] = {
            "excluded_task_ids": sorted(excluded),
            "breakdown": detail_with_cost,
            "total": add_cost(total, **price_args),
        }
    (args.report_root / "provider_usage_and_cost.json").write_text(
        json.dumps(usage_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit_ok": audit["ok"], "report_root": str(args.report_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
