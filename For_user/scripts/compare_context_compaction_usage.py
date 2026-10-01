#!/usr/bin/env python3
"""Compare provider-reported token usage between two evaluation roots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


USAGE_FIELDS = (
    "input_tokens",
    "uncached_input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "total_tokens",
    "request_count",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def phase_for(relative_path: Path) -> str | None:
    parts = relative_path.parts
    if "skill_learning" in parts:
        return "skill_learning"
    if "skill_application" in parts:
        return "skill_application"
    if "in_context_l4" in parts:
        return "in_context_application"
    if "basic_l2" in parts:
        return "basic_l2"
    if "zero_shot_l4" in parts:
        return "zero_shot_l4"
    return None


def task_id_for(relative_path: Path) -> str:
    for part in reversed(relative_path.parts):
        if part.startswith("task_") and part not in {"task_runs", "task_run"}:
            return part
    return "(aggregate)"


def collect(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.rglob("primary_result.json")):
        relative_path = path.relative_to(root)
        phase = phase_for(relative_path)
        if phase is None:
            continue
        payload = read_json(path)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            continue
        task_id = task_id_for(relative_path)
        key = (phase, task_id)
        records[key] = {
            "phase": phase,
            "task_id": task_id,
            "relative_path": str(relative_path),
            "usage": {field: usage.get(field) for field in USAGE_FIELDS},
            "usage_source": usage.get("usage_source"),
            "usage_complete": usage.get("usage_complete"),
        }
    return records


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def compare_record(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "phase": before["phase"],
        "task_id": before["task_id"],
        "relative_path": before["relative_path"],
        "before_usage": before["usage"],
        "after_usage": after["usage"],
        "usage_source_before": before.get("usage_source"),
        "usage_source_after": after.get("usage_source"),
        "usage_complete_before": before.get("usage_complete"),
        "usage_complete_after": after.get("usage_complete"),
        "delta": {},
        "percent_change": {},
    }
    for field in USAGE_FIELDS:
        old = numeric(before["usage"].get(field))
        new = numeric(after["usage"].get(field))
        if old is None or new is None:
            result["delta"][field] = None
            result["percent_change"][field] = None
            continue
        result["delta"][field] = new - old
        result["percent_change"][field] = None if old == 0 else (new - old) / old * 100
    return result


def aggregate(records: list[dict[str, Any]], side: str) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = defaultdict(lambda: {field: 0 for field in USAGE_FIELDS})
    for record in records:
        phase = record["phase"]
        usage = record[f"{side}_usage"]
        for field in USAGE_FIELDS:
            value = numeric(usage.get(field))
            if value is not None:
                totals[phase][field] += value
    return dict(totals)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(
    path: Path,
    *,
    before_root: Path,
    after_root: Path,
    comparisons: list[dict[str, Any]],
    before_totals: dict[str, dict[str, Any]],
    after_totals: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Provider Usage Comparison",
        "",
        "This report uses the provider-reported usage fields from each `primary_result.json`; it does not estimate tokens from characters.",
        "",
        f"- Before: `{before_root}`",
        f"- After: `{after_root}`",
        "",
        "## Matched Runs",
        "",
        "| phase | task | input before | input after | input delta | total before | total after | total delta | requests before/after |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        old = item["before_usage"]
        new = item["after_usage"]
        delta = item["delta"]
        lines.append(
            "| {phase} | `{task}` | {ib} | {ia} | {idelta} | {tb} | {ta} | {tdelta} | {rb}/{ra} |".format(
                phase=item["phase"],
                task=item["task_id"],
                ib=fmt(old.get("input_tokens")),
                ia=fmt(new.get("input_tokens")),
                idelta=fmt(delta.get("input_tokens")),
                tb=fmt(old.get("total_tokens")),
                ta=fmt(new.get("total_tokens")),
                tdelta=fmt(delta.get("total_tokens")),
                rb=fmt(old.get("request_count")),
                ra=fmt(new.get("request_count")),
            )
        )
    lines.extend(
        [
            "",
            "## Phase Totals",
            "",
            "| phase | input before | input after | input delta | total before | total after | total delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for phase in sorted(set(before_totals) | set(after_totals)):
        old = before_totals.get(phase, {})
        new = after_totals.get(phase, {})
        lines.append(
            f"| {phase} | {fmt(old.get('input_tokens'))} | {fmt(new.get('input_tokens'))} | "
            f"{fmt((new.get('input_tokens') or 0) - (old.get('input_tokens') or 0))} | "
            f"{fmt(old.get('total_tokens'))} | {fmt(new.get('total_tokens'))} | "
            f"{fmt((new.get('total_tokens') or 0) - (old.get('total_tokens') or 0))} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", required=True, type=Path)
    parser.add_argument("--after-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    before = collect(args.before_root)
    after = collect(args.after_root)
    comparisons = [
        compare_record(before[key], after[key])
        for key in sorted(set(before) & set(after))
    ]
    before_totals = aggregate(comparisons, "before")
    after_totals = aggregate(comparisons, "after")
    report = {
        "schema_version": "tongbench_provider_usage_comparison_v1",
        "before_root": str(args.before_root.resolve()),
        "after_root": str(args.after_root.resolve()),
        "matched_record_count": len(comparisons),
        "before_record_count": len(before),
        "after_record_count": len(after),
        "comparisons": comparisons,
        "phase_totals": {"before": before_totals, "after": after_totals},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        args.output_md,
        before_root=args.before_root,
        after_root=args.after_root,
        comparisons=comparisons,
        before_totals=before_totals,
        after_totals=after_totals,
    )
    print(json.dumps({"matched_record_count": len(comparisons), "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
