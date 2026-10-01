#!/usr/bin/env python3
"""Render the paper N-sample SR figure from completed task grades.

The run uses a larger execution budget so that one trajectory can be
thresholded at several extra-step budgets without calling the API again.
This script selects the requested budget from the recorded grade fields.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "analyze/N-sample/output/full/qwen3_7_plus/hermesagent"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "analyze/N-sample/input/subset_10pct_child"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analyze/N-sample/reports/paper_extra_steps_3"
PAPER_BUNDLE_CSV = SCRIPT_ROOT.parent / "n_sample_extra_steps_3.csv"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_task_ids(input_root: Path) -> list[str]:
    payload = read_json(input_root / "ready_l4_tasks_with_images.json")
    if not isinstance(payload, list):
        raise ValueError("ready_l4_tasks_with_images.json must contain a list")
    result = [
        str(item.get("task_id") or "").strip()
        for item in payload
        if isinstance(item, dict) and str(item.get("task_id") or "").strip()
    ]
    if not result:
        raise ValueError("No target L4 task ids found")
    if len(result) != len(set(result)):
        raise ValueError("Target L4 task ids are not unique")
    return result


def parse_grade_path(path: Path, run_root: Path) -> tuple[int, str, str]:
    parts = path.relative_to(run_root).parts
    k_index = next(
        (index for index, part in enumerate(parts) if part.startswith("k") and part[1:].isdigit()),
        None,
    )
    if k_index is None or k_index + 1 >= len(parts):
        raise ValueError(f"Cannot parse k/mode from {path}")
    k = int(parts[k_index][1:])
    mode = parts[k_index + 1]
    if mode not in {"zero_shot", "in_context", "skill"}:
        raise ValueError(f"Unexpected mode in {path}: {mode}")
    task_id = next((part for part in parts if part.startswith("task_L4_")), "")
    if not task_id:
        raise ValueError(f"Cannot parse L4 task id from {path}")
    return k, mode, task_id


def collect_rows(*, run_root: Path, input_root: Path, extra_steps: int) -> list[dict[str, Any]]:
    if extra_steps < 0:
        raise ValueError("extra_steps must be non-negative")
    expected_ids = expected_task_ids(input_root)
    expected_set = set(expected_ids)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()

    for grade_path in sorted(run_root.rglob("grade.json")):
        try:
            k, mode, task_id = parse_grade_path(grade_path, run_root)
        except ValueError:
            continue
        if task_id not in expected_set:
            continue
        if k == 0 and mode != "zero_shot":
            continue
        if k > 0 and mode not in {"in_context", "skill"}:
            continue
        key = (k, mode, task_id)
        if key in seen:
            raise ValueError(f"Duplicate result for {key}")
        seen.add(key)
        payload = read_json(grade_path)
        reached_goal = bool(payload.get("reached_goal"))
        actual_extra_steps = payload.get("extra_steps")
        if not isinstance(actual_extra_steps, (int, float)) or isinstance(actual_extra_steps, bool):
            raise ValueError(f"Missing numeric extra_steps in {grade_path}")
        within_budget = reached_goal and actual_extra_steps <= extra_steps
        rows.append(
            {
                "k": k,
                "mode": mode,
                "task_id": task_id,
                "reached_goal": int(reached_goal),
                "actual_extra_steps": actual_extra_steps,
                "within_budget": int(within_budget),
                "source_path": str(grade_path),
            }
        )

    expected_conditions = [(0, "zero_shot")] + [
        (k, mode) for k in range(1, 5) for mode in ("in_context", "skill")
    ]
    for k, mode in expected_conditions:
        observed = {
            row["task_id"]
            for row in rows
            if row["k"] == k and row["mode"] == mode
        }
        missing = expected_set - observed
        extra = observed - expected_set
        if missing or extra or len(observed) != len(expected_set):
            raise ValueError(
                f"Incomplete {k=}, {mode=}: observed={len(observed)}, "
                f"expected={len(expected_set)}, missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["k"]), str(row["mode"]))].append(row)
    result: list[dict[str, Any]] = []
    for (k, mode), items in sorted(groups.items()):
        successes = sum(int(item["within_budget"]) for item in items)
        n = len(items)
        result.append(
            {
                "harness": "hermesagent",
                "k": k,
                "mode": mode,
                "successes": successes,
                "n": n,
                "sr_percent": 100.0 * successes / n,
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["k", "mode", "successes", "n", "sr_percent"]
    if rows and "harness" in rows[0]:
        fieldnames.insert(0, "harness")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_aggregate_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if not raw:
                continue
            rows.append(
                {
                    "harness": raw.get("harness") or "hermesagent",
                    "k": int(raw["k"]),
                    "mode": raw["mode"],
                    "successes": int(raw["successes"]),
                    "n": int(raw["n"]),
                    "sr_percent": float(raw["sr_percent"]),
                }
            )
    harnesses = sorted({str(row["harness"]) for row in rows})
    max_k = max(int(row["k"]) for row in rows)
    expected = {
        (harness, k, mode)
        for harness in harnesses
        for k, modes in [(0, ("zero_shot",)), *[(value, ("in_context", "skill")) for value in range(1, max_k + 1)]]
        for mode in modes
    }
    observed = {
        (str(row["harness"]), int(row["k"]), str(row["mode"]))
        for row in rows
    }
    if observed != expected:
        raise ValueError(f"Unexpected aggregate CSV conditions: {sorted(observed)}")
    if any(int(row["n"]) != int(rows[0]["n"]) for row in rows):
        raise ValueError("Aggregate CSV has inconsistent sample counts")
    return rows


def plot(*, aggregate_rows: list[dict[str, Any]], output_path: Path) -> dict[str, list[float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    by_key = {
        (str(row.get("harness") or "hermesagent"), int(row["k"]), str(row["mode"])): float(row["sr_percent"])
        for row in aggregate_rows
    }
    harnesses = sorted({key[0] for key in by_key})
    max_k = max(key[1] for key in by_key)
    k_values = list(range(max_k + 1))
    curves: dict[str, list[float]] = {}
    for harness in harnesses:
        baseline = by_key[(harness, 0, "zero_shot")]
        curves[f"{harness}_icl"] = [baseline] + [
            by_key[(harness, k, "in_context")] for k in range(1, max_k + 1)
        ]
        curves[f"{harness}_skill"] = [baseline] + [
            by_key[(harness, k, "skill")] for k in range(1, max_k + 1)
        ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )

    colors = {"hermesagent": "#4C78A8", "openclaw": "#F58518"}
    labels = {"hermesagent": "Hermes", "openclaw": "OpenClaw"}
    gray = "#888888"
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for harness in harnesses:
        color = colors.get(harness, "#54A24B")
        label = labels.get(harness, harness)
        ax.plot(k_values, curves[f"{harness}_icl"], "o-", color=color, label=f"{label} - ICL")
        ax.plot(
            k_values,
            curves[f"{harness}_skill"],
            "s--",
            color=color,
            label=f"{label} - Skill",
            markerfacecolor="white",
            markeredgewidth=1.8,
        )
    ax.axvline(x=2, color=gray, ls=":", lw=0.9, zorder=0)
    ax.text(2.12, 2.0, "default\n($k$=2)", fontsize=8.5, color=gray, va="bottom")
    ax.set_xlabel("Number of Learning Samples ($k$)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_xticks(k_values)
    ax.set_xlim(-0.25, max_k + 0.25)
    upper = max(60.0, 10.0 * ((max(max(values) for values in curves.values()) + 9.9) // 10))
    ax.set_ylim(-2, min(100, upper + 5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the paper N-sample SR figure.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extra-steps", type=int, default=3)
    parser.add_argument(
        "--csv-input",
        type=Path,
        default=PAPER_BUNDLE_CSV if PAPER_BUNDLE_CSV.exists() else None,
        help="Use a pre-aggregated CSV instead of reading grade.json files.",
    )
    args = parser.parse_args()

    csv_input = args.csv_input.resolve() if args.csv_input else None
    run_root = args.run_root.resolve()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    if csv_input is not None:
        aggregate_rows = read_aggregate_csv(csv_input)
        source_run_root: str | None = None
        source_input_root: str | None = None
        expected_count = max(int(row["n"]) for row in aggregate_rows)
    else:
        rows = collect_rows(run_root=run_root, input_root=input_root, extra_steps=args.extra_steps)
        aggregate_rows = aggregate(rows)
        source_run_root = str(run_root)
        source_input_root = str(input_root)
        expected_count = len(expected_task_ids(input_root))
    curve_values = plot(
        aggregate_rows=aggregate_rows,
        output_path=output_dir / f"n_sample_extra_steps_{args.extra_steps}.png",
    )
    csv_output_path = output_dir / f"n_sample_extra_steps_{args.extra_steps}.csv"
    write_csv(csv_output_path, aggregate_rows)
    metadata = {
        "schema_version": "tongbench_paper_n_sample_figure_v1",
        "source_run_root": source_run_root,
        "input_root": source_input_root,
        "csv_input": csv_output_path.name,
        "model": "qwen3.8-flash",
        "harnesses": sorted({str(row.get("harness") or "hermesagent") for row in aggregate_rows}),
        "extra_steps": args.extra_steps,
        "expected_task_count": expected_count,
        "aggregate_rows": aggregate_rows,
        "curve_values_percent": curve_values,
        "openclaw_included": any(str(row.get("harness")) == "openclaw" for row in aggregate_rows),
        "budget_rule": "reached_goal and recorded actual_extra_steps <= extra_steps",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
