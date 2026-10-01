#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def as_bool_success(result: dict[str, Any]) -> bool:
    score = result.get("score") or {}
    if isinstance(score, dict) and "SR" in score:
        return bool(score.get("SR"))
    details = score.get("details") if isinstance(score, dict) else {}
    if isinstance(details, dict) and "reached_goal" in details:
        return bool(details.get("reached_goal"))
    if isinstance(score, dict) and "success" in score:
        return bool(score.get("success"))
    if result.get("success") is not None:
        return bool(result.get("success"))
    return as_float(score.get("overall_score") if isinstance(score, dict) else 0.0) > 0.99


def metric_from_result(result: dict[str, Any], key: str) -> float:
    score = result.get("score") or {}
    if isinstance(score, dict) and key == "er" and "ER" in score:
        return as_float(score.get("ER"))
    details = score.get("details") if isinstance(score, dict) else {}
    if isinstance(details, dict):
        if key == "sr":
            return 1.0 if as_bool_success(result) else 0.0
        if key == "er":
            extra_steps = details.get("extra_steps")
            return as_float(extra_steps)
    return 0.0


def load_grade_for_record(record: dict[str, Any]) -> dict[str, Any]:
    grade = record.get("grade") if isinstance(record.get("grade"), dict) else {}
    if grade:
        return grade
    output_dir = str(record.get("output_dir") or "").strip()
    if not output_dir:
        return {}
    grade_path = Path(output_dir) / "task_output" / "grade.json"
    if not grade_path.exists():
        return {}
    try:
        payload = read_json(grade_path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def row_from_scored_record(
    *,
    stage: str,
    model: str,
    harness: str,
    setting: str,
    task_id: str,
    record: dict[str, Any],
    summary_path: Path,
    max_steps_extra: Any,
    status: str = "",
    returncode: Any = None,
) -> dict[str, Any]:
    score = record.get("score") or {}
    details = score.get("details") if isinstance(score, dict) else {}
    grade = load_grade_for_record(record)
    max_steps = record.get("max_steps")
    shortest = None
    if max_steps is not None and max_steps_extra is not None:
        shortest = int(max_steps) - int(max_steps_extra)
    if shortest is None and grade.get("shortest_success_trace_length") is not None:
        shortest = int(grade["shortest_success_trace_length"])
    env_action_count = None
    if grade.get("valid_action_count") is not None and grade.get("invalid_action_count") is not None:
        env_action_count = int(grade["valid_action_count"]) + int(grade["invalid_action_count"])
    success_step = None
    if as_bool_success(record):
        if env_action_count is not None:
            success_step = env_action_count
        else:
            for source in (details, grade):
                if not isinstance(source, dict):
                    continue
                for key in ("goal_reached_step", "success_step", "steps_to_success"):
                    if source.get(key) is not None:
                        success_step = int(source[key])
                        break
                if success_step is not None:
                    break
    success = int(as_bool_success(record))
    return {
        "stage": stage,
        "model": model,
        "harness": harness,
        "setting": setting,
        "task_id": task_id,
        "success": success,
        "er": metric_from_result(record, "er") if success else None,
        "max_steps": max_steps,
        "max_steps_extra": max_steps_extra,
        "shortest_estimate": shortest,
        "env_action_count": env_action_count,
        "success_step": success_step,
        "status": status,
        "returncode": returncode,
        "summary_path": str(summary_path),
    }


def infer_meta(summary_path: Path, output_root: Path) -> tuple[str, str, str, str]:
    rel = summary_path.relative_to(output_root)
    parts = rel.parts
    if len(parts) >= 5:
        return parts[0], parts[1], parts[2], parts[3]
    return ("unknown", "unknown", "unknown", "unknown")


def iter_summary_files(output_root: Path, stages: set[str]) -> list[Path]:
    files = []
    roots = [output_root / stage for stage in sorted(stages)] if stages else [output_root]
    for root in roots:
        if not root.exists():
            continue
        for name in ("single_task_summary.json", "summary.json"):
            files.extend(root.glob(f"**/{name}"))
    return sorted(set(files))


def rows_from_summary(path: Path, output_root: Path) -> list[dict[str, Any]]:
    stage, model, harness, setting = infer_meta(path, output_root)
    payload = read_json(path)
    if path.name == "summary.json" and (payload.get("results") or payload.get("tasks")):
        records = payload.get("results") or payload.get("tasks") or []
        rows: list[dict[str, Any]] = []
        for result in records:
            if not isinstance(result, dict):
                continue
            task_id = str(result.get("task_id") or result.get("conversation_task_id") or "")
            application = result.get("application") if isinstance(result.get("application"), dict) else {}
            steps = application.get("steps") if isinstance(application.get("steps"), list) else None
            if steps:
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                    step_task_id = str(step.get("task_id") or task_id)
                    score = step.get("score") if isinstance(step.get("score"), dict) else {}
                    grade = step.get("grade") if isinstance(step.get("grade"), dict) else {}
                    if not score and grade:
                        score = {
                            "overall_score": grade.get("overall"),
                            "details": {
                                "reached_goal": grade.get("reached_goal"),
                                "extra_steps": grade.get("extra_steps"),
                                "breakpoint_depth": grade.get("breakpoint_depth"),
                            },
                        }
                    rows.append(
                        row_from_scored_record(
                            stage=stage,
                            model=model,
                            harness=harness,
                            setting=setting,
                            task_id=step_task_id,
                            record={**step, "score": score, "grade": grade},
                            summary_path=path,
                            max_steps_extra=payload.get("max_steps_extra"),
                            status=str(result.get("status") or ""),
                            returncode=result.get("returncode"),
                        )
                    )
                continue
            score = result.get("score") if isinstance(result.get("score"), dict) else {}
            if score:
                rows.append(
                    row_from_scored_record(
                        stage=stage,
                        model=model,
                        harness=harness,
                        setting=setting,
                        task_id=task_id,
                        record=result,
                        summary_path=path,
                        max_steps_extra=payload.get("max_steps_extra"),
                        status=str(result.get("status") or ""),
                        returncode=result.get("returncode"),
                    )
                )
        return rows
    records = payload.get("results") or payload.get("tasks") or []
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        task_id = str(record.get("task_id") or record.get("conversation_task_id") or "")
        if not task_id:
            continue
        rows.append(
            row_from_scored_record(
                stage=stage,
                model=model,
                harness=harness,
                setting=setting,
                task_id=task_id,
                record=record,
                summary_path=path,
                max_steps_extra=payload.get("max_steps_extra"),
                status=str(record.get("status") or ""),
                returncode=record.get("returncode"),
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_") or "unknown"


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["model"], row["harness"], row["setting"])].append(row)
    out = []
    for key, items in sorted(groups.items()):
        stage, model, harness, setting = key
        n = len(items)
        er_values = [float(item["er"]) for item in items if item["er"] is not None]
        out.append(
            {
                "stage": stage,
                "model": model,
                "harness": harness,
                "setting": setting,
                "n": n,
                "sr": sum(float(item["success"]) for item in items) / max(1, n),
                "er": sum(er_values) / len(er_values) if er_values else None,
                "missing_budget_fields": sum(
                    1
                    for item in items
                    if item["shortest_estimate"] is None
                    or (bool(item["success"]) and item["success_step"] is None)
                ),
            }
        )
    return out


def budget_rows(rows: list[dict[str, Any]], max_extra: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["model"], row["harness"], row["setting"])].append(row)
    out = []
    for key, items in sorted(groups.items()):
        stage, model, harness, setting = key
        usable = [item for item in items if item["shortest_estimate"] is not None]
        for extra in range(max_extra + 1):
            successes = 0
            for item in usable:
                budget = int(item["shortest_estimate"]) + extra
                success_step = item.get("success_step")
                successes += int(bool(item["success"]) and success_step is not None and int(success_step) <= budget)
            out.append(
                {
                    "stage": stage,
                    "model": model,
                    "harness": harness,
                    "setting": setting,
                    "extra_steps": extra,
                    "n": len(usable),
                    "sr": successes / max(1, len(usable)),
                    "missing_budget_fields": len(items) - len(usable),
                }
            )
    return out


def plot_sr_bars_svg(rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stage"]), str(row["model"]))].append(row)
    for (stage, model), items in sorted(groups.items()):
        items = sorted(items, key=lambda row: (str(row["harness"]), str(row["setting"])))
        width = 900
        row_height = 38
        top = 48
        left = 250
        bar_width = 500
        height = max(180, top + row_height * len(items) + 45)
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect width='100%' height='100%' fill='white'/>",
            (
                f"<text x='20' y='28' font-family='Arial' font-size='18' font-weight='700'>"
                f"{html.escape(stage)} / {html.escape(model)} Success Rate</text>"
            ),
        ]
        for index, row in enumerate(items):
            y = top + index * row_height
            sr = max(0.0, min(1.0, float(row["sr"])))
            label = f"{row['harness']} / {row['setting']} (n={row['n']})"
            svg.append(f"<text x='20' y='{y + 23}' font-family='Arial' font-size='13'>{html.escape(label)}</text>")
            svg.append(f"<rect x='{left}' y='{y}' width='{bar_width}' height='24' fill='#eeeeee'/>")
            svg.append(f"<rect x='{left}' y='{y}' width='{bar_width * sr:.1f}' height='24' fill='#4c78a8'/>")
            svg.append(f"<text x='{left + bar_width + 15}' y='{y + 18}' font-family='Arial' font-size='13'>{sr * 100:.1f}%</text>")
        svg.append("</svg>\n")
        path = output_dir / f"sr_{safe_stem(stage)}_{safe_stem(model)}.svg"
        path.write_text("\n".join(svg), encoding="utf-8")


def plot_budget_curves_svg(rows: list[dict[str, Any]], output_dir: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["stage"]), str(row["model"]))].append(row)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for (stage, model), items in sorted(groups.items()):
        settings = sorted({str(row["setting"]) for row in items})
        if not settings:
            continue
        width = 900
        panel_height = 175
        left = 70
        top = 52
        plot_width = 700
        plot_height = 105
        height = top + panel_height * len(settings) + 45
        max_extra = max(int(row["extra_steps"]) for row in items)
        svg = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
            "<rect width='100%' height='100%' fill='white'/>",
            (
                f"<text x='20' y='28' font-family='Arial' font-size='18' font-weight='700'>"
                f"{html.escape(stage)} / {html.escape(model)} Budget Curve</text>"
            ),
        ]
        for panel_index, setting in enumerate(settings):
            origin_y = top + panel_index * panel_height
            setting_rows = [row for row in items if str(row["setting"]) == setting]
            harnesses = sorted({str(row["harness"]) for row in setting_rows})
            svg.append(f"<text x='20' y='{origin_y - 8}' font-family='Arial' font-size='14' font-weight='700'>{html.escape(setting)}</text>")
            svg.append(f"<line x1='{left}' y1='{origin_y + plot_height}' x2='{left + plot_width}' y2='{origin_y + plot_height}' stroke='#333'/>")
            svg.append(f"<line x1='{left}' y1='{origin_y}' x2='{left}' y2='{origin_y + plot_height}' stroke='#333'/>")
            for y_value in (0.0, 0.5, 1.0):
                y = origin_y + plot_height * (1.0 - y_value)
                svg.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_width}' y2='{y:.1f}' stroke='#dddddd'/>")
                svg.append(f"<text x='28' y='{y + 4:.1f}' font-family='Arial' font-size='11'>{y_value:.1f}</text>")
            for x_value in range(max_extra + 1):
                x = left + (plot_width * x_value / max(1, max_extra))
                svg.append(f"<text x='{x - 4:.1f}' y='{origin_y + plot_height + 18}' font-family='Arial' font-size='11'>{x_value}</text>")
            for harness_index, harness in enumerate(harnesses):
                line_rows = sorted(
                    [row for row in setting_rows if str(row["harness"]) == harness],
                    key=lambda row: int(row["extra_steps"]),
                )
                points = []
                for row in line_rows:
                    x_value = int(row["extra_steps"])
                    sr = max(0.0, min(1.0, float(row["sr"])))
                    x = left + (plot_width * x_value / max(1, max_extra))
                    y = origin_y + plot_height * (1.0 - sr)
                    points.append((x, y))
                if not points:
                    continue
                color = colors[harness_index % len(colors)]
                path_data = " ".join(
                    ("M" if index == 0 else "L") + f"{x:.1f},{y:.1f}"
                    for index, (x, y) in enumerate(points)
                )
                svg.append(f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='3'/>")
                for x, y in points:
                    svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.5' fill='{color}'/>")
                legend_x = left + harness_index * 140
                legend_y = origin_y + plot_height + 42
                svg.append(f"<rect x='{legend_x}' y='{legend_y - 9}' width='14' height='4' fill='{color}'/>")
                svg.append(f"<text x='{legend_x + 20}' y='{legend_y - 5}' font-family='Arial' font-size='11'>{html.escape(harness)}</text>")
        svg.append(
            f"<text x='{left + 210}' y='{height - 12}' font-family='Arial' font-size='12'>"
            "extra environment actions beyond shortest path</text>"
        )
        svg.append("</svg>\n")
        path = output_dir / f"budget_curve_{safe_stem(stage)}_{safe_stem(model)}.svg"
        path.write_text("\n".join(svg), encoding="utf-8")


def plot_budget_curves(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    by_stage_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stage_model[(str(row["stage"]), str(row["model"]))].append(row)
    for (stage, model), model_rows in by_stage_model.items():
        settings = sorted({row["setting"] for row in model_rows})
        if not settings:
            continue
        fig, axes = plt.subplots(len(settings), 1, figsize=(8, max(3, 2.4 * len(settings))), squeeze=False)
        for ax, setting in zip(axes[:, 0], settings):
            setting_rows = [row for row in model_rows if row["setting"] == setting]
            for harness in sorted({row["harness"] for row in setting_rows}):
                line_rows = sorted([row for row in setting_rows if row["harness"] == harness], key=lambda item: int(item["extra_steps"]))
                ax.plot([int(row["extra_steps"]) for row in line_rows], [float(row["sr"]) for row in line_rows], marker="o", label=harness)
            ax.set_title(f"{stage} / {model} / {setting}")
            ax.set_xlabel("extra environment actions beyond shortest path")
            ax.set_ylabel("success rate")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"budget_curve_{safe_stem(stage)}_{safe_stem(model)}.png", dpi=180)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize For_user L2/L4 results.")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument("--extra-steps", type=int, default=3)
    parser.add_argument(
        "--stages",
        default="stage0_smoke,stage2_20pct,stage3_full",
        help="Comma-separated stage directories to summarize. Empty means scan all output.",
    )
    args = parser.parse_args()
    stages = {part.strip() for part in args.stages.replace(",", " ").split() if part.strip()}

    rows: list[dict[str, Any]] = []
    for path in iter_summary_files(args.output_root, stages):
        try:
            rows.extend(rows_from_summary(path, args.output_root))
        except Exception as exc:
            rows.append(
                {
                    "stage": "parse_error",
                    "model": "unknown",
                    "harness": "unknown",
                    "setting": "unknown",
                    "task_id": "",
                    "success": 0,
                    "er": 0,
                    "max_steps": None,
                    "max_steps_extra": None,
                    "shortest_estimate": None,
                    "env_action_count": None,
                    "success_step": None,
                    "status": f"parse_error: {exc}",
                    "returncode": None,
                    "summary_path": str(path),
                }
            )
    args.report_root.mkdir(parents=True, exist_ok=True)
    # Remove artifacts produced by older versions of the summarizer. The
    # current public report contains only the fixed extra-3-step SR outputs.
    for stale in (args.report_root / "budget_curves.csv",):
        stale.unlink(missing_ok=True)
    for stale in args.report_root.glob("budget_curve_*"):
        if stale.is_file():
            stale.unlink()
    report_rows = []
    for row in rows:
        item = {key: row.get(key) for key in ("stage", "model", "harness", "setting", "task_id")}
        shortest = row.get("shortest_estimate")
        success_step = row.get("success_step")
        item["success_at_extra_steps"] = int(
            bool(row.get("success"))
            and shortest is not None
            and success_step is not None
            and int(success_step) <= int(shortest) + args.extra_steps
        )
        report_rows.append(item)
    write_csv(args.report_root / "task_rows.csv", report_rows)
    budget = budget_rows(rows, args.extra_steps)
    selected = [row for row in budget if int(row["extra_steps"]) == args.extra_steps]
    agg = [
        {
            "stage": row["stage"],
            "model": row["model"],
            "harness": row["harness"],
            "setting": row["setting"],
            "n": row["n"],
            "sr": row["sr"],
        }
        for row in selected
    ]
    write_csv(args.report_root / "main_table.csv", agg)
    plot_sr_bars_svg(agg, args.report_root)
    print(json.dumps({"task_rows": len(report_rows), "groups": len(agg), "extra_steps": args.extra_steps, "report_root": str(args.report_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
