#!/usr/bin/env python3
"""Summarize N-sample scores and render SR/ER curves as dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "analyze/N-sample/input/subset_10pct_child"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "analyze/N-sample/output/full"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "analyze/N-sample/reports/full"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def expected_task_ids(input_root: Path, phase: str) -> list[str]:
    payload = read_json(input_root / "ready_l4_tasks_with_images.json")
    ids = [str(item.get("task_id") or "") for item in payload if isinstance(item, dict)]
    return [value for value in ids if value][:1] if phase == "smoke" else [value for value in ids if value]


def score_row(
    *,
    payload: dict[str, Any],
    task_id: str,
    model: str,
    harness: str,
    k: int,
    mode: str,
    score_path: Path,
    expected_count: int,
) -> dict[str, Any]:
    details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
    reached = details.get("reached_goal")
    success = int(bool(reached))
    return {
        "model": model,
        "harness": harness,
        "k": k,
        "mode": mode,
        "task_id": task_id,
        "success": success,
        "er": float(details.get("efficiency_score") or 0.0),
        "pq": float(details.get("path_quality_score") or 0.0),
        "ip": float(details.get("invalid_penalty") or 0.0),
        "error_type": str(details.get("error_type") or ""),
        "score_path": str(score_path),
        "expected_count": expected_count,
    }


def parse_condition_path(run_root: Path, score_path: Path) -> tuple[str, str, int, str] | None:
    rel = score_path.relative_to(run_root)
    parts = rel.parts
    if len(parts) < 5:
        return None
    model, harness, k_text, mode = parts[:4]
    if not k_text.startswith("k") or mode not in {"zero_shot", "in_context", "skill"}:
        return None
    try:
        k = int(k_text[1:])
    except ValueError:
        return None
    return model, harness, k, mode


def load_rows(run_root: Path, expected_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for score_path in sorted(run_root.rglob("score.json")):
        condition = parse_condition_path(run_root, score_path)
        if condition is None:
            continue
        model, harness, k, mode = condition
        try:
            payload = read_json(score_path)
        except (OSError, json.JSONDecodeError):
            continue
        task_id = str(payload.get("task_id") or "").strip()
        if task_id not in expected_ids or not task_id.startswith("task_L4_"):
            continue
        key = (model, harness, k, mode, task_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            score_row(
                payload=payload,
                task_id=task_id,
                model=model,
                harness=harness,
                k=k,
                mode=mode,
                score_path=score_path,
                expected_count=len(expected_ids),
            )
        )
    return rows


def aggregate(rows: list[dict[str, Any]], expected_count: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["harness"], int(row["k"]), row["mode"])].append(row)
    output: list[dict[str, Any]] = []
    for (model, harness, k, mode), items in sorted(groups.items()):
        observed = len(items)
        output.append(
            {
                "model": model,
                "harness": harness,
                "k": k,
                "mode": mode,
                "expected_n": expected_count,
                "observed_n": observed,
                "missing_n": expected_count - observed,
                "sr": sum(int(item["success"]) for item in items) / max(1, observed),
                "er": sum(float(item["er"]) for item in items) / max(1, observed),
                "pq": sum(float(item["pq"]) for item in items) / max(1, observed),
                "ip": sum(float(item["ip"]) for item in items) / max(1, observed),
                "status": "complete" if observed == expected_count else "incomplete",
            }
        )
    return output


def curve_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [dict(row) for row in aggregate_rows if int(row["k"]) > 0]
    baseline = [row for row in aggregate_rows if int(row["k"]) == 0 and row["mode"] == "zero_shot"]
    for row in baseline:
        for mode in ("in_context", "skill"):
            output.append({**row, "mode": mode, "baseline_mode": "zero_shot"})
    return sorted(output, key=lambda row: (row["model"], row["harness"], row["mode"], int(row["k"])))


def marginal_rows(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in curves:
        groups[(row["model"], row["harness"], row["mode"])][int(row["k"])] = row
    output: list[dict[str, Any]] = []
    for (model, harness, mode), values in sorted(groups.items()):
        previous: dict[str, Any] | None = None
        for k in sorted(values):
            row = values[k]
            output.append(
                {
                    "model": model,
                    "harness": harness,
                    "mode": mode,
                    "from_k": "" if previous is None else int(previous["k"]),
                    "to_k": k,
                    "sr": row["sr"],
                    "delta_sr": "" if previous is None else float(row["sr"]) - float(previous["sr"]),
                    "er": row["er"],
                    "delta_er": "" if previous is None else float(row["er"]) - float(previous["er"]),
                    "status": row["status"],
                }
            )
            previous = row
    return output


def plot_svg(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["observed_n"]) > 0:
            groups[(row["model"], row["harness"], row["mode"])].append(row)
    width, height = 920, 560
    left, right, top, bottom = 78, 220, 62, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    svg: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{left}' y='30' font-family='Arial' font-size='20' font-weight='700'>N-sample {metric.upper()} curve</text>",
        f"<text x='{left}' y='{height - 18}' font-family='Arial' font-size='14'>Learning samples (k)</text>",
        f"<text x='18' y='{top + plot_height / 2}' transform='rotate(-90 18 {top + plot_height / 2})' font-family='Arial' font-size='14'>{metric.upper()}</text>",
    ]
    for tick in range(6):
        x = left + plot_width * tick / 5
        svg.append(f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_height}' stroke='#eeeeee'/>")
        svg.append(f"<text x='{x:.1f}' y='{top + plot_height + 24}' text-anchor='middle' font-family='Arial' font-size='12'>{tick if tick <= 4 else ''}</text>")
    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        svg.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_width}' y2='{y:.1f}' stroke='#eeeeee'/>")
        svg.append(f"<text x='{left - 12}' y='{y + 4:.1f}' text-anchor='end' font-family='Arial' font-size='12'>{value:.1f}</text>")
    svg.append(f"<line x1='{left}' y1='{top + plot_height}' x2='{left + plot_width}' y2='{top + plot_height}' stroke='#333333'/>")
    svg.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' stroke='#333333'/>")

    for index, (group, group_rows) in enumerate(sorted(groups.items())):
        model, harness, mode = group
        group_rows = sorted(group_rows, key=lambda row: int(row["k"]))
        color = colors[index % len(colors)]
        points: list[str] = []
        for row in group_rows:
            k = int(row["k"])
            value = max(0.0, min(1.0, float(row[metric])))
            x = left + plot_width * k / 4
            y = top + plot_height * (1 - value)
            points.append(f"{x:.1f},{y:.1f}")
        dash = " stroke-dasharray='8,5'" if mode == "skill" else ""
        if len(points) >= 2:
            svg.append(f"<polyline points='{' '.join(points)}' fill='none' stroke='{color}' stroke-width='3'{dash}/>")
        for row in group_rows:
            k = int(row["k"])
            value = max(0.0, min(1.0, float(row[metric])))
            x = left + plot_width * k / 4
            y = top + plot_height * (1 - value)
            svg.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5' fill='{color}'/>")

    legend_x = left + plot_width + 28
    legend_y = top + 12
    for index, (model, harness, mode) in enumerate(sorted(groups)):
        color = colors[index % len(colors)]
        y = legend_y + index * 30
        dash = " stroke-dasharray='8,5'" if mode == "skill" else ""
        svg.append(f"<line x1='{legend_x}' y1='{y}' x2='{legend_x + 26}' y2='{y}' stroke='{color}' stroke-width='3'{dash}/>")
        svg.append(f"<text x='{legend_x + 34}' y='{y + 4}' font-family='Arial' font-size='12'>{html.escape(model + ' / ' + harness + ' / ' + mode)}</text>")
    svg.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")


def markdown_report(
    *,
    path: Path,
    phase: str,
    expected_ids: list[str],
    aggregate_rows: list[dict[str, Any]],
    marginal: list[dict[str, Any]],
) -> None:
    lines = [
        "# N-sample 结果汇总",
        "",
        f"- 阶段：`{phase}`",
        f"- 固定 child L4 测试任务数：`{len(expected_ids)}`",
        "- 模型 / Harness：Qwen3.7-Plus / HermesAgent",
        "- `k=0` 是 zero-shot baseline；ICL 和 Skill 曲线共享该起点。",
        "- SR/ER 按已产生 score 的 target task 计算；若 observed 数量不足，标记为 `incomplete`，不把缺失结果当作模型失败。",
        "",
        "## 曲线数据",
        "",
        "| k | mode | observed/expected | SR | ER | PQ | IP | status |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    curves = curve_rows(aggregate_rows)
    for row in curves:
        lines.append(
            f"| {row['k']} | {row['mode']} | {row['observed_n']}/{row['expected_n']} | "
            f"{float(row['sr']):.4f} | {float(row['er']):.4f} | {float(row['pq']):.4f} | {float(row['ip']):.4f} | {row['status']} |"
        )
    lines.extend(["", "## 边际收益", "", "| mode | from k | to k | delta SR | delta ER | status |", "|---|---:|---:|---:|---:|---|"])
    for row in marginal:
        delta_sr = "" if row["delta_sr"] == "" else f"{float(row['delta_sr']):.4f}"
        delta_er = "" if row["delta_er"] == "" else f"{float(row['delta_er']):.4f}"
        lines.append(f"| {row['mode']} | {row['from_k']} | {row['to_k']} | {delta_sr} | {delta_er} | {row['status']} |")
    lines.extend(["", "## 固定测试任务", "", *[f"- `{task_id}`" for task_id in expected_ids], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize N-sample experiment outputs.")
    parser.add_argument("--phase", choices=("smoke", "full"), default="full")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--report-root", type=Path, default=None)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    run_root = (args.run_root or (PROJECT_ROOT / "analyze/N-sample/output" / args.phase)).resolve()
    report_root = (args.report_root or (PROJECT_ROOT / "analyze/N-sample/reports" / args.phase)).resolve()
    expected_ids = expected_task_ids(input_root, args.phase)
    rows = load_rows(run_root, set(expected_ids)) if run_root.exists() else []
    aggregate_rows = aggregate(rows, len(expected_ids))
    curves = curve_rows(aggregate_rows)
    marginal = marginal_rows(curves)
    write_csv(report_root / "task_metrics.csv", rows)
    write_csv(report_root / "aggregate_metrics.csv", aggregate_rows)
    write_csv(report_root / "curve_metrics.csv", curves)
    write_csv(report_root / "marginal_sr.csv", marginal)
    plot_svg(curves, "sr", report_root / "n_sample_sr.svg")
    plot_svg(curves, "er", report_root / "n_sample_er.svg")
    markdown_report(
        path=report_root / "n_sample_summary.md",
        phase=args.phase,
        expected_ids=expected_ids,
        aggregate_rows=aggregate_rows,
        marginal=marginal,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "run_root": str(run_root),
                "report_root": str(report_root),
                "expected_tasks": len(expected_ids),
                "observed_scores": len(rows),
                "aggregate_rows": len(aggregate_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
