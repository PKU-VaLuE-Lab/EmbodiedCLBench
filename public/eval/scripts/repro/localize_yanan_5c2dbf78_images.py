#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_PACKAGE_REL = Path("important_results_eval/8.10_remote_yanan_5c2dbf78")
DEFAULT_IMAGE_REUSE_REL = Path("input/036_02/generated_from_imaged_l1/image_reuse")
DEFAULT_SOURCE_RELS = (
    Path("important_results_eval/8.10/input/image_reuse_l2_only"),
    Path("important_results_eval/8.10/input/image_reuse_l3_partial"),
    Path("important_results_eval/8.10/input/taskgen_output_036_02/image_reuse_l2_only"),
    Path("important_results_eval/8.10/input/taskgen_output_036_02/image_reuse_l3_partial"),
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _image_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _ready_items(ready_json: Path) -> list[dict[str, Any]]:
    payload = _read_json(ready_json)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("tasks", "ready_tasks", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Unsupported ready JSON structure: {ready_json}")


def _level_of(task_id: str) -> str:
    if task_id.startswith("task_L1_"):
        return "L1"
    if task_id.startswith("task_L2_"):
        return "L2"
    if task_id.startswith("task_L3_"):
        return "L3"
    return ""


def _component_l1_ids(item: dict[str, Any]) -> list[str]:
    sources = item.get("component_sources") if isinstance(item.get("component_sources"), dict) else {}
    raw_ids = sources.get("mapped_source_l1_task_ids") or []
    return [str(task_id).strip() for task_id in raw_ids if str(task_id).strip()]


def _select_composite_items(args: argparse.Namespace, image_reuse_root: Path) -> list[dict[str, Any]]:
    ready_json = image_reuse_root / "ready_l2_l3_tasks_with_images.json"
    items = _ready_items(ready_json)
    explicit_ids = set(args.tasks or [])
    if explicit_ids:
        selected = [item for item in items if str(item.get("task_id") or "") in explicit_ids]
        missing = sorted(explicit_ids - {str(item.get("task_id") or "") for item in selected})
        if missing:
            raise SystemExit(f"Requested tasks are missing from ready JSON: {', '.join(missing)}")
        return selected

    selected: list[dict[str, Any]] = []
    levels = {level.upper() for level in args.levels}
    limits = {"L2": args.l2_task_limit, "L3": args.l3_task_limit}
    for level in ("L2", "L3"):
        if level not in levels:
            continue
        limit = limits[level]
        level_items = [item for item in items if _level_of(str(item.get("task_id") or "")) == level]
        selected.extend(level_items if limit is None else level_items[: int(limit)])
    return selected


def _target_task_ids(items: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        composite_id = str(item.get("task_id") or "").strip()
        for task_id in [*_component_l1_ids(item), composite_id]:
            if task_id and task_id not in seen:
                seen.add(task_id)
                ordered.append(task_id)
    return ordered


def _expected_state_count(image_reuse_root: Path, task_id: str) -> int | None:
    unique_states = image_reuse_root / "tasks" / task_id / "input" / f"{task_id}_unique_states.json"
    if not unique_states.exists():
        return None
    payload = _read_json(unique_states)
    states = payload.get("states", []) if isinstance(payload, dict) else []
    return len([item for item in states if isinstance(item, dict)])


def _source_output_dir(source_roots: list[Path], task_id: str) -> Path | None:
    for source_root in source_roots:
        candidate = source_root / "tasks" / task_id / "output"
        unique_states = candidate / f"{task_id}_unique_states"
        if unique_states.exists() and _image_count(unique_states) > 0:
            return candidate
    return None


def _copy_task_output(
    *,
    source_output: Path,
    destination_output: Path,
    force: bool,
    dry_run: bool,
) -> str:
    if destination_output.exists() and _image_count(destination_output) > 0 and not force:
        return "kept_existing"
    if dry_run:
        return "would_copy"
    if destination_output.exists():
        shutil.rmtree(destination_output)
    shutil.copytree(source_output, destination_output)
    return "copied"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy missing unique-state images into the remote Yanan 5c2dbf78 package. "
            "Only image outputs are copied; graph JSON stays from the remote input."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--image-reuse-root", type=Path)
    parser.add_argument("--source-root", action="append", type=Path, default=[])
    parser.add_argument("--levels", nargs="+", default=["L2", "L3"])
    parser.add_argument("--l2-task-limit", type=int, default=2)
    parser.add_argument("--l3-task-limit", type=int, default=2)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_root = args.project_root.resolve()
    package_root = (args.package_root or project_root / DEFAULT_PACKAGE_REL).resolve()
    image_reuse_root = (args.image_reuse_root or package_root / DEFAULT_IMAGE_REUSE_REL).resolve()
    source_roots = [
        (path if path.is_absolute() else project_root / path).resolve()
        for path in (args.source_root or list(DEFAULT_SOURCE_RELS))
    ]
    report_path = (args.report_path or package_root / "IMAGE_LOCALIZATION_REPORT.json").resolve()

    if not image_reuse_root.exists():
        raise SystemExit(f"Image reuse root does not exist: {image_reuse_root}")

    selected_items = _select_composite_items(args, image_reuse_root)
    target_ids = _target_task_ids(selected_items)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for task_id in target_ids:
        source_output = _source_output_dir(source_roots, task_id)
        destination_output = image_reuse_root / "tasks" / task_id / "output"
        expected_count = _expected_state_count(image_reuse_root, task_id)
        before_count = _image_count(destination_output)
        status = "missing_source"
        if source_output is None:
            failures.append(task_id)
        else:
            status = _copy_task_output(
                source_output=source_output,
                destination_output=destination_output,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
        after_count = before_count if args.dry_run else _image_count(destination_output)
        count_ok = expected_count is None or after_count == expected_count
        if not count_ok:
            failures.append(task_id)
        rows.append(
            {
                "task_id": task_id,
                "level": _level_of(task_id),
                "source_output": str(source_output) if source_output else None,
                "destination_output": str(destination_output),
                "expected_state_count": expected_count,
                "image_count_before": before_count,
                "image_count_after": after_count,
                "image_count_matches_states": count_ok,
                "status": status,
            }
        )

    report = {
        "package_root": str(package_root),
        "image_reuse_root": str(image_reuse_root),
        "source_roots": [str(path) for path in source_roots],
        "selected_composite_task_ids": [str(item.get("task_id") or "") for item in selected_items],
        "localized_task_ids": target_ids,
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "tasks": rows,
        "ok": not failures,
        "failures": sorted(set(failures)),
    }
    _write_json(report_path, report)
    print(f"Wrote image localization report: {report_path}")
    print(f"Localized/check tasks: {len(rows)}; ok={report['ok']}")
    if failures:
        print("Failures: " + ", ".join(sorted(set(failures))))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
