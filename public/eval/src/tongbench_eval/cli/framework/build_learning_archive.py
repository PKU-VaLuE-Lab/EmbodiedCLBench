#!/usr/bin/env python3
"""Merge independently evaluated L2 sessions into reusable native archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from tongbench_eval.agents.base import AgentCheckpoint
from tongbench_eval.agents.checkpoint_merge import merge_checkpoints


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slug(value: str) -> str:
    characters = [character.lower() if character.isalnum() else "_" for character in str(value)]
    return "".join(characters).strip("_") or "item"


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _source_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_entries = payload.get("l2_tasks") or payload.get("source_tasks") or payload.get("entries")
    if isinstance(raw_entries, dict):
        raw_entries = [dict(value, task_id=key) for key, value in raw_entries.items() if isinstance(value, dict)]
    entries: dict[str, dict[str, Any]] = {}
    for raw in _list_or_empty(raw_entries):
        if not isinstance(raw, dict):
            continue
        task_id = str(raw.get("task_id") or raw.get("learning_task_id") or "").strip()
        checkpoint_dir = str(
            raw.get("checkpoint_dir")
            or raw.get("source_checkpoint_dir")
            or raw.get("path")
            or ""
        ).strip()
        if task_id and checkpoint_dir:
            entries[task_id] = {**raw, "task_id": task_id, "checkpoint_dir": checkpoint_dir}
    return entries


def _target_entries(payload: dict[str, Any], source_ids: list[str]) -> list[dict[str, Any]]:
    raw_targets = payload.get("targets") or payload.get("learning_sets")
    if isinstance(raw_targets, dict):
        raw_targets = [dict(value, target_task_id=key) for key, value in raw_targets.items() if isinstance(value, dict)]
    targets: list[dict[str, Any]] = []
    for raw in _list_or_empty(raw_targets):
        if not isinstance(raw, dict):
            continue
        target_id = str(raw.get("target_task_id") or raw.get("task_id") or "").strip()
        learning_ids = raw.get("learning_task_ids") or raw.get("icl_learning_task_ids") or []
        learning_ids = [str(value).strip() for value in learning_ids if str(value).strip()]
        if target_id and learning_ids:
            targets.append({**raw, "target_task_id": target_id, "learning_task_ids": learning_ids})
    if targets:
        return targets
    return [
        {"target_task_id": task_id, "learning_task_ids": [task_id]}
        for task_id in source_ids
    ]


def _copy_source_checkpoints(
    sources: dict[str, dict[str, Any]],
    output_root: Path,
    *,
    force: bool,
) -> dict[str, dict[str, Any]]:
    """Copy source archives so the resulting learning bundle is portable."""
    source_root = output_root / "source_checkpoints"
    if source_root.exists():
        if not force:
            raise FileExistsError(f"Source archive directory exists: {source_root}")
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)

    copied: dict[str, dict[str, Any]] = {}
    for task_id, source in sources.items():
        original = Path(source["checkpoint_dir"]).expanduser().resolve()
        if not original.is_dir():
            raise FileNotFoundError(f"Missing source checkpoint for {task_id}: {original}")
        destination = source_root / slug(task_id)
        shutil.copytree(original, destination)
        copied[task_id] = {
            **source,
            "task_id": task_id,
            "checkpoint_dir": str(destination.resolve()),
            "original_checkpoint_dir": str(original),
        }
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--backend", help="Override the backend recorded in the source manifest.")
    parser.add_argument("--model", help="Override the model recorded in the source manifest.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_manifest_path = args.source_manifest.expanduser().resolve()
    source_payload = read_json(source_manifest_path)
    if not isinstance(source_payload, dict):
        raise ValueError("Source manifest must be a JSON object")
    sources = _source_entries(source_payload)
    if not sources:
        raise ValueError("Source manifest has no checkpoint entries")
    backend = str(args.backend or source_payload.get("backend") or "").strip()
    model = str(args.model or source_payload.get("model") or "").strip()
    if not backend:
        raise ValueError("Source manifest must define backend")
    targets = _target_entries(source_payload, sorted(sources))

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    local_source_manifest = output_root / "source_manifest.json"
    if local_source_manifest.exists() and not args.force:
        raise FileExistsError(f"Archive source manifest exists: {local_source_manifest}")
    shutil.copy2(source_manifest_path, local_source_manifest)
    sources = _copy_source_checkpoints(sources, output_root, force=args.force)
    merged_root = output_root / "merged_checkpoints" / slug(backend) / slug(model or "model")
    reuse_entries: list[dict[str, Any]] = []
    archive_targets: list[dict[str, Any]] = []
    for target in targets:
        target_id = target["target_task_id"]
        learning_ids = target["learning_task_ids"]
        missing = [task_id for task_id in learning_ids if task_id not in sources]
        if missing:
            raise KeyError(f"{target_id} references missing L2 checkpoints: {missing}")
        checkpoints = [
            AgentCheckpoint.load(
                Path(sources[task_id]["checkpoint_dir"]).expanduser().resolve(),
                expected_backend=backend,
            )
            for task_id in learning_ids
        ]
        target_root = merged_root / slug(target_id)
        checkpoint_dir = target_root / "checkpoint"
        summary_path = target_root / "learning_summary.json"
        if target_root.exists():
            if not args.force:
                raise FileExistsError(f"Archive target exists: {target_root}")
            shutil.rmtree(target_root)
        checkpoint = merge_checkpoints(
            checkpoints,
            checkpoint_dir,
            metadata={
                "target_task_id": target_id,
                "learning_task_ids": learning_ids,
                "source_manifest": str(local_source_manifest.resolve()),
                "model": model,
            },
        )
        learning_summary = {
            "schema_version": "tongbench_independent_learning_summary_v1",
            "phase": "shared_learning",
            "target_task_id": target_id,
            "learning_task_ids": learning_ids,
            "source_mode": "independent_l2_zero_shot_archive",
            "steps": [],
            "task_outcomes": list(checkpoint.metadata.get("task_outcomes") or []),
            "source_checkpoints": [
                {
                    "task_id": task_id,
                    "checkpoint_dir": str(Path(sources[task_id]["checkpoint_dir"]).expanduser().resolve()),
                    "payload_sha256": checkpoints[index].payload_sha256,
                }
                for index, task_id in enumerate(learning_ids)
            ],
        }
        write_json(summary_path, learning_summary)
        reuse_entries.append(
            {
                "target_task_id": target_id,
                "source_parent_task_id": target_id,
                "source_checkpoint_dir": str(checkpoint_dir.resolve()),
                "source_learning_summary_path": str(summary_path.resolve()),
                "learning_task_ids": learning_ids,
                "archive_checkpoint": checkpoint.as_dict(),
            }
        )
        archive_targets.append(
            {
                "target_task_id": target_id,
                "learning_task_ids": learning_ids,
                "checkpoint_dir": str(checkpoint_dir.resolve()),
                "learning_summary_path": str(summary_path.resolve()),
                "payload_sha256": checkpoint.payload_sha256,
            }
        )

    reuse_manifest = output_root / "learning_reuse_manifest.json"
    write_json(
        reuse_manifest,
        {
            "schema_version": "tongbench_learning_reuse_manifest_v2",
            "mode": "independent_l2_zero_shot_archive",
            "backend": backend,
            "model": model,
                "source_manifest": str(local_source_manifest.resolve()),
            "entries": reuse_entries,
        },
    )
    archive_manifest = {
        "schema_version": "tongbench_learning_archive_v1",
        "mode": "independent_l2_zero_shot_archive",
        "backend": backend,
        "model": model,
        "source_manifest": str(local_source_manifest.resolve()),
        "source_entries": list(sources.values()),
        "targets": archive_targets,
        "reuse_manifest": str(reuse_manifest.resolve()),
    }
    write_json(output_root / "archive_manifest.json", archive_manifest)
    print(json.dumps({"ok": True, "backend": backend, "model": model, "target_count": len(archive_targets), "reuse_manifest": str(reuse_manifest)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
