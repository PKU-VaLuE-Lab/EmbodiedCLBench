#!/usr/bin/env python3
"""Prepare analysis evaluation inputs without generating tasks or options.

This adapter only indexes already generated artifacts, patches the model-facing
description into a private graph copy, and builds native learning manifests
from already exported L2 checkpoints. It never calls a task/option API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_backend_ks(values: list[str]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    for value in values:
        backend, separator, raw_ks = value.partition(":")
        backend = backend.strip()
        if not separator or not backend:
            raise ValueError(f"Expected BACKEND:K,K for --n-sample-backend-ks, got {value!r}")
        ks = tuple(sorted({int(item.strip()) for item in raw_ks.split(",") if item.strip()}))
        if not ks or any(k < 1 or k > 6 for k in ks):
            raise ValueError(f"N-sample k values must be in 1..6, got {value!r}")
        result[backend] = ks
    return result


def signature(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        for key in ("state", "target_state", "predicates"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list):
        return ()
    return tuple(sorted(str(item) for item in value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_dir(raw: str | Path) -> Path:
    path = Path(raw).expanduser().resolve()
    if (path / "agent_checkpoint.json").is_file():
        return path
    nested = path / "checkpoint"
    if (nested / "agent_checkpoint.json").is_file():
        return nested
    raise FileNotFoundError(f"Missing agent_checkpoint.json under {path}")


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def load_l2_sources(repo_root: Path, backend: str = "hermesagent") -> dict[str, str]:
    manifest_paths = [
        repo_root / f"For_user/output/stage2_20pct_qwen38_all_harness_formal_20260905/stage2_20pct/_learning_archive_sources/qwen/{backend}/source_manifest.json",
        repo_root / f"For_user/output/stage3_remaining_80pct_qwen38_formal_20260907/stage3_full/_learning_archive_sources/qwen/{backend}/source_manifest.json",
    ]
    sources: dict[str, str] = {}
    for manifest_path in manifest_paths:
        payload = read_json(manifest_path)
        for row in payload.get("l2_tasks", []):
            task_id = str(row.get("task_id") or "").strip()
            raw_path = str(row.get("checkpoint_dir") or "").strip()
            if not task_id or not raw_path:
                continue
            sources[task_id] = str(checkpoint_dir(raw_path))
    if not sources:
        raise ValueError(f"No exported {backend}/Qwen L2 checkpoints were found")
    return sources


def load_main_learning_entries(repo_root: Path) -> dict[str, dict[str, Any]]:
    manifest_paths = [
        repo_root / "For_user/output/stage2_20pct_qwen38_all_harness_formal_20260905/stage2_20pct/_learning_archives/qwen/hermesagent/learning_reuse_manifest.json",
        repo_root / "For_user/output/stage3_remaining_80pct_qwen38_formal_20260907/stage3_full/_learning_archives/qwen/hermesagent/learning_reuse_manifest.json",
    ]
    entries: dict[str, dict[str, Any]] = {}
    for manifest_path in manifest_paths:
        payload = read_json(manifest_path)
        for row in payload.get("entries", []):
            target_id = str(row.get("target_task_id") or "").strip()
            if not target_id:
                continue
            source_path = checkpoint_dir(row.get("source_checkpoint_dir") or "")
            summary_path = Path(str(row.get("source_learning_summary_path") or "")).expanduser().resolve()
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing learning summary for {target_id}: {summary_path}")
            entries[target_id] = {
                **row,
                "source_checkpoint_dir": str(source_path),
                "source_learning_summary_path": str(summary_path),
                "source_manifest": str(manifest_path.resolve()),
            }
    return entries


def make_source_manifest(
    path: Path,
    sources: dict[str, str],
    targets: list[dict[str, Any]],
    *,
    backend: str,
) -> None:
    write_json(
        path,
        {
            "schema_version": "tongbench_l2_zero_shot_source_manifest_v1",
            "mode": "independent_l2_zero_shot_archive",
            "backend": backend,
            "model": "qwen3.8-flash",
            "source_kind": "main_experiment_l2_zero_shot_exports",
            "l2_tasks": [
                {"task_id": task_id, "checkpoint_dir": checkpoint}
                for task_id, checkpoint in sorted(sources.items())
            ],
            "targets": targets,
        },
    )


def build_n_sample_archives(
    *,
    repo_root: Path,
    prepared_root: Path,
    source_index: dict[str, str],
    backend: str = "hermesagent",
    ks: tuple[int, ...] = (1, 3, 4),
    force: bool,
) -> dict[int, Path]:
    plan_root = repo_root / "For_user/analyze/N-sample/input/formal_child"
    archive_paths: dict[int, Path] = {}
    env = os.environ.copy()
    eval_src = repo_root / "public/eval/src"
    env["PYTHONPATH"] = str(eval_src) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ks:
        plan_path = plan_root / f"learning_plan_k{k}.json"
        plan = read_json(plan_path)
        targets = [
            {
                "target_task_id": str(row["target_task_id"]),
                "learning_task_ids": [str(item) for item in row["learning_task_ids"]],
            }
            for row in plan.get("targets", [])
        ]
        needed_ids = {task_id for row in targets for task_id in row["learning_task_ids"]}
        missing = sorted(needed_ids - set(source_index))
        if missing:
            raise KeyError(f"N-sample k={k} has missing L2 checkpoints: {missing}")
        archive_root = prepared_root / "learning_archives" / "n_sample" / backend / f"k{k}"
        archive_manifest = archive_root / "learning_reuse_manifest.json"
        if force and archive_root.exists():
            shutil.rmtree(archive_root)
        archive_root.mkdir(parents=True, exist_ok=True)
        if not archive_manifest.is_file():
            sources = {task_id: source_index[task_id] for task_id in sorted(needed_ids)}
            source_manifest = prepared_root / "source_manifests" / f"n_sample_{backend}_k{k}.json"
            make_source_manifest(source_manifest, sources, targets, backend=backend)
            command = [
                sys.executable,
                "-m",
                "tongbench_eval.cli.framework.build_learning_archive",
                "--source-manifest",
                str(source_manifest),
                "--output-root",
                str(archive_root),
                "--backend",
                backend,
                "--model",
                "qwen3.8-flash",
                "--force",
            ]
            result = subprocess.run(command, cwd=repo_root, env=env)
            if result.returncode:
                raise RuntimeError(f"Failed to build N-sample k={k} archive: exit {result.returncode}")
        archive_paths[k] = archive_manifest
    return archive_paths


def build_l_n_learning_manifest(
    *,
    prepared_root: Path,
    main_entries: dict[str, dict[str, Any]],
    ready_items: list[dict[str, Any]],
) -> Path:
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in ready_items:
        task_id = str(item.get("task_id") or "").strip()
        nested_task = item.get("task") if isinstance(item.get("task"), dict) else {}
        parent_id = str(nested_task.get("source_parent_task_id") or "").strip()
        if "_adult_" in parent_id:
            parent_id = parent_id.replace("_adult_", "_child_")
        source = main_entries.get(parent_id)
        if source is None:
            missing.append(f"{task_id} -> {parent_id}")
            continue
        entries.append(
            {
                "target_task_id": task_id,
                "source_parent_task_id": parent_id,
                "source_checkpoint_dir": source["source_checkpoint_dir"],
                "source_learning_summary_path": source["source_learning_summary_path"],
                "learning_task_ids": list(source.get("learning_task_ids") or []),
                "source_main_manifest": source["source_manifest"],
            }
        )
    if missing:
        raise KeyError(f"Missing main child-L4 learning archives for: {missing}")
    path = prepared_root / "learning_archives" / "l_n" / "learning_reuse_manifest.json"
    write_json(
        path,
        {
            "schema_version": "tongbench_learning_reuse_manifest_v2",
            "mode": "reuse_main_l4_child_learning_archive_for_l_n",
            "backend": "hermesagent",
            "model": "qwen3.8-flash",
            "source_kind": "main_experiment_l2_zero_shot_exports",
            "entries": entries,
        },
    )
    return path


def graph_source(repo_root: Path, level: str, task_id: str) -> Path:
    if level == "L3":
        return repo_root / "For_user/analyze/L_N/input/formal_v1/input/image_reuse/tasks" / task_id / "input" / f"{task_id}_atomic_expanded_graph.json"
    return repo_root / "For_user/analyze/L_N/input/formal_v2_tuned/input/taskgen_output/graph" / f"{task_id}_atomic_expanded_graph.json"


def success_source(repo_root: Path, level: str, task_id: str) -> Path:
    if level == "L3":
        return repo_root / "For_user/analyze/L_N/input/formal_v1/input/image_reuse/tasks" / task_id / "input" / f"{task_id}_success_paths.json"
    return repo_root / "For_user/analyze/L_N/input/formal_v2_tuned/input/taskgen_output/graph" / f"{task_id}_success_paths.json"


def source_image_index(item: dict[str, Any], ready_root: Path) -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    for state in item.get("states", []):
        if not isinstance(state, dict):
            continue
        raw_image = str(state.get("copied_image_path") or state.get("reused_image_path") or "").strip()
        if not raw_image:
            paths = state.get("source_image_paths") or []
            raw_image = str(paths[0]) if paths else ""
        if not raw_image:
            continue
        image = Path(raw_image).expanduser()
        if not image.is_absolute():
            image = (ready_root / image).resolve()
        if image.is_file():
            index.setdefault(signature(state.get("target_state") or state.get("state") or []), str(image))
    return index


def shortest_node_ids(graph: dict[str, Any], success: dict[str, Any]) -> list[str]:
    nodes_by_signature = {
        signature(node.get("state")): str(node.get("id"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    paths = [path for path in success.get("success_paths", []) if path.get("is_shortest_success_path")]
    if not paths:
        paths = list(success.get("success_paths", []))[:1]
    if not paths:
        return []
    steps = paths[0].get("atomic_steps") or []
    nodes = graph.get("nodes", [])
    result = [str(nodes[0].get("id") or "as0")] if nodes else []
    for step in steps:
        node_id = nodes_by_signature.get(signature(step.get("after_state")))
        if node_id and (not result or node_id != result[-1]):
            result.append(node_id)
    return result


def validate_option_bank(
    *,
    bank_path: Path,
    ready_items: list[dict[str, Any]],
    graph_for_task: dict[str, Path],
    strict_tasks: bool = True,
) -> dict[str, Any]:
    expected = {str(item.get("task_id") or "").strip() for item in ready_items}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for line_number, line in enumerate(bank_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid option-bank JSON at {bank_path}:{line_number}: {exc}") from exc
        task_id = str(row.get("task_id") or "").strip()
        if task_id:
            rows_by_task.setdefault(task_id, []).append(row)
    missing_tasks = sorted(expected - set(rows_by_task))
    unknown_tasks = sorted(set(rows_by_task) - expected)
    if missing_tasks or (strict_tasks and unknown_tasks):
        raise ValueError(f"Option bank task mismatch for {bank_path}: missing={missing_tasks[:5]}, unknown={unknown_tasks[:5]}")
    invalid_rows: list[str] = []
    for task_id in sorted(expected):
        graph = read_json(graph_for_task[task_id])
        node_ids = {str(node.get("id")) for node in graph.get("nodes", []) if isinstance(node, dict)}
        for row in rows_by_task[task_id]:
            state_id = str(row.get("state_id") or "").strip()
            choices = row.get("choices")
            if state_id not in node_ids or not isinstance(choices, list) or len(choices) != 4:
                invalid_rows.append(f"{task_id}:{state_id}:state_or_choice_count")
                continue
            if len({str(choice.get("original_id") or "") for choice in choices}) != 4:
                invalid_rows.append(f"{task_id}:{state_id}:duplicate_option_ids")
            for choice in choices:
                target = choice.get("to_state")
                if target and str(target) not in node_ids:
                    invalid_rows.append(f"{task_id}:{state_id}:unknown_to_state:{target}")
    if invalid_rows:
        raise ValueError(f"Invalid option-bank rows for {bank_path}: {invalid_rows[:10]}")
    return {
        "path": str(bank_path.resolve()),
        "task_count": len(expected),
        "row_count": sum(len(rows) for rows in rows_by_task.values()),
        "invalid_count": 0,
    }


def prepare_l_n_data(
    *,
    repo_root: Path,
    prepared_root: Path,
    ready_items: list[dict[str, Any]],
    force: bool,
) -> tuple[dict[str, Path], Path]:
    old_ready_root = repo_root / "For_user/analyze/L_N/input/formal_v1/input/image_reuse"
    runtime_root = prepared_root / "l_n" / "runtime"
    if force and runtime_root.exists():
        shutil.rmtree(runtime_root)
    ready_by_level: dict[str, list[dict[str, Any]]] = {"L3": [], "L5": [], "L6": []}
    option_sources = {
        "L3": repo_root / "For_user/analyze/L_N/input/formal_v1/input/api_pipeline/L3/final_option_bank_all.jsonl",
        "L5": repo_root / "For_user/analyze/L_N/input/formal_v2_tuned/input/api_pipeline/L5/final_option_bank_all.jsonl",
        "L6": repo_root / "For_user/analyze/L_N/input/formal_v2_tuned/input/api_pipeline/L6/final_option_bank_all.jsonl",
    }
    output_bank = prepared_root / "l_n" / "final_option_bank_all_levels.jsonl"
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    if force and output_bank.exists():
        output_bank.unlink()
    with output_bank.open("w", encoding="utf-8") as output:
        for level, source in option_sources.items():
            if not source.is_file():
                raise FileNotFoundError(f"Missing final option bank for {level}: {source}")
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    output.write(line + "\n")

    for item in ready_items:
        task_id = str(item.get("task_id") or "").strip()
        level = str(item.get("difficulty_level") or item.get("level") or "").strip()
        if level not in ready_by_level:
            continue
        graph_path = graph_source(repo_root, level, task_id)
        success_path = success_source(repo_root, level, task_id)
        if not graph_path.is_file() or not success_path.is_file():
            raise FileNotFoundError(f"Missing graph/success input for {task_id}: {graph_path}")
        graph = read_json(graph_path)
        success = read_json(success_path)
        images = source_image_index(item, old_ready_root)
        nodes = graph.get("nodes", [])
        manifest_states: list[dict[str, Any]] = []
        missing_images: list[str] = []
        for node in nodes:
            node_id = str(node.get("id") or "").strip()
            image = images.get(signature(node.get("state") or []))
            if not image:
                missing_images.append(node_id)
                continue
            manifest_states.append(
                {
                    "state_id": node_id,
                    "state": list(node.get("state") or []),
                    "copied_image_path": image,
                    "source_state_id": str(node.get("unique_state_id") or ""),
                }
            )
        stage1_path = repo_root / "For_user/analyze/L_N/input" / ("formal_v1" if level == "L3" else "formal_v2_tuned") / "input/api_pipeline" / level / task_id / "stage1_task_output.json"
        stage1 = read_json(stage1_path)
        description = str(stage1.get("description_highlevel_hard") or "").strip()
        if not description:
            raise ValueError(f"Stage 1 output has no description_highlevel_hard: {stage1_path}")
        graph["description_highlevel_hard"] = description
        graph["description_highlevel"] = description
        shortest_ids = shortest_node_ids(graph, success)
        manifest_node_ids = {str(state["state_id"]) for state in manifest_states}
        missing_shortest = [state_id for state_id in shortest_ids if state_id not in manifest_node_ids]
        if missing_shortest:
            raise ValueError(f"Shortest path images are missing for {task_id}: {missing_shortest}")
        task_dir = runtime_root / "tasks" / task_id
        input_dir = task_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        graph_destination = input_dir / graph_path.name
        write_json(graph_destination, graph)
        extra_names = (
            graph_path.name.replace("_atomic_expanded_graph.json", "_atomic_transition_manifest.json"),
            graph_path.name.replace("_atomic_expanded_graph.json", "_unique_states.json"),
        )
        for name in (success_path.name, *extra_names):
            source = success_path.parent / name if level == "L3" else graph_path.parent / name
            if source.is_file():
                link_file(source, input_dir / name)
        write_json(task_dir / "task_image_manifest.json", {"task_id": task_id, "states": manifest_states})
        enriched = dict(item)
        enriched["task_dir"] = str(Path("tasks") / task_id)
        enriched["copied_graph_files"] = [str(Path("tasks") / task_id / "input" / name) for name in [graph_destination.name, success_path.name]]
        enriched["state_count"] = len(nodes)
        enriched["reused_state_count"] = len(nodes)
        enriched["missing_state_count"] = len(missing_images)
        enriched["shortest_path_state_ids"] = shortest_ids
        enriched["shortest_path_state_count"] = len(enriched["shortest_path_state_ids"])
        enriched["shortest_path_missing_state_count"] = len(missing_shortest)
        enriched["downstream_ready"] = True
        enriched["fully_image_ready"] = not bool(missing_images)
        enriched_task = dict(enriched.get("task") or {})
        enriched_task["description_highlevel_hard"] = description
        enriched["task"] = enriched_task
        ready_by_level[level].append(enriched)

    ready_paths: dict[str, Path] = {}
    for level, items in ready_by_level.items():
        path = prepared_root / "l_n" / f"ready_{level.lower()}_tasks.json"
        write_json(path, items)
        ready_paths[level] = path
    write_json(
        prepared_root / "l_n" / "input_manifest.json",
        {
            "schema_version": "tongbench_analysis_eval_input_manifest_v1",
            "task_generation_performed": False,
            "option_generation_performed": False,
            "source_ready_json": str((old_ready_root / "ready_l3_l5_l6_tasks_with_images.json").resolve()),
            "ready_counts": {level: len(items) for level, items in ready_by_level.items()},
            "option_bank_sources": {level: str(path.resolve()) for level, path in option_sources.items()},
            "option_bank_sha256": {level: sha256_file(path) for level, path in option_sources.items()},
            "runtime_results_root": str(runtime_root.resolve()),
            "notes": "L5/L6 use the tuned v2 graph and option banks; L3 uses the completed v1 shortest-path bank.",
        },
    )
    return ready_paths, output_bank


def prepare_n_sample(
    *,
    repo_root: Path,
    prepared_root: Path,
    force: bool,
    backend_ks: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, dict[int, Path]], Path, Path]:
    source = repo_root / "For_user/analyze/N-sample/input/formal_child"
    destination = prepared_root / "n_sample"
    destination.mkdir(parents=True, exist_ok=True)
    ready_path = destination / "ready_l4_tasks.json"
    write_json(ready_path, read_json(source / "ready_l4_tasks_with_images.json"))
    option_bank = repo_root / "For_user/data/input/merged/option_bank_highlevel_v2_smoke.jsonl"
    if not option_bank.is_file():
        raise FileNotFoundError(option_bank)
    backend_ks = backend_ks or {"hermesagent": (1, 3, 4)}
    archives: dict[str, dict[int, Path]] = {}
    for backend, ks in backend_ks.items():
        source_index = load_l2_sources(repo_root, backend)
        archives[backend] = build_n_sample_archives(
            repo_root=repo_root,
            prepared_root=prepared_root,
            source_index=source_index,
            backend=backend,
            ks=ks,
            force=force,
        )
    copied_ks = list(range(7))
    for k in copied_ks:
        plan_source = source / f"learning_plan_k{k}.json"
        write_json(destination / f"learning_plan_k{k}.json", read_json(plan_source))
    write_json(
        destination / "input_manifest.json",
        {
            "schema_version": "tongbench_analysis_eval_input_manifest_v1",
            "task_generation_performed": False,
            "option_generation_performed": False,
            "ready_json": str(ready_path.resolve()),
            "option_bank": str(option_bank.resolve()),
            "option_bank_sha256": sha256_file(option_bank),
            "learning_archives": {
                backend: {str(k): str(path.resolve()) for k, path in paths.items()}
                for backend, paths in archives.items()
            },
            "k0_and_k2": "reference main-experiment results; not rerun here",
        },
    )
    return archives, ready_path, option_bank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-sample-only", action="store_true")
    parser.add_argument(
        "--n-sample-backend-ks",
        action="append",
        default=[],
        metavar="BACKEND:K,K",
        help="Build selected N-sample archives per backend; may be repeated.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    prepared_root = args.prepared_root.expanduser().resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    if args.force:
        for child in (prepared_root / "n_sample", prepared_root / "l_n"):
            if child.exists():
                shutil.rmtree(child)
    backend_ks = parse_backend_ks(args.n_sample_backend_ks) if args.n_sample_backend_ks else None
    n_archives, n_ready, n_bank = prepare_n_sample(
        repo_root=repo_root,
        prepared_root=prepared_root,
        force=args.force,
        backend_ks=backend_ks,
    )
    if args.n_sample_only:
        write_json(
            prepared_root / "prepare_summary.json",
            {
                "schema_version": "tongbench_analysis_prepare_summary_v1",
                "task_generation_performed": False,
                "option_generation_performed": False,
                "n_sample": {
                    "target_count": 50,
                    "archive_count": sum(len(paths) for paths in n_archives.values()),
                    "archive_count_by_backend": {
                        backend: len(paths) for backend, paths in n_archives.items()
                    },
                    "ready_json": str(n_ready),
                    "option_bank": str(n_bank),
                },
            },
        )
        n_ready_items = read_json(n_ready)
        n_graphs = {
            str(item["task_id"]): repo_root / "For_user/data/input/merged/tasks" / str(item["task_id"]) / "input" / f"{item['task_id']}_atomic_expanded_graph.json"
            for item in n_ready_items
        }
        write_json(
            prepared_root / "input_validation.json",
            {
                "n_sample_option_bank": validate_option_bank(
                    bank_path=n_bank,
                    ready_items=n_ready_items,
                    graph_for_task=n_graphs,
                    strict_tasks=False,
                )
            },
        )
        print(json.dumps({"ok": True, "prepared_root": str(prepared_root), "n_sample_only": True}, ensure_ascii=False, indent=2))
        return 0

    source_index = load_l2_sources(repo_root)
    main_entries = load_main_learning_entries(repo_root)
    old_ready = repo_root / "For_user/analyze/L_N/input/formal_v1/input/image_reuse/ready_l3_l5_l6_tasks_with_images.json"
    ready_items = read_json(old_ready)
    l_n_ready, l_n_bank = prepare_l_n_data(repo_root=repo_root, prepared_root=prepared_root, ready_items=ready_items, force=args.force)
    l_n_manifest = build_l_n_learning_manifest(prepared_root=prepared_root, main_entries=main_entries, ready_items=ready_items)
    l_n_manifest_payload = read_json(l_n_manifest)
    write_json(
        prepared_root / "l_n" / "learning_plan.json",
        {
            "schema_version": "tongbench_l_n_reuse_plan_v1",
            "learning_task_ids_are_reused": True,
            "targets": [
                {
                    "target_task_id": str(entry["target_task_id"]),
                    "learning_task_ids": list(entry.get("learning_task_ids") or []),
                    "source_parent_task_id": str(entry.get("source_parent_task_id") or ""),
                }
                for entry in l_n_manifest_payload.get("entries", [])
            ],
        },
    )
    write_json(
        prepared_root / "prepare_summary.json",
        {
            "schema_version": "tongbench_analysis_prepare_summary_v1",
            "task_generation_performed": False,
            "option_generation_performed": False,
            "n_sample": {"target_count": 50, "archive_count": sum(len(paths) for paths in n_archives.values()), "ready_json": str(n_ready), "option_bank": str(n_bank)},
            "l_n": {"ready_counts": {level: sum(1 for item in ready_items if str(item.get("difficulty_level") or item.get("level")) == level) for level in ("L3", "L5", "L6")}, "ready_json": {level: str(path) for level, path in l_n_ready.items()}, "option_bank": str(l_n_bank), "learning_manifest": str(l_n_manifest)},
            "source_l2_checkpoint_count": len(source_index),
            "main_learning_entry_count": len(main_entries),
        },
    )
    n_ready_items = read_json(n_ready)
    n_graphs = {
        str(item["task_id"]): repo_root / "For_user/data/input/merged/tasks" / str(item["task_id"]) / "input" / f"{item['task_id']}_atomic_expanded_graph.json"
        for item in n_ready_items
    }
    l_n_graphs = {
        str(item["task_id"]): prepared_root / "l_n/runtime/tasks" / str(item["task_id"]) / "input" / f"{item['task_id']}_atomic_expanded_graph.json"
        for item in ready_items
    }
    validation = {
        "n_sample_option_bank": validate_option_bank(bank_path=n_bank, ready_items=n_ready_items, graph_for_task=n_graphs, strict_tasks=False),
        "l_n_option_bank": validate_option_bank(bank_path=l_n_bank, ready_items=ready_items, graph_for_task=l_n_graphs),
    }
    write_json(prepared_root / "input_validation.json", validation)
    print(json.dumps({"ok": True, "prepared_root": str(prepared_root), "task_generation_performed": False, "option_generation_performed": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
