#!/usr/bin/env python3
"""Run the prepared N-sample and L_N analysis evaluations.

The script launches only the existing native evaluation CLIs. It does not
generate tasks, graphs, images, or options. A smoke run selects one target per
actual evaluation job; a formal run submits all selected jobs in one scheduler
batch and lets each existing CLI parallelize task dialogues.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def redacted(command: list[str]) -> str:
    result = list(command)
    for index, part in enumerate(result[:-1]):
        if part == "--api-key":
            result[index + 1] = "***"
    return " ".join(shlex.quote(part) for part in result)


def task_ids(ready_json: Path) -> list[str]:
    payload = read_json(ready_json)
    if not isinstance(payload, list):
        raise ValueError(f"Ready JSON must be a list: {ready_json}")
    result = [str(row.get("task_id") or "").strip() for row in payload if isinstance(row, dict)]
    result = [value for value in result if value]
    if not result:
        raise ValueError(f"No task ids in {ready_json}")
    return result


def split_values(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def backend_extra_args(backend: str) -> list[str]:
    if backend == "hermesagent":
        return []
    if backend == "openclaw":
        return ["--codex-direct-image-input"]
    raise ValueError(f"Unsupported analysis backend: {backend}")


def parse_backend_ks(values: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for value in values:
        backend, separator, raw_ks = value.partition(":")
        backend = backend.strip()
        if not separator or not backend:
            raise ValueError(f"Expected BACKEND:K,K, got {value!r}")
        result[backend] = sorted({int(item.strip()) for item in raw_ks.split(",") if item.strip()})
    return result


def common_args(
    args: argparse.Namespace,
    option_bank: Path,
    results_root: Path,
    backend: str,
) -> list[str]:
    key = env_first("QWEN_API_KEY", "DASHSCOPE_API_KEY")
    if not key and not args.dry_run:
        raise RuntimeError("Set QWEN_API_KEY or DASHSCOPE_API_KEY before running evaluation")
    return [
        "--results-root", str(results_root),
        "--backend", backend,
        "--model", args.model,
        "--runtime", "native",
        "--api-key", key or "DRY_RUN_API_KEY",
        "--base-url", env_first("QWEN_BASE_URL", "DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--description-field", "description_highlevel_hard",
        "--framework-interaction-mode", "semantic_tool",
        "--public-interface-mode", "natural_language",
        "--protocol-surface", "safe_choice",
        "--choice-bank-jsonl", str(option_bank),
        "--choice-count", "4",
        "--choice-seed", "42",
        "--write-human-trace-md",
        "--generic-invalid-action-feedback",
        "--max-steps-extra", str(args.extra_steps),
        "--timeout-sec", str(args.timeout_sec),
        "--max-iterations", str(args.max_iterations),
        "--context-mode", args.context_mode,
        "--atomic-template-path", str(args.atomic_template),
        "--subtask-template-path", str(args.subtask_template),
        "--thinking", args.thinking,
        *backend_extra_args(backend),
    ]


def command_for_job(args: argparse.Namespace, job: dict[str, Any]) -> list[str]:
    ready = Path(job["ready_json"])
    output = Path(job["output_dir"])
    option_bank = Path(job["option_bank"])
    common = common_args(args, option_bank, Path(job["results_root"]), str(job["backend"]))
    command = [str(args.python_bin)]
    if job["setting"] == "zero_shot":
        command += [
            "-m", "tongbench_eval.cli.framework.ready_single_tasks",
            "--ready-json", str(ready),
            "--output-dir", str(output),
            "--composite-only",
            "--parallel-jobs", str(args.task_parallel_jobs),
            "--skip-existing",
        ]
    else:
        command += [
            "-m", "tongbench_eval.cli.framework.dialogue",
            "--ready-json", str(ready),
            "--output-dir", str(output),
            "--hide-decomposition",
            "--self-evolution-mode", str(job["mode"]),
            "--learning-plan-json", str(job["learning_plan"]),
            "--learning-checkpoint-root", str(output.parent / "_learning_checkpoints"),
            "--reuse-learning-manifest", str(job["reuse_manifest"]),
            "--parallel-jobs", str(args.task_parallel_jobs),
            "--skip-existing",
        ]
    command += common
    if args.mode == "smoke":
        command += ["--limit", "1"]
    else:
        for task_id in job["selected_task_ids"]:
            command += ["--task-id", task_id]
    return command


def make_jobs(args: argparse.Namespace, prepared: Path, output_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if args.experiment in {"all", "n_sample"}:
        n_root = prepared / "n_sample"
        n_ready = n_root / "ready_l4_tasks.json"
        n_ids = task_ids(n_ready)
        for backend, backend_ks in args.n_sample_backend_ks.items():
            for k in backend_ks:
                for mode in ("in_context", "skill"):
                    jobs.append(
                        {
                            "experiment": "n_sample",
                            "label": f"n_sample_{backend}_k{k}_{mode}",
                            "backend": backend,
                            "level": "L4",
                            "setting": "learning",
                            "mode": mode,
                            "ready_json": str(n_ready),
                            "option_bank": str(args.n_sample_option_bank),
                            "learning_plan": str(n_root / f"learning_plan_k{k}.json"),
                            "reuse_manifest": str(prepared / "learning_archives" / "n_sample" / backend / f"k{k}" / "learning_reuse_manifest.json"),
                            "selected_task_ids": n_ids,
                            "results_root": str(args.repo_root / "For_user/data/input/merged"),
                            "output_dir": str(output_root / "n_sample" / backend / f"k{k}" / mode),
                        }
                    )
    if args.experiment == "n_sample":
        return jobs
    l_root = prepared / "l_n"
    l_manifest = prepared / "learning_archives" / "l_n" / "learning_reuse_manifest.json"
    l_plan = l_root / "learning_plan.json"
    if not l_plan.exists():
        source_items = []
        for level in ("L3", "L5", "L6"):
            source_items.extend(read_json(l_root / f"ready_{level.lower()}_tasks.json"))
        write_json(
            l_plan,
            {
                "schema_version": "tongbench_l_n_reuse_plan_v1",
                "learning_task_ids_are_reused": True,
                "targets": [
                    {
                        "target_task_id": str(item["task_id"]),
                        "learning_task_ids": [],
                        "source_parent_task_id": str((item.get("task") or {}).get("source_parent_task_id") or ""),
                    }
                    for item in source_items
                ],
            },
        )
    for level in ("L3", "L5", "L6"):
        ready = l_root / f"ready_{level.lower()}_tasks.json"
        ids = task_ids(ready)
        for setting in ("zero_shot", "in_context", "skill"):
            jobs.append(
                {
                    "experiment": "l_n",
                    "label": f"l_n_{level}_{setting}",
                    "backend": "hermesagent",
                    "level": level,
                    "setting": "zero_shot" if setting == "zero_shot" else "learning",
                    "mode": None if setting == "zero_shot" else setting,
                    "ready_json": str(ready),
                    "option_bank": str(args.l_n_option_bank),
                    "learning_plan": str(l_plan),
                    "reuse_manifest": str(l_manifest),
                    "selected_task_ids": ids,
                    "results_root": str(prepared / "l_n/runtime"),
                    "output_dir": str(output_root / "l_n" / level / setting),
                }
            )
    return jobs


def run_job(args: argparse.Namespace, job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    command = command_for_job(args, job)
    log_path = output / "scheduler.log"
    metadata_path = output / "job.json"
    record = {
        **job,
        "command": redacted(command),
        "started_at": time.time(),
        "task_generation_performed": False,
        "option_generation_performed": False,
    }
    if args.dry_run:
        command.append("--dry-run")
        record["command"] = redacted(command)
    write_json(metadata_path, record)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + redacted(command) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT)
        code = process.returncode
    record["finished_at"] = time.time()
    record["returncode"] = code
    write_json(metadata_path, record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--experiment", choices=("all", "n_sample", "l_n"), default="all")
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--thinking", default="medium")
    parser.add_argument("--context-mode", choices=("full", "task_compact"), default="task_compact")
    parser.add_argument("--extra-steps", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-iterations", type=int, default=320)
    parser.add_argument("--task-parallel-jobs", type=int, default=4)
    parser.add_argument("--job-concurrency", type=int, default=6)
    parser.add_argument("--backends", default="hermesagent")
    parser.add_argument("--n-sample-ks", default="1,3,4")
    parser.add_argument(
        "--n-sample-backend-ks",
        action="append",
        default=[],
        metavar="BACKEND:K,K",
        help="Select different k values per backend; overrides --backends and --n-sample-ks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.atomic_template = args.repo_root / "private/taskgen/outputs/atomic_templates_updated_env_v3.json"
    args.subtask_template = args.repo_root / "private/taskgen/outputs/subtask_templates_compressed_updated_env_v3.json"
    args.backends = split_values(args.backends)
    args.n_sample_ks = sorted({int(value) for value in split_values(args.n_sample_ks)})
    args.n_sample_backend_ks = (
        parse_backend_ks(args.n_sample_backend_ks)
        if args.n_sample_backend_ks
        else {backend: list(args.n_sample_ks) for backend in args.backends}
    )
    args.backends = list(args.n_sample_backend_ks)
    if any(backend not in {"hermesagent", "openclaw"} for backend in args.backends):
        raise ValueError("--backends supports hermesagent and openclaw")
    selected_ks = {k for ks in args.n_sample_backend_ks.values() for k in ks}
    if not selected_ks or any(k < 1 or k > 6 for k in selected_ks):
        raise ValueError("--n-sample-ks values must be in 1..6")
    args.n_sample_option_bank = args.repo_root / "For_user/data/input/merged/option_bank_highlevel_v2_smoke.jsonl"
    args.l_n_option_bank = args.prepared_root / "l_n/final_option_bank_all_levels.jsonl"
    if not args.prepared_root.is_dir():
        raise FileNotFoundError(f"Missing prepared inputs: {args.prepared_root}; run prepare_eval_inputs.py first")
    jobs = make_jobs(args, args.prepared_root, args.output_root)
    if args.experiment != "all":
        jobs = [job for job in jobs if job["experiment"] == args.experiment]
    if args.mode == "smoke":
        for job in jobs:
            job["selected_task_ids"] = list(job["selected_task_ids"][:1])
    if not jobs:
        raise ValueError("No analysis jobs selected")
    write_json(
        args.output_root / "run_manifest.json",
        {
            "schema_version": "tongbench_analysis_eval_run_manifest_v1",
            "mode": args.mode,
            "experiment": args.experiment,
            "model": args.model,
            "backends": args.backends,
            "n_sample_ks": args.n_sample_ks,
            "n_sample_backend_ks": args.n_sample_backend_ks,
            "runtime": "native",
            "context_mode": args.context_mode,
            "extra_steps": args.extra_steps,
            "task_generation_performed": False,
            "option_generation_performed": False,
            "job_count": len(jobs),
            "jobs": jobs,
        },
    )
    print(f"Submitting {len(jobs)} jobs with job concurrency {args.job_concurrency}; task concurrency per job {args.task_parallel_jobs}", flush=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.job_concurrency)) as executor:
        futures = [executor.submit(run_job, args, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"{result['label']}: returncode={result['returncode']}", flush=True)
    failures = [result for result in results if result.get("returncode") not in (0, None)]
    write_json(args.output_root / "run_summary.json", {"job_count": len(results), "failure_count": len(failures), "results": results})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
