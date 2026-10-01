#!/usr/bin/env python3
"""Run the shared three-stage API pipeline for a small L_N task subset.

The API key is read from the environment by the shared pipeline.  This
wrapper deliberately does not persist credentials in the analysis bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_TASK_IDS = (
    "task_L3_LN_smoke_0001",
    "task_L5_LN_smoke_0001",
    "task_L6_LN_smoke_0001",
)


def run(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum completion tokens for each API call; Stage 3 needs more than the short smoke default.",
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    external = args.external_root.resolve()
    bundle = args.bundle_root.resolve()
    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    output_root = bundle / "input/api_pipeline"
    taskgen_output = bundle / "input/taskgen_output"
    inventory = bundle / "input/highlevel/l1_inventory.json"
    eval_input = bundle / "input/image_reuse"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "private/taskgen/src") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if not (env.get("DASHSCOPE_API_KEY") or env.get("OPENAI_API_KEY")):
        raise RuntimeError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running the API smoke.")
    python = env.get("PYTHON", "python3")
    module = "tongbench_taskgen.cli.highlevel_api_pipeline_v2"
    task_csv = ",".join(task_ids)
    log_root = bundle / "output/api_smoke_logs"
    export_command = [
        python, "-m", module, "export-inputs",
        "--project-root", str(external),
        "--output-root", str(output_root),
        "--taskgen-output-dir", str(taskgen_output),
        "--l1-inventory", str(inventory),
        "--eval-input-dir", str(eval_input),
        "--task-ids", task_csv,
        "--target-valid-detours-per-state", "2",
        "--min-valid-detours-per-state", "0",
    ]
    run(export_command, cwd=external, env=env, log_path=log_root / "export_inputs.log")
    run_all_command = [
        python, "-m", module, "run-all",
        "--project-root", str(external),
        "--output-root", str(output_root),
        "--task-ids", task_csv,
        "--model", "qwen3.7-plus",
        "--base-url", "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "--temperature", "0.4",
        "--max-tokens", str(args.max_tokens),
        "--timeout", str(args.timeout),
        "--max-concurrency", str(args.max_concurrency),
    ]
    run(run_all_command, cwd=external, env=env, log_path=log_root / "run_all.log")
    manifest = {
        "analysis": "L_N",
        "task_ids": task_ids,
        "model": "qwen3.7-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_concurrency": args.max_concurrency,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "credentials_persisted": False,
        "output_root": str(output_root),
    }
    (bundle / "output/api_smoke_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
