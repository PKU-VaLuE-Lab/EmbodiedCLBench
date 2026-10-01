#!/usr/bin/env python3
"""Run a zero-shot Hermes/Qwen smoke after the L_N option bank is complete."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TASK_IDS = (
    "task_L3_LN_smoke_0001",
    "task_L5_LN_smoke_0001",
    "task_L6_LN_smoke_0001",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASK_IDS))
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--max-steps-extra", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-iterations", type=int, default=320)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    bundle = args.bundle_root.resolve()
    ready_json = bundle / "input/image_reuse/ready_highlevel_v2_tasks_with_images.json"
    results_root = bundle / "input/image_reuse"
    choice_bank = bundle / "input/api_pipeline/final_option_bank_all.jsonl"
    if not ready_json.is_file():
        raise FileNotFoundError(f"Missing ready image manifest: {ready_json}")
    if not choice_bank.is_file():
        raise FileNotFoundError(f"Missing final option bank; finish API Stage 3 first: {choice_bank}")

    task_ids = [item.strip() for item in args.task_ids.split(",") if item.strip()]
    output_dir = bundle / "output/eval_smoke" / args.model.replace("/", "_").replace(".", "_") / "hermesagent" / "zero_shot"
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running eval smoke.")

    docker_check = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if docker_check.returncode:
        detail = (docker_check.stderr or "Docker daemon is unavailable").strip()
        raise RuntimeError(
            "Docker preflight failed. Run this smoke as a user with Docker daemon access "
            f"(or through sudo), then retry. Details: {detail}"
        )

    eval_src = repo / "public/eval/src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(eval_src)
    env["PYTHONNOUSERSITE"] = "1"
    command = [
        sys.executable,
        "-m",
        "tongbench_eval.cli.framework.ready_single_tasks",
        "--ready-json", str(ready_json),
        "--results-root", str(results_root),
        "--output-dir", str(output_dir),
        "--backend", "hermesagent",
        "--model", args.model,
        "--description-field", "description_highlevel_hard",
        "--api-key", api_key,
        "--base-url", args.base_url,
        "--framework-interaction-mode", "semantic_tool",
        "--public-interface-mode", "natural_language",
        "--protocol-surface", "safe_choice",
        "--choice-count", "4",
        "--choice-bank-jsonl", str(choice_bank),
        "--write-human-trace-md",
        "--generic-invalid-action-feedback",
        "--max-steps-extra", str(args.max_steps_extra),
        "--timeout-sec", str(args.timeout_sec),
        "--max-iterations", str(args.max_iterations),
        "--parallel-jobs", "1",
        "--composite-only",
    ]
    for task_id in task_ids:
        command.extend(["--task-id", task_id])
    print("Running L_N zero-shot smoke for:", ", ".join(task_ids), flush=True)
    completed = subprocess.run(command, cwd=repo, env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
