#!/usr/bin/env python3
"""Run the child-only N-sample experiment with the existing eval CLIs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_ROOT.parents[3]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "analyze/N-sample/input/subset_10pct_child"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analyze/N-sample/output"
DEFAULT_SOURCE_RESULTS = PROJECT_ROOT / "For_user/data/input/merged"
DEFAULT_ATOMIC_TEMPLATE = PROJECT_ROOT / "private/taskgen/outputs/atomic_templates_updated_env_v3.json"
DEFAULT_SUBTASK_TEMPLATE = PROJECT_ROOT / "private/taskgen/outputs/subtask_templates_compressed_updated_env_v3.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def split_values(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def redacted_command(command: list[str]) -> str:
    redacted = list(command)
    for index, value in enumerate(redacted[:-1]):
        if value == "--api-key":
            redacted[index + 1] = "***"
    return " ".join(shlex.quote(value) for value in redacted)


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> int:
    print(f"\n$ {redacted_command(command)}", flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, env=env).returncode


def model_key_from_name(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HermesAgent + Qwen N-sample analysis.")
    parser.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--source-results-root", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", "qwen3.7-plus"))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--ks", default="1,2,3,4")
    parser.add_argument("--modes", default="in_context,skill")
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-iterations", type=int, default=320)
    parser.add_argument("--extra-steps", type=int, default=6)
    parser.add_argument("--choice-seed", type=int, default=42)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    input_root = args.input_root.resolve()
    source_results_root = args.source_results_root.resolve()
    output_root = args.output_root.resolve() / args.phase
    ready_l4 = input_root / "ready_l4_tasks_with_images.json"
    ready_l2 = input_root / "ready_l2_tasks_with_images.json"
    choice_bank = source_results_root / "option_bank_highlevel_v2_smoke.jsonl"
    atomic_template = project_root / "private/taskgen/outputs/atomic_templates_updated_env_v3.json"
    subtask_template = project_root / "private/taskgen/outputs/subtask_templates_compressed_updated_env_v3.json"
    for path in (ready_l4, ready_l2, choice_bank, atomic_template, subtask_template):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    ready_items = read_json(ready_l4)
    target_ids = [str(item.get("task_id") or "") for item in ready_items if isinstance(item, dict)]
    target_ids = [value for value in target_ids if value]
    if not target_ids:
        raise ValueError(f"No target tasks found in {ready_l4}")
    selected_ids = args.task_id or (target_ids[:1] if args.phase == "smoke" else target_ids)
    selected_set = set(selected_ids)
    unknown_ids = selected_set - set(target_ids)
    if unknown_ids:
        raise ValueError(f"Unknown target task ids: {sorted(unknown_ids)}")

    api_key = args.api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("Set --api-key, QWEN_API_KEY, or DASHSCOPE_API_KEY before running.")
    if not api_key:
        api_key = "DRY_RUN_API_KEY"
    base_url = args.base_url or os.environ.get("QWEN_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    requested_ks = sorted({int(value) for value in split_values(args.ks)})
    if any(value < 0 or value > 4 for value in requested_ks):
        raise ValueError("--ks values must be in 0..4")
    # k=0 is supplied by the main experiment zero-shot run and is not duplicated here.
    ks = [value for value in requested_ks if value != 0]
    modes = split_values(args.modes)
    if any(mode not in {"in_context", "skill"} for mode in modes):
        raise ValueError("--modes accepts only in_context and skill")
    if not modes:
        raise ValueError("At least one mode is required")
    if "skill" in modes and "in_context" not in modes:
        raise ValueError("skill mode requires in_context in the same run so its learning checkpoint can be reused")
    modes = ["in_context", "skill"] if set(modes) == {"in_context", "skill"} else modes

    env = os.environ.copy()
    eval_src = project_root / "public/eval/src"
    # Keep child processes on this checkout's eval implementation.  The shared
    # environment also contains an editable install of an older checkout, and
    # appending its path can mix incompatible dialogue_runtime modules.
    env["PYTHONPATH"] = str(eval_src)
    env["PYTHONNOUSERSITE"] = "1"
    output_root.mkdir(parents=True, exist_ok=True)

    common = [
        "--results-root",
        str(source_results_root),
        "--backend",
        "hermesagent",
        "--model",
        args.model,
        "--api-key",
        api_key,
        "--base-url",
        base_url,
        "--description-field",
        "description_highlevel_hard",
        "--framework-interaction-mode",
        "semantic_tool",
        "--public-interface-mode",
        "natural_language",
        "--protocol-surface",
        "safe_choice",
        "--choice-bank-jsonl",
        str(choice_bank),
        "--choice-count",
        "4",
        "--choice-seed",
        str(args.choice_seed),
        "--write-human-trace-md",
        "--generic-invalid-action-feedback",
        "--max-steps-extra",
        str(args.extra_steps),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-iterations",
        str(args.max_iterations),
        "--atomic-template-path",
        str(atomic_template),
        "--subtask-template-path",
        str(subtask_template),
        "--parallel-jobs",
        str(max(1, args.parallel_jobs)),
        "--skip-existing",
    ]

    manifest = {
        "schema_version": "tongbench_n_sample_run_v1",
        "phase": args.phase,
        "model": args.model,
        "harness": "hermesagent",
        "target_ids": selected_ids,
        "requested_ks": requested_ks,
        "ks": ks,
        "zero_shot_reference": "main experiment zero-shot output; no N-sample k=0 process is launched",
        "modes": modes,
        "max_steps_extra": args.extra_steps,
        "mcp_compress_images": False,
        "dry_run": args.dry_run,
        "source_results_root": str(source_results_root),
        "ready_l4": str(ready_l4),
        "ready_l2": str(ready_l2),
        "choice_bank": str(choice_bank),
        "commands": [],
        "failures": [],
    }

    failures: list[dict[str, object]] = []
    for k in ks:
        plan_path = input_root / f"learning_plan_k{k}.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"Missing learning plan: {plan_path}")
        checkpoint_root = output_root / "_learning_checkpoints" / model_key_from_name(args.model) / "hermesagent" / f"k{k}"
        icl_completed = True
        for mode in modes:
            condition_dir = output_root / model_key_from_name(args.model) / "hermesagent" / f"k{k}" / mode
            command = [
                str(args.python_bin),
                "-m",
                "tongbench_eval.cli.framework.dialogue",
                "--ready-json",
                str(ready_l4),
                "--output-dir",
                str(condition_dir),
                "--hide-decomposition",
                "--self-evolution-mode",
                mode,
                "--learning-plan-json",
                str(plan_path),
                "--learning-checkpoint-root",
                str(checkpoint_root),
                *common,
            ]
            if args.phase == "smoke":
                command.extend(["--limit", "1"])
            elif args.task_id:
                for selected_id in selected_ids:
                    command.extend(["--task-id", selected_id])
            manifest["commands"].append(redacted_command(command))
            if mode == "skill" and not icl_completed:
                failures.append({"k": k, "mode": mode, "returncode": "skipped_after_icl_failure"})
                continue
            code = run_command(command, env=env, dry_run=args.dry_run)
            if code:
                failures.append({"k": k, "mode": mode, "returncode": code})
                if mode == "in_context":
                    icl_completed = False

    manifest["failures"] = failures
    manifest_name = "run_manifest_dry_run.json" if args.dry_run else "run_manifest.json"
    (output_root / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        print(json.dumps({"ok": False, "failures": failures}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"ok": True, "phase": args.phase, "target_count": len(selected_ids), "requested_ks": requested_ks, "executed_ks": ks, "zero_shot": "reused from main experiment"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
