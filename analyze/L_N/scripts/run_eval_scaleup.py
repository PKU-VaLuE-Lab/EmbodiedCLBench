#!/usr/bin/env python3
"""Run L_N scale-up eval modes with bounded parallelism and API recovery."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


LEVELS = ("L3", "L5", "L6")
API_FAILURE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "quota",
    "insufficient_quota",
    "allocated quota",
    "timed out",
    "timeout",
    "connection reset",
    "connection aborted",
    "remote end closed",
    "http error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
)


def model_key(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


def redact(command: list[str]) -> str:
    values = list(command)
    for index, value in enumerate(values[:-1]):
        if value == "--api-key":
            values[index + 1] = "***"
    return " ".join(shlex.quote(value) for value in values)


def is_api_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in API_FAILURE_MARKERS)


def read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def request_count(path: Path) -> int:
    payload = read_json(path)
    try:
        return int(payload.get("request_count") or 0)
    except (TypeError, ValueError):
        return 0


def task_run_dir(*, mode: str, output_root: Path, task_id: str) -> Path:
    if mode == "zero_shot":
        return output_root / "task_runs" / task_id
    return output_root / task_id


def audit_mode_output(
    *,
    mode: str,
    output_root: Path,
    task_list: list[str],
) -> list[dict[str, str]]:
    """Reject infrastructure failures without treating genuine model failures as errors."""
    failures: list[dict[str, str]] = []
    for task_id in task_list:
        run_dir = task_run_dir(mode=mode, output_root=output_root, task_id=task_id)
        reasons: list[str] = []
        if mode == "zero_shot":
            score_path = run_dir / "score.json"
            usage_path = run_dir / "usage.json"
            result = read_json(run_dir / "single_task_run_result.json")
            if not score_path.is_file():
                reasons.append("missing score.json")
            if request_count(usage_path) <= 0:
                reasons.append("no successful API requests")
            if result.get("status") != "ok" or result.get("returncode") != 0:
                reasons.append("single-task runner did not finish cleanly")
        elif mode == "in_context":
            summary = read_json(run_dir / "sequence_summary.json")
            if summary.get("error"):
                reasons.append(f"sequence error: {summary['error']}")
            if request_count(run_dir / "usage.json") <= 0:
                reasons.append("no successful target-task API requests")
            scores = list((run_dir / "task_runs").glob(f"*{task_id}*/score.json"))
            if not scores:
                reasons.append("missing target score.json")
        else:
            summary = read_json(run_dir / "skill_mode_summary.json")
            if summary.get("error"):
                reasons.append(f"skill error: {summary['error']}")
            learning_dir = run_dir / "skill_learning"
            application_dir = run_dir / "skill_application"
            if request_count(learning_dir / "usage.json") <= 0:
                reasons.append("no successful skill-summary API request")
            if request_count(application_dir / "usage.json") <= 0:
                reasons.append("no successful target-task API requests")
            scores = list((application_dir / "task_runs").glob(f"*{task_id}*/score.json"))
            if not scores:
                reasons.append("missing target score.json")
        if reasons:
            diagnostic_text = ""
            for candidate in run_dir.rglob("agent.log") if run_dir.exists() else []:
                try:
                    diagnostic_text += candidate.read_text(encoding="utf-8", errors="replace")[-20000:]
                except OSError:
                    pass
            failures.append(
                {
                    "task_id": task_id,
                    "reasons": "; ".join(reasons),
                    "diagnostic_tail": diagnostic_text[-20000:],
                }
            )
    return failures


def clear_failed_outputs(*, mode: str, output_root: Path, failures: list[dict[str, str]]) -> None:
    for failure in failures:
        task_id = failure["task_id"]
        run_dir = task_run_dir(mode=mode, output_root=output_root, task_id=task_id)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        if mode == "in_context":
            continuation = output_root / "_reused_in_context_checkpoints" / task_id
            if continuation.exists():
                shutil.rmtree(continuation)


def run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {redact(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=env["TONGBENCH_REPO_ROOT"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        for line in process.stdout or []:
            output.append(line)
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
    return process.wait(), "".join(output)


def task_ids(ready_json: Path, levels: set[str]) -> list[str]:
    payload = json.loads(ready_json.read_text(encoding="utf-8"))
    result = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id") or "").strip()
        task = item.get("task") if isinstance(item.get("task"), dict) else {}
        level = str(
            item.get("difficulty_level")
            or item.get("level")
            or task.get("difficulty_level")
            or task.get("level")
            or ""
        ).strip().upper()
        if task_id and level in levels:
            result.append(task_id)
    return result


def common_args(args: argparse.Namespace, *, choice_bank: Path, results_root: Path) -> list[str]:
    return [
        "--results-root", str(results_root),
        "--backend", "hermesagent",
        "--model", args.model,
        "--api-key", args.api_key,
        "--base-url", args.base_url,
        "--description-field", "description_highlevel_hard",
        "--framework-interaction-mode", "semantic_tool",
        "--public-interface-mode", "natural_language",
        "--protocol-surface", "safe_choice",
        "--choice-bank-jsonl", str(choice_bank),
        "--choice-count", "4",
        "--choice-seed", "42",
        "--write-human-trace-md",
        "--generic-invalid-action-feedback",
        "--max-steps-extra", str(args.extra_steps),
        "--timeout-sec", str(args.timeout_sec),
        "--max-iterations", str(args.max_iterations),
        "--atomic-template-path", str(args.atomic_template),
        "--subtask-template-path", str(args.subtask_template),
        "--parallel-jobs", str(args.parallel_jobs),
        "--skip-existing",
    ]


def mode_command(
    args: argparse.Namespace,
    *,
    mode: str,
    task_list: list[str],
    ready_json: Path,
    results_root: Path,
    output_root: Path,
    choice_bank: Path,
    learning_plan: Path,
    reuse_manifest: Path,
) -> list[str]:
    if mode == "zero_shot":
        command = [
            str(args.python_bin), "-m", "tongbench_eval.cli.framework.ready_single_tasks",
            "--ready-json", str(ready_json), "--output-dir", str(output_root),
            *common_args(args, choice_bank=choice_bank, results_root=results_root),
            "--composite-only",
        ]
    else:
        command = [
            str(args.python_bin), "-m", "tongbench_eval.cli.framework.dialogue",
            "--ready-json", str(ready_json), "--output-dir", str(output_root),
            "--hide-decomposition", "--self-evolution-mode", mode,
            "--learning-plan-json", str(learning_plan),
            "--reuse-learning-manifest", str(reuse_manifest),
            "--learning-checkpoint-root", str(output_root.parent.parent / "_reused_learning_checkpoints"),
            *common_args(args, choice_bank=choice_bank, results_root=results_root),
        ]
    for task_id in task_list:
        command.extend(["--task-id", task_id])
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the L_N scale-up eval.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--levels", default="L3,L5,L6")
    parser.add_argument("--modes", default="zero_shot,in_context,skill")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--extra-steps", type=int, default=6)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-iterations", type=int, default=320)
    parser.add_argument("--quota-wait-sec", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--choice-bank-jsonl",
        type=Path,
        help="Override the option bank. Defaults to the bundled L3/L5/L6 bank.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    bundle = args.bundle_root.resolve()
    args.python_bin = args.python_bin.resolve()
    if not args.api_key:
        raise RuntimeError("Set --api-key or OPENAI_API_KEY.")
    selected_levels = {value.strip().upper() for value in args.levels.replace(",", " ").split() if value.strip()}
    selected_modes = [value.strip() for value in args.modes.replace(",", " ").split() if value.strip()]
    if not selected_levels.issubset(set(LEVELS)):
        raise ValueError(f"Unsupported levels: {sorted(selected_levels - set(LEVELS))}")
    if any(mode not in {"zero_shot", "in_context", "skill"} for mode in selected_modes):
        raise ValueError(f"Unsupported modes: {selected_modes}")
    if "skill" in selected_modes and "in_context" not in selected_modes:
        raise ValueError("skill mode requires in_context in the same run")

    args.atomic_template = repo / "private/taskgen/outputs/atomic_templates_updated_env_v3.json"
    args.subtask_template = repo / "private/taskgen/outputs/subtask_templates_compressed_updated_env_v3.json"
    for path in (args.atomic_template, args.subtask_template):
        if not path.is_file():
            raise FileNotFoundError(path)

    prepare = [
        str(args.python_bin), str(repo / "analyze/L_N/scripts/prepare_eval_scaleup.py"),
        "--bundle-root", str(bundle),
    ]
    if args.force_prepare:
        prepare.append("--force")
    prep_env = os.environ.copy()
    prep_env["TONGBENCH_REPO_ROOT"] = str(repo)
    code, _ = run_command(prepare, env=prep_env, log_path=bundle / "output/eval_scaleup/prepare_eval_ready.log")
    if code:
        return code

    eval_ready_root = bundle / "input/eval_ready_scaleup"
    ready_json = eval_ready_root / "ready_l3_l5_l6_api_descriptions.json"
    results_root = eval_ready_root
    choice_bank = (
        args.choice_bank_jsonl.expanduser().resolve()
        if args.choice_bank_jsonl
        else bundle / "input/api_pipeline/final_option_bank_all_levels.jsonl"
    )
    learning_plan = bundle / "input/learning_plan_reuse.json"
    reuse_manifest = bundle / "input/learning_reuse_manifest.json"
    for path in (ready_json, choice_bank, learning_plan, reuse_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = task_ids(ready_json, selected_levels)
    if not ids:
        raise ValueError("No tasks selected")

    env = os.environ.copy()
    env["TONGBENCH_REPO_ROOT"] = str(repo)
    env["PYTHONPATH"] = str(repo / "public/eval/src")
    env["PYTHONNOUSERSITE"] = "1"
    output_base = bundle / "output/eval_scaleup" / model_key(args.model) / "hermesagent"
    manifest = {
        "schema_version": "tongbench_l_n_eval_scaleup_v1",
        "model": args.model,
        "backend": "hermesagent",
        "levels": sorted(selected_levels),
        "modes": selected_modes,
        "task_count": len(ids),
        "parallel_jobs": max(1, int(args.parallel_jobs)),
        "extra_steps": args.extra_steps,
        "mcp_compress_images": False,
        "ready_json": str(ready_json),
        "choice_bank": str(choice_bank),
        "learning_plan": str(learning_plan),
        "reuse_manifest": str(reuse_manifest),
        "attempts": [],
    }
    manifest_path = output_base / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for mode in selected_modes:
        mode_output = output_base / mode
        log_path = bundle / "output/eval_scaleup/logs" / f"{mode}.log"
        command = mode_command(
            args, mode=mode, task_list=ids, ready_json=ready_json,
            results_root=results_root, output_root=mode_output,
            choice_bank=choice_bank, learning_plan=learning_plan,
            reuse_manifest=reuse_manifest,
        )
        succeeded = False
        for attempt in range(1, max(1, args.max_attempts) + 1):
            code, output = run_command(command, env=env, log_path=log_path)
            failures = audit_mode_output(mode=mode, output_root=mode_output, task_list=ids)
            diagnostic_text = "\n".join(failure["diagnostic_tail"] for failure in failures)
            raw_api_failure = is_api_failure(output) or is_api_failure(diagnostic_text)
            api_failure = bool((code != 0 or failures) and raw_api_failure)
            manifest["attempts"].append(
                {
                    "mode": mode,
                    "attempt": attempt,
                    "returncode": code,
                    "api_failure_detected": api_failure,
                    "invalid_output_count": len(failures),
                    "invalid_outputs": [
                        {key: value for key, value in failure.items() if key != "diagnostic_tail"}
                        for failure in failures
                    ],
                }
            )
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if code == 0 and not failures:
                succeeded = True
                break
            if not api_failure or attempt >= max(1, args.max_attempts):
                print(f"{mode} failed with a non-recoverable or exhausted error; see {log_path}", flush=True)
                return code or 1
            clear_failed_outputs(mode=mode, output_root=mode_output, failures=failures)
            print(f"{mode} encountered an API/connection failure; waiting {args.quota_wait_sec}s before retry {attempt + 1}.", flush=True)
            time.sleep(max(0, int(args.quota_wait_sec)))
        if not succeeded:
            return 1
    manifest["ok"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "task_count": len(ids), "modes": selected_modes, "output_root": str(output_base)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
