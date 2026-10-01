#!/usr/bin/env python3
"""Run the three-stage task-generation API pipeline level by level.

This wrapper intentionally resumes missing work instead of retrying malformed
responses.  A failed stage is retried only when the provider reports a quota
or rate-limit condition; the shared pipeline skips already-valid outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


QUOTA_MARKERS = (
    "429",
    "quota",
    "insufficient_quota",
    "allocated quota",
    "quota exceeded",
    "rate limit",
    "too many requests",
)

NON_QUOTA_FAILURE_MARKERS = (
    "validation failed",
    "option_count_not_4",
    "role_mismatch",
    "state_count_mismatch",
    "empty_text",
    "empty model response",
    "read operation timed out",
    "timed out",
    "parse",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_quota_error(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in QUOTA_MARKERS)


def is_quota_only_failure(text: str) -> bool:
    """Resume only when every reported failure is quota/rate-limit related."""
    lowered = text.lower()
    if not is_quota_error(lowered):
        return False
    return not any(marker in lowered for marker in NON_QUOTA_FAILURE_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--levels", default="L3,L5,L6")
    parser.add_argument("--prompt-dir", default="private/taskgen/docs/api_pipeline_round2")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--stage1-concurrency", type=int, default=16)
    parser.add_argument("--stage2-concurrency", type=int, default=64)
    parser.add_argument("--stage3-concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--quota-wait-sec", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    bundle = args.bundle_root.resolve()
    external = args.external_root.resolve()
    taskgen = bundle / "input/taskgen_output"
    eval_input = bundle / "input/image_reuse"
    scoped_inventory = taskgen / "l1_inventory_child_scoped.json"
    tasks = read_json(taskgen / "tasks_compositional.json")
    levels = [item.strip().upper() for item in args.levels.split(",") if item.strip()]
    python = sys.executable
    module = "tongbench_taskgen.cli.highlevel_api_pipeline_v2"
    api_root = bundle / "input/api_pipeline"
    log_root = bundle / "output/api_logs"
    status_path = bundle / "output/api_scaleup_status.json"
    status = read_json(status_path) if status_path.is_file() else {"schema_version": "tongbench_api_scaleup_status_v1", "levels": {}}

    def run_command(level: str, stage: str, command: list[str], concurrency: int | None = None) -> None:
        level_log = log_root / level / f"{stage}.log"
        level_log.parent.mkdir(parents=True, exist_ok=True)
        current_concurrency = max(1, int(concurrency or 1))
        consecutive_quota = 0
        while True:
            actual = list(command)
            if "--max-concurrency" in actual:
                index = actual.index("--max-concurrency") + 1
                actual[index] = str(current_concurrency)
            result = subprocess.run(actual, cwd=repo, capture_output=True, text=True, check=False)
            with level_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage} concurrency={current_concurrency} ===\n")
                handle.write(result.stdout or "")
                if result.stderr:
                    handle.write("\n[stderr]\n")
                    handle.write(result.stderr)
            if result.returncode == 0:
                return
            latest_output = (result.stdout or "") + "\n" + (result.stderr or "")
            if not is_quota_only_failure(latest_output):
                status.setdefault("levels", {}).setdefault(level, {})[stage] = {
                    "status": "failed",
                    "log": str(level_log),
                    "error": latest_output[-3000:],
                }
                write_json(status_path, status)
                raise RuntimeError(
                    f"{level}/{stage} failed with a non-quota or mixed failure; "
                    f"inspect {level_log} and repair the failed inputs explicitly"
                )
            consecutive_quota += 1
            if consecutive_quota >= 3:
                current_concurrency = max(1, current_concurrency // 2)
                consecutive_quota = 0
            status.setdefault("levels", {}).setdefault(level, {})[stage] = {
                "status": "waiting_for_quota",
                "next_concurrency": current_concurrency,
                "wait_sec": int(args.quota_wait_sec),
                "log": str(level_log),
            }
            write_json(status_path, status)
            time.sleep(max(1, int(args.quota_wait_sec)))

    for level in levels:
        rows = [
            row for row in tasks
            if isinstance(row, dict) and str(row.get("difficulty_level") or "").upper() == level
        ]
        task_ids = ",".join(str(row["task_id"]) for row in rows)
        if not task_ids:
            raise ValueError(f"No tasks found for level {level}")
        output_root = api_root / level
        common = [
            "--project-root", str(repo),
            "--output-root", str(output_root),
            "--task-ids", task_ids,
        ]
        py = [python, "-m", module]
        export = py + ["export-inputs", *common,
            "--taskgen-output-dir", str(taskgen),
            "--l1-inventory", str(scoped_inventory),
            "--eval-input-dir", str(eval_input),
            "--target-valid-detours-per-state", "2",
            "--min-valid-detours-per-state", "0",
        ]
        api_common = [
            "--project-root", str(repo), "--output-root", str(output_root), "--task-ids", task_ids,
            "--prompt-dir", args.prompt_dir, "--model", args.model, "--base-url", args.base_url,
            "--temperature", str(args.temperature), "--max-tokens", str(args.max_tokens),
            "--timeout", str(args.timeout),
        ]
        if args.api_key_file:
            api_common += ["--api-key-file", str(Path(args.api_key_file).expanduser().resolve())]
        stage_commands = [
            ("export_inputs", export, None),
            ("stage1", py + ["run-stage1", *api_common, "--max-concurrency", str(args.stage1_concurrency)], args.stage1_concurrency),
            ("stage2", py + ["run-stage2", *api_common, "--max-concurrency", str(args.stage2_concurrency)], args.stage2_concurrency),
            ("build_stage3_inputs", py + ["build-stage3-inputs", *api_common], None),
            ("stage3", py + ["run-stage3", *api_common, "--max-concurrency", str(args.stage3_concurrency)], args.stage3_concurrency),
            ("build_final_bank", py + ["build-final-bank", *api_common], None),
        ]
        for stage, command, concurrency in stage_commands:
            status.setdefault("levels", {}).setdefault(level, {})[stage] = {"status": "running"}
            write_json(status_path, status)
            run_command(level, stage, command, concurrency)
            status["levels"][level][stage] = {"status": "completed"}
            write_json(status_path, status)

    combined = []
    for level in levels:
        path = api_root / level / "final_option_bank_all.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        combined.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    combined_path = api_root / "final_option_bank_all_levels.jsonl"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined), encoding="utf-8")
    status["combined_option_bank"] = str(combined_path)
    status["status"] = "completed"
    status["record_count"] = len(combined)
    write_json(status_path, status)
    print(json.dumps({"ok": True, "levels": levels, "record_count": len(combined), "combined_option_bank": str(combined_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
