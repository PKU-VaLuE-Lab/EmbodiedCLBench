from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.cli.framework.dialogue import (  # noqa: E402
    UPDATED_ATOMIC_TEMPLATE_NAME,
    UPDATED_SUBTASK_TEMPLATE_NAME,
    _component_l1_ids,
    _graph_path_for_task,
    _infer_results_root,
    _load_ready_items,
    _prepare_direct_task_image_manifest,
    _resolve_default_template_path,
    _safe_slug,
    _task_level,
    _write_json,
)
from tongbench_eval.utils.python_utils import current_python  # noqa: E402
from tongbench_eval.utils.native_runtime import activate_runtime  # noqa: E402


@dataclass
class SingleTaskPlan:
    task_id: str
    level: str
    graph_path: Path
    unique_state_image_dir: Path
    output_dir: Path
    max_steps: int
    source_composite_task_ids: list[str] = field(default_factory=list)


def _task_id_from_item(item: dict[str, Any]) -> str:
    task_id = str(item.get("task_id") or "").strip()
    if task_id:
        return task_id
    task = item.get("task") if isinstance(item.get("task"), dict) else {}
    task_id = str(task.get("task_id") or "").strip()
    if task_id:
        return task_id
    raise ValueError("Ready item is missing task_id")


def _ordered_single_task_ids(items: list[dict[str, Any]], *, include_components: bool = True) -> dict[str, list[str]]:
    sources_by_task: dict[str, list[str]] = {}
    for item in items:
        composite_task_id = _task_id_from_item(item)
        task_ids = [*_component_l1_ids(item), composite_task_id] if include_components else [composite_task_id]
        for task_id in task_ids:
            if not task_id:
                continue
            sources = sources_by_task.setdefault(task_id, [])
            if composite_task_id not in sources:
                sources.append(composite_task_id)
    return sources_by_task


def _shortest_success_trace_length(task_id: str, graph_path: Path) -> int:
    success_path = graph_path.with_name(f"{task_id}_success_paths.json")
    if not success_path.is_file():
        matches = sorted(graph_path.parent.glob(f"{task_id}*success_paths.json"))
        success_path = matches[0] if matches else success_path
    if not success_path.is_file():
        return 0
    try:
        payload = json.loads(success_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    paths = payload.get("success_paths", []) if isinstance(payload, dict) else []
    lengths: list[int] = []
    for path in paths:
        if not isinstance(path, dict):
            continue
        steps = path.get("atomic_steps") or path.get("actions") or path.get("trace") or []
        if isinstance(steps, list) and steps:
            lengths.append(len(steps))
    return min(lengths, default=0)


def _max_steps_for_task(task_id: str, level: str, graph_path: Path, args: argparse.Namespace) -> int:
    extra = getattr(args, f"{str(level).lower()}_max_steps_extra", None)
    if extra is None:
        extra = getattr(args, "max_steps_extra", None)
    if extra is not None and level in {"L1", "L2", "L3", "L4", "L5", "L6"}:
        shortest = _shortest_success_trace_length(task_id, graph_path)
        if shortest > 0:
            return max(shortest, shortest + int(extra))

    factor = float(args.max_steps_factor or 0)
    if level == "L1":
        factor = float(getattr(args, "l1_max_steps_factor", 0) or factor)
    elif level == "L2":
        factor = float(getattr(args, "l2_max_steps_factor", 0) or factor)
    elif level == "L3":
        factor = float(getattr(args, "l3_max_steps_factor", 0) or factor)
    elif level == "L4":
        factor = float(getattr(args, "l4_max_steps_factor", 0) or factor)
    elif level == "L5":
        factor = float(getattr(args, "l5_max_steps_factor", 0) or factor)
    elif level == "L6":
        factor = float(getattr(args, "l6_max_steps_factor", 0) or factor)
    if factor > 0 and level in {"L1", "L2", "L3", "L4", "L5", "L6"}:
        shortest = _shortest_success_trace_length(task_id, graph_path)
        if shortest > 0:
            return max(shortest, int(math.ceil(shortest * factor)))
    if level == "L1":
        return int(args.l1_max_steps)
    if level == "L3":
        return int(args.l3_max_steps)
    if level == "L4":
        return int(args.l4_max_steps)
    if level == "L5":
        return int(args.l5_max_steps)
    if level == "L6":
        return int(args.l6_max_steps)
    return int(args.l2_max_steps)


def _build_task_plans(
    *,
    ready_items: list[dict[str, Any]],
    results_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> list[SingleTaskPlan]:
    sources_by_task = _ordered_single_task_ids(ready_items, include_components=not bool(args.composite_only))
    stage_root = output_root / "prepared_unique_state_images"
    runs_root = output_root / "task_runs"
    plans: list[SingleTaskPlan] = []

    for task_id, source_composite_task_ids in sources_by_task.items():
        level = _task_level(task_id)
        if args.l2_only and level != "L2":
            continue
        graph_path = _graph_path_for_task(results_root, task_id).resolve()
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing graph file for {task_id}: {graph_path}")
        stage_dir = stage_root / _safe_slug(task_id)
        unique_state_image_dir = _prepare_direct_task_image_manifest(
            results_root=results_root,
            task_id=task_id,
            stage_dir=stage_dir,
        )
        plans.append(
            SingleTaskPlan(
                task_id=task_id,
                level=level,
                graph_path=graph_path,
                unique_state_image_dir=unique_state_image_dir,
                output_dir=runs_root / task_id,
                max_steps=_max_steps_for_task(task_id, level, graph_path, args),
                source_composite_task_ids=source_composite_task_ids,
            )
        )
    return plans


def _maybe_add(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def _single_task_command(plan: SingleTaskPlan, args: argparse.Namespace) -> list[str]:
    script = ROOT / "scripts" / "run_tongsim_framework_agent.py"
    cmd = [
        current_python(),
        str(script),
        "--task-dir",
        str(plan.graph_path),
        "--unique-state-image-dir",
        str(plan.unique_state_image_dir),
        "--output-dir",
        str(plan.output_dir),
        "--backend",
        str(args.backend),
        "--model",
        str(args.model),
        "--runtime",
        str(args.runtime),
        "--framework-interaction-mode",
        str(args.framework_interaction_mode),
        "--public-interface-mode",
        str(args.public_interface_mode),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-steps",
        str(plan.max_steps),
        "--action-interface",
        str(args.action_interface),
        "--action-library-mode",
        str(args.action_library_mode),
        "--exclusive-in-view",
        str(args.exclusive_in_view),
    ]
    _maybe_add(cmd, "--models-config", args.models_config)
    _maybe_add(cmd, "--api-key", args.api_key)
    _maybe_add(cmd, "--base-url", args.base_url)
    _maybe_add(cmd, "--gateway-port", args.gateway_port)
    _maybe_add(cmd, "--openclaw-image-model", args.openclaw_image_model)
    _maybe_add(cmd, "--description-field", args.description_field)
    _maybe_add(cmd, "--protocol-surface", args.protocol_surface)
    _maybe_add(cmd, "--choice-count", args.choice_count)
    _maybe_add(cmd, "--choice-seed", args.choice_seed)
    _maybe_add(cmd, "--choice-bank-jsonl", args.choice_bank_jsonl)
    if not args.no_model_temperature:
        _maybe_add(cmd, "--temperature", args.temperature)
    if not args.no_model_seed:
        _maybe_add(cmd, "--seed", args.seed)
    _maybe_add(cmd, "--thinking", args.thinking)
    if args.responses_non_stream:
        cmd.append("--responses-non-stream")
    _maybe_add(cmd, "--context-mode", args.context_mode)
    _maybe_add(cmd, "--max-iterations", args.max_iterations)
    _maybe_add(cmd, "--mcp-python-command", args.mcp_python_command)
    _maybe_add(cmd, "--atomic-template-path", args.atomic_template_path)
    _maybe_add(cmd, "--subtask-template-path", args.subtask_template_path)
    if args.decision_diagnostics:
        cmd.append("--decision-diagnostics")
    if args.decision_only:
        cmd.append("--decision-only")
    if args.generic_invalid_action_feedback:
        cmd.append("--generic-invalid-action-feedback")
    if args.no_image_input:
        cmd.append("--no-image-input")
    if args.write_human_trace_md:
        cmd.append("--write-human-trace-md")
    if args.codex_direct_image_input:
        cmd.append("--codex-direct-image-input")
    if args.codex_mcp_only_tools:
        cmd.append("--codex-mcp-only-tools")
    if args.mcp_compress_images:
        cmd.append("--mcp-compress-images")
        _maybe_add(cmd, "--mcp-image-max-side", args.mcp_image_max_side)
        _maybe_add(cmd, "--mcp-image-max-base64-chars", args.mcp_image_max_base64_chars)
        _maybe_add(cmd, "--mcp-image-jpeg-quality", args.mcp_image_jpeg_quality)
    if args.export_checkpoint_root and plan.level == "L2":
        _maybe_add(
            cmd,
            "--export-checkpoint-dir",
            Path(args.export_checkpoint_root).expanduser().resolve() / plan.task_id,
        )
    return cmd


def _redact_command(command: list[str]) -> list[str]:
    redacted = list(command)
    for index, token in enumerate(redacted[:-1]):
        if token in {"--api-key"}:
            redacted[index + 1] = "***"
    return redacted


def _read_score(output_dir: Path) -> dict[str, Any] | None:
    score_path = output_dir / "score.json"
    if not score_path.exists():
        return None
    try:
        return json.loads(score_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_exported_checkpoint(output_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = output_dir / "exported_checkpoint.json"
    if not checkpoint_path.is_file():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _exported_checkpoint_is_complete(output_dir: Path) -> bool:
    payload = _read_exported_checkpoint(output_dir)
    if not payload:
        return False
    checkpoint_dir = str(payload.get("path") or payload.get("checkpoint_dir") or "").strip()
    return bool(checkpoint_dir) and (Path(checkpoint_dir).expanduser() / "agent_checkpoint.json").is_file()


def _read_existing_single_result(
    output_dir: Path,
    *,
    require_exported_checkpoint: bool = False,
) -> dict[str, Any] | None:
    result_path = output_dir / "single_task_run_result.json"
    if not result_path.exists() or not (output_dir / "agent.log").exists():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("returncode", 1) != 0:
        return None

    score = payload.get("score") or _read_score(output_dir)
    if score is None:
        return None

    # A score file is not proof that the agent actually completed.  In
    # particular, an earlier native smoke run could leave a score behind even
    # when the harness child exited with an error.  Reuse an existing run only
    # when the runner result and its usage record both agree that execution
    # reached the grading phase.
    usage_path = output_dir / "usage.json"
    if usage_path.exists():
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(usage, dict):
            return None
        if usage.get("error"):
            return None
        agent_returncode = usage.get("agent_returncode")
        if agent_returncode not in (None, 0):
            return None
    if require_exported_checkpoint and not _exported_checkpoint_is_complete(output_dir):
        return None

    return {
        **payload,
        "status": "skipped_existing",
        "skipped_existing": True,
        "existing_result_path": str(result_path.resolve()),
        "returncode": 0,
        "score": score,
        "exported_checkpoint": _read_exported_checkpoint(output_dir),
    }


def _run_plan(plan: SingleTaskPlan, args: argparse.Namespace) -> dict[str, Any]:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    command = _single_task_command(plan, args)
    metadata = {
        "task_id": plan.task_id,
        "level": plan.level,
        "graph_path": str(plan.graph_path),
        "unique_state_image_dir": str(plan.unique_state_image_dir),
        "output_dir": str(plan.output_dir),
        "max_steps": plan.max_steps,
        "source_composite_task_ids": plan.source_composite_task_ids,
        "command": _redact_command(command),
    }
    _write_json(plan.output_dir / "single_task_run_metadata.json", metadata)

    if args.dry_run:
        return {**metadata, "status": "dry_run", "returncode": None, "score": None}
    if args.skip_existing and not args.force:
        existing_result = _read_existing_single_result(
            plan.output_dir,
            require_exported_checkpoint=bool(
                args.export_checkpoint_root and plan.level == "L2"
            ),
        )
        if existing_result is not None:
            return {**metadata, **existing_result}

    start_time = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed_time = time.perf_counter() - start_time
    (plan.output_dir / "runner_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (plan.output_dir / "runner_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    result = {
        **metadata,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "elapsed_time": round(elapsed_time, 2),
        "score": _read_score(plan.output_dir),
        "exported_checkpoint": _read_exported_checkpoint(plan.output_dir),
    }
    _write_json(plan.output_dir / "single_task_run_result.json", result)
    return result


def _failed_plan_result(plan: SingleTaskPlan, exc: BaseException) -> dict[str, Any]:
    return {
        "task_id": plan.task_id,
        "level": plan.level,
        "graph_path": str(plan.graph_path),
        "unique_state_image_dir": str(plan.unique_state_image_dir),
        "output_dir": str(plan.output_dir),
        "max_steps": plan.max_steps,
        "source_composite_task_ids": plan.source_composite_task_ids,
        "status": "failed",
        "returncode": 1,
        "error": f"{type(exc).__name__}: {exc}",
        "score": _read_score(plan.output_dir),
    }


def _write_summary(output_root: Path, plan_payload: dict[str, Any], results: list[dict[str, Any]]) -> None:
    failed = [item for item in results if item.get("returncode") not in (0, None)]
    _write_json(
        output_root / "single_task_summary.json",
        {
            **plan_payload,
            "completed_count": len(results),
            "failed_count": len(failed),
            "results": results,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run every ready composite task and its component L1 tasks as independent "
            "TongSIM single-task evaluations."
        )
    )
    parser.add_argument("--ready-json", required=True, help="Path to ready task JSON.")
    parser.add_argument("--results-root", help="Root of the image-reuse result bundle. Defaults from the ready JSON location.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["hermesagent", "codex", "claudecode", "openclaw"], default="hermesagent")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--runtime",
        choices=["docker", "native"],
        default=os.environ.get("TONGBENCH_RUNTIME", "docker"),
        help="Execution runtime. docker uses the Docker daemon; native uses the bundled user-space runtime.",
    )
    parser.add_argument(
        "--description-field",
        default="description",
        help="Task JSON description field exposed to the model. Defaults to the legacy description field.",
    )
    parser.add_argument("--models-config")
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--gateway-port", type=int, help="OpenClaw gateway port. Defaults to GATEWAY_PORT or 18789.")
    parser.add_argument("--openclaw-image-model", help="Optional OpenClaw image model override. Defaults to --model.")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-model-seed", action="store_true")
    parser.add_argument("--no-model-temperature", action="store_true")
    parser.add_argument("--thinking")
    parser.add_argument("--responses-non-stream", action="store_true")
    parser.add_argument("--context-mode", choices=["full", "task_compact"], default="full")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--framework-interaction-mode", choices=["semantic_tool"], default="semantic_tool")
    parser.add_argument("--l1-max-steps", type=int, default=6)
    parser.add_argument("--l2-max-steps", type=int, default=12)
    parser.add_argument("--l3-max-steps", type=int, default=18)
    parser.add_argument("--l4-max-steps", type=int, default=24)
    parser.add_argument("--l5-max-steps", type=int, default=30)
    parser.add_argument("--l6-max-steps", type=int, default=36)
    parser.add_argument(
        "--max-steps-extra",
        type=int,
        default=None,
        help="When set, use shortest success trace length plus this many extra steps for all high-level levels.",
    )
    parser.add_argument("--l1-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L1 tasks.")
    parser.add_argument("--l2-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L2 tasks.")
    parser.add_argument("--l3-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L3 tasks.")
    parser.add_argument("--l4-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L4 tasks.")
    parser.add_argument("--l5-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L5 tasks.")
    parser.add_argument("--l6-max-steps-extra", type=int, default=None, help="Override --max-steps-extra for L6 tasks.")
    parser.add_argument(
        "--max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, set high-level task max_steps to ceil(factor * shortest success trace length).",
    )
    parser.add_argument(
        "--l1-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L1 tasks only.",
    )
    parser.add_argument(
        "--l2-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L2 tasks only.",
    )
    parser.add_argument(
        "--l3-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L3 tasks only.",
    )
    parser.add_argument(
        "--l4-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L4 tasks only.",
    )
    parser.add_argument(
        "--l5-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L5 tasks only.",
    )
    parser.add_argument(
        "--l6-max-steps-factor",
        type=float,
        default=0.0,
        help="When >0, override --max-steps-factor for L6 tasks only.",
    )
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--mcp-python-command")
    parser.add_argument("--action-interface", choices=["full_action", "factorized", "global_factorized", "library_factorized"], default="library_factorized")
    parser.add_argument("--action-library-mode", choices=["atomic_only", "subtask_only", "atomic_plus_subtask"], default="atomic_only")
    parser.add_argument("--public-interface-mode", choices=["structured", "natural_language"], default="natural_language")
    parser.add_argument("--protocol-surface", choices=["legacy", "safe_choice"], default="legacy")
    parser.add_argument("--choice-count", type=int, default=4)
    parser.add_argument("--choice-seed", type=int, default=0)
    parser.add_argument("--choice-bank-jsonl")
    parser.add_argument("--write-human-trace-md", action="store_true")
    parser.add_argument("--exclusive-in-view", choices=["true", "false"], default="true")
    parser.add_argument("--atomic-template-path")
    parser.add_argument("--subtask-template-path")
    parser.add_argument("--decision-diagnostics", action="store_true")
    parser.add_argument("--decision-only", action="store_true")
    parser.add_argument("--generic-invalid-action-feedback", action="store_true")
    parser.add_argument(
        "--no-image-input",
        action="store_true",
        help="Run a text-only language-bias control with no observation images attached.",
    )
    parser.add_argument(
        "--codex-direct-image-input",
        action="store_true",
        help=(
            "For Codex/OpenClaw single-task runs, attach observation images directly in MCP tool results "
            "instead of text-only image paths plus .wildclaw_image.py."
        ),
    )
    parser.add_argument(
        "--codex-mcp-only-tools",
        action="store_true",
        help=(
            "For the Codex backend, route Responses requests through the in-container DashScope adapter "
            "and forward only TongSIM MCP tools to the model."
        ),
    )
    parser.add_argument("--mcp-compress-images", action="store_true")
    parser.add_argument("--mcp-image-max-side", type=int, default=768)
    parser.add_argument("--mcp-image-max-base64-chars", type=int, default=70000)
    parser.add_argument("--mcp-image-jpeg-quality", type=int, default=70)
    parser.add_argument(
        "--export-checkpoint-root",
        help="Export each independently run L2 component below this directory.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip a task when its output score.json already exists.")
    parser.add_argument("--force", action="store_true", help="Run even when --skip-existing would skip.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--parallel-jobs", type=int, default=1, help="Number of independent single tasks to run concurrently.")
    parser.add_argument("--task-id", action="append", help="Only include tasks from the given composite task_id. Can be repeated.")
    parser.add_argument("--composite-only", action="store_true", help="Run only the ready composite task, not its component L1 tasks.")
    parser.add_argument(
        "--l2-only",
        action="store_true",
        help="Run only independently evaluable L2 component tasks; skip composite tasks.",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of composite tasks to include after filtering.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.runtime = activate_runtime(args.runtime)
    if args.protocol_surface == "safe_choice":
        args.action_interface = "full_action"

    ready_json = Path(args.ready_json).resolve()
    results_root = _infer_results_root(ready_json, args.results_root)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    args.atomic_template_path = _resolve_default_template_path(
        args.atomic_template_path,
        "TONGSIM_ATOMIC_TEMPLATE_PATH",
        UPDATED_ATOMIC_TEMPLATE_NAME,
        results_root,
    )
    args.subtask_template_path = _resolve_default_template_path(
        args.subtask_template_path,
        "TONGSIM_SUBTASK_TEMPLATE_PATH",
        UPDATED_SUBTASK_TEMPLATE_NAME,
        results_root,
    )

    ready_items = _load_ready_items(ready_json)
    selected_task_ids = set(args.task_id or [])
    if selected_task_ids:
        ready_items = [item for item in ready_items if _task_id_from_item(item) in selected_task_ids]
    if args.limit is not None:
        ready_items = ready_items[: int(args.limit)]
    if not ready_items:
        raise ValueError("No ready tasks selected.")
    plans = _build_task_plans(
        ready_items=ready_items,
        results_root=results_root,
        output_root=output_root,
        args=args,
    )
    plan_payload = {
        "ready_json": str(ready_json),
        "results_root": str(results_root),
        "output_root": str(output_root),
        "backend": args.backend,
        "runtime": args.runtime,
        "task_count": len(plans),
        "l2_only": args.l2_only,
        "max_steps_extra": args.max_steps_extra,
        "l1_max_steps_extra": args.l1_max_steps_extra,
        "l2_max_steps_extra": args.l2_max_steps_extra,
        "l3_max_steps_extra": args.l3_max_steps_extra,
        "l4_max_steps_extra": args.l4_max_steps_extra,
        "l6_max_steps_extra": args.l6_max_steps_extra,
        "max_steps_factor": float(args.max_steps_factor or 0),
        "l1_max_steps_factor": float(args.l1_max_steps_factor or 0),
        "l2_max_steps_factor": float(args.l2_max_steps_factor or 0),
        "l3_max_steps_factor": float(args.l3_max_steps_factor or 0),
        "l4_max_steps_factor": float(args.l4_max_steps_factor or 0),
        "l6_max_steps_factor": float(args.l6_max_steps_factor or 0),
        "atomic_template_path": str(args.atomic_template_path or ""),
        "subtask_template_path": str(args.subtask_template_path or ""),
        "export_checkpoint_root": str(args.export_checkpoint_root or ""),
        "tasks": [
            {
                "task_id": plan.task_id,
                "level": plan.level,
                "graph_path": str(plan.graph_path),
                "unique_state_image_dir": str(plan.unique_state_image_dir),
                "output_dir": str(plan.output_dir),
                "max_steps": plan.max_steps,
                "source_composite_task_ids": plan.source_composite_task_ids,
            }
            for plan in plans
        ],
    }
    _write_json(output_root / "single_task_plan.json", plan_payload)

    results: list[dict[str, Any]] = []
    parallel_jobs = max(1, int(args.parallel_jobs or 1))
    if parallel_jobs == 1:
        for index, plan in enumerate(plans, start=1):
            print(f"[{index}/{len(plans)}] {plan.task_id} ({plan.level}, max_steps={plan.max_steps})", flush=True)
            try:
                result = _run_plan(plan, args)
            except Exception as exc:
                result = _failed_plan_result(plan, exc)
            results.append(result)
            _write_summary(output_root, plan_payload, results)
            if result.get("returncode") not in (0, None) and args.stop_on_error:
                break
    else:
        print(f"Running {len(plans)} single tasks with parallel_jobs={parallel_jobs}", flush=True)
        with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {}
            for index, plan in enumerate(plans, start=1):
                print(f"[submit {index}/{len(plans)}] {plan.task_id} ({plan.level}, max_steps={plan.max_steps})", flush=True)
                futures[executor.submit(_run_plan, plan, args)] = (index, plan)
            for future in as_completed(futures):
                index, plan = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_plan_result(plan, exc)
                print(f"[done {index}/{len(plans)}] {plan.task_id} status={result.get('status')}", flush=True)
                results.append(result)
                _write_summary(output_root, plan_payload, results)

    failed = [item for item in results if item.get("returncode") not in (0, None)]
    _write_summary(output_root, plan_payload, results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
