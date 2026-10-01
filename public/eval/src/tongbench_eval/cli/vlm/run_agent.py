from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.graph_env.agents.backends.dashscope_qwen import DashScopeQwenBackend
from tongbench_eval.graph_env.agents.backends.heuristic import HeuristicBackend
from tongbench_eval.graph_env.agents.backends.local_hf_qwen import LocalHFQwenBackend
from tongbench_eval.graph_env.agents.backends.mock import MockBackend
from tongbench_eval.graph_env.agents.backends.openai_compatible import OpenAICompatibleBackend
from tongbench_eval.graph_env.agents.vlm_loop import run_vlm_agent_loop
from tongbench_eval.graph_env.loader import load_tongsim_task
from tongbench_eval.graph_env.runner import (
    _copy_outputs_from_workspace,
    _default_usage,
    _ensure_workspace,
    _normalize_trace_for_grader,
    _read_jsonl_trace,
    _score_payload,
    _write_json,
)
from tongbench_eval.graph_env.grader import grade_trace


def _build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        scripted_actions: list[object] = []
        if args.scripted_actions:
            try:
                parsed = json.loads(args.scripted_actions)
                if isinstance(parsed, list):
                    scripted_actions = parsed
                else:
                    scripted_actions = [parsed]
            except json.JSONDecodeError:
                scripted_actions = [item for item in args.scripted_actions.split("|||") if item]
        return MockBackend(scripted_actions=scripted_actions)
    if args.backend == "heuristic":
        return HeuristicBackend()
    if args.backend == "openai":
        return OpenAICompatibleBackend(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    if args.backend == "dashscope-qwen":
        return DashScopeQwenBackend(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    if args.backend == "local-hf-qwen":
        return LocalHFQwenBackend(
            model_path=args.model_path,
            device_map=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def _augment_grade_with_decisions(grade: dict, decision_summary: dict) -> dict:
    parse_error_count = int(decision_summary.get("parse_error_count", 0))
    invalid_selection_count = int(decision_summary.get("model_selected_invalid_action_count", 0))
    summary_failure_type = decision_summary.get("model_decision_failure_type")
    if summary_failure_type in {"repeated_invalid_factorized_action", "too_many_invalid_actions"}:
        failure_type = summary_failure_type
    elif grade.get("reached_goal"):
        failure_type = "none"
    elif parse_error_count > 0:
        failure_type = "parse_error"
    elif invalid_selection_count > 0:
        failure_type = "invalid_action"
    elif grade.get("error_type") == "perception_error":
        failure_type = "wrong_object"
    elif grade.get("error_type") == "manipulation_error":
        failure_type = "wrong_action_type"
    elif grade.get("error_type") == "sequencing_error":
        failure_type = "sequencing_error"
    elif grade.get("error_type") == "exploration_or_planning_error":
        failure_type = "planning_loop"
    else:
        failure_type = "none"
    augmented = dict(grade)
    augmented.update(
        {
            "model_selected_invalid_action_count": invalid_selection_count,
            "parse_error_count": parse_error_count,
            "first_parse_error_step": decision_summary.get("first_parse_error_step"),
            "first_invalid_selection_step": decision_summary.get("first_invalid_selection_step"),
            "invalid_selected_action_ids": decision_summary.get("invalid_selected_action_ids", []),
            "model_decision_failure_type": failure_type,
        }
    )
    return augmented


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a TongSIM VLM policy agent locally")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", required=True, choices=["mock", "heuristic", "openai", "dashscope-qwen", "local-hf-qwen"])
    parser.add_argument("--workspace-dir")
    parser.add_argument("--action-interface", choices=["full_action", "factorized", "global_factorized", "library_factorized"], default="full_action")
    parser.add_argument("--action-library-mode", choices=["atomic_only", "subtask_only", "atomic_plus_subtask"], default="atomic_only")
    parser.add_argument("--atomic-template-path", default="tests/fixtures/atomic_templates.json")
    parser.add_argument("--subtask-template-path", default="tests/fixtures/subtask_templates.json")
    parser.add_argument("--debug-expose-valid-bindings", action="store_true")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--scripted-actions")
    parser.add_argument("--max-invalid-repeats", type=int, default=5)
    parser.add_argument("--max-total-invalid-actions", type=int, default=15)
    parser.add_argument("--hide-state-predicates-in-prompt", action="store_true")
    parser.add_argument("--hide-goal-state-in-prompt", action="store_true")
    parser.add_argument("--unique-state-image-dir")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    workspace_dir = Path(args.workspace_dir) if args.workspace_dir else output_dir / "workspace"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    task = load_tongsim_task(task_dir)
    _ensure_workspace(
        task_dir,
        workspace_dir,
        task,
        memory_file=None,
        unique_state_image_dir=Path(args.unique_state_image_dir).resolve() if args.unique_state_image_dir else None,
    )
    backend = _build_backend(args)
    loop_result = run_vlm_agent_loop(
        workspace_dir=workspace_dir,
        backend=backend,
        max_steps=args.max_steps,
        decision_log_path=output_dir / "decision_log.jsonl",
        prompt_log_path=output_dir / "prompt_log.txt",
        action_interface=args.action_interface,
        action_library_mode=args.action_library_mode,
        atomic_template_path=Path(args.atomic_template_path).resolve() if args.atomic_template_path else None,
        subtask_template_path=Path(args.subtask_template_path).resolve() if args.subtask_template_path else None,
        debug_expose_valid_bindings=bool(args.debug_expose_valid_bindings),
        max_invalid_repeats=args.max_invalid_repeats,
        max_total_invalid_actions=args.max_total_invalid_actions,
        include_state_predicates_in_prompt=not args.hide_state_predicates_in_prompt,
        include_goal_state_in_prompt=not args.hide_goal_state_in_prompt,
    )

    task_output_dir = output_dir / "task_output"
    _copy_outputs_from_workspace(workspace_dir, task_output_dir)
    _write_json(output_dir / "decision_summary.json", loop_result["decision_summary"])

    trace_entries = _read_jsonl_trace(task_output_dir / "trace.jsonl")
    normalized_trace = _normalize_trace_for_grader(trace_entries)
    grade = grade_trace(task, normalized_trace)
    grade = _augment_grade_with_decisions(grade, loop_result["decision_summary"])
    _write_json(task_output_dir / "grade.json", grade)

    score = _score_payload(task, grade)
    score["details"].update(
        {
            "model_selected_invalid_action_count": grade["model_selected_invalid_action_count"],
            "parse_error_count": grade["parse_error_count"],
            "first_parse_error_step": grade["first_parse_error_step"],
            "first_invalid_selection_step": grade["first_invalid_selection_step"],
            "invalid_selected_action_ids": grade["invalid_selected_action_ids"],
            "model_decision_failure_type": grade["model_decision_failure_type"],
            "not_executable_by_state_count": loop_result["decision_summary"].get("not_executable_by_state_count", 0),
            "executable_but_no_transition_count": loop_result["decision_summary"].get("executable_but_no_transition_count", 0),
            "matched_transition_count": loop_result["decision_summary"].get("matched_transition_count", 0),
            "template_executable_count": loop_result["decision_summary"].get("template_executable_count", 0),
            "template_not_executable_count": loop_result["decision_summary"].get("template_not_executable_count", 0),
            "atomic_action_count": loop_result["decision_summary"].get("atomic_action_count", 0),
            "subtask_action_count": loop_result["decision_summary"].get("subtask_action_count", 0),
            "virtual_state_count": loop_result["decision_summary"].get("virtual_state_count", 0),
            "rendered_observation_coverage": loop_result["decision_summary"].get("rendered_observation_coverage", 0.0),
            "dag_alignment_match_count": loop_result["decision_summary"].get("dag_alignment_match_count", 0),
            "dag_alignment_miss_count": loop_result["decision_summary"].get("dag_alignment_miss_count", 0),
        }
    )
    _write_json(output_dir / "score.json", score)
    usage = _default_usage(time.time() - start_time, False, grade)
    _write_json(output_dir / "usage.json", usage)
    (output_dir / "agent.log").write_text(
        json.dumps(
            {
                "backend": backend.name,
                "decision_log_path": loop_result["decision_log_path"],
                "prompt_log_path": loop_result["prompt_log_path"],
                "decision_summary_path": loop_result["decision_summary_path"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(score, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
