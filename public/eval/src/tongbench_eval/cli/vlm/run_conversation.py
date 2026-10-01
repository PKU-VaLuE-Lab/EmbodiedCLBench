from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.graph_env.agents.nl_conversation_loop import (
    LocalHFQwenConversationBackend,
    MockNaturalLanguageBackend,
    OpenAICompatibleConversationBackend,
    run_natural_language_conversation_loop,
)
from tongbench_eval.graph_env.grader import grade_trace
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


class DashScopeNaturalLanguageBackend(OpenAICompatibleConversationBackend):
    name = "dashscope-qwen-nl"


def _build_backend(args: argparse.Namespace):
    if args.backend == "mock":
        scripted_choices: list[str] = []
        if args.scripted_choices:
            try:
                parsed = json.loads(args.scripted_choices)
                if isinstance(parsed, list):
                    scripted_choices = [str(item) for item in parsed]
                else:
                    scripted_choices = [str(parsed)]
            except json.JSONDecodeError:
                scripted_choices = [item.strip() for item in args.scripted_choices.split("|||") if item.strip()]
        return MockNaturalLanguageBackend(scripted_choices=scripted_choices)

    if args.backend == "openai":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        model = args.model or os.environ.get("OPENAI_MODEL")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for openai backend")
        if not model:
            raise RuntimeError("Missing --model or OPENAI_MODEL for openai backend")
        return OpenAICompatibleConversationBackend(
            base_url=args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=api_key,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if args.backend == "dashscope-qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        model = args.model or os.environ.get("DASHSCOPE_MODEL")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY for dashscope-qwen backend")
        if not model:
            raise RuntimeError("Missing --model or DASHSCOPE_MODEL for dashscope-qwen backend")
        return DashScopeNaturalLanguageBackend(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    if args.backend == "local-hf-qwen":
        return LocalHFQwenConversationBackend(
            model_path=args.model_path,
            device_map=args.device,
            torch_dtype=args.torch_dtype,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    raise ValueError(f"Unsupported backend: {args.backend}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a TongSIM VLM agent as one natural-language multi-turn task conversation"
    )
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", required=True, choices=["mock", "openai", "dashscope-qwen", "local-hf-qwen"])
    parser.add_argument("--workspace-dir")
    parser.add_argument("--action-interface", choices=["full_action", "library_factorized"], default="full_action")
    parser.add_argument("--action-library-mode", choices=["atomic_only", "subtask_only", "atomic_plus_subtask"], default="atomic_only")
    parser.add_argument("--atomic-template-path", default="tests/fixtures/atomic_templates.json")
    parser.add_argument("--subtask-template-path", default="tests/fixtures/subtask_templates.json")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--scripted-choices")
    parser.add_argument("--memory-file")
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
        memory_file=Path(args.memory_file).resolve() if args.memory_file else None,
        unique_state_image_dir=Path(args.unique_state_image_dir).resolve() if args.unique_state_image_dir else None,
    )
    backend = _build_backend(args)
    loop_result = run_natural_language_conversation_loop(
        workspace_dir=workspace_dir,
        backend=backend,
        max_steps=args.max_steps,
        action_interface=args.action_interface,
        action_library_mode=args.action_library_mode,
        atomic_template_path=Path(args.atomic_template_path).resolve() if args.atomic_template_path else None,
        subtask_template_path=Path(args.subtask_template_path).resolve() if args.subtask_template_path else None,
        decision_log_path=output_dir / "nl_decision_log.jsonl",
        conversation_log_path=output_dir / "nl_conversation.jsonl",
        prompt_log_path=output_dir / "nl_prompt_log.txt",
    )

    task_output_dir = output_dir / "task_output"
    _copy_outputs_from_workspace(workspace_dir, task_output_dir)
    _write_json(output_dir / "nl_decision_summary.json", loop_result["decision_summary"])

    trace_entries = _read_jsonl_trace(task_output_dir / "trace.jsonl")
    normalized_trace = _normalize_trace_for_grader(trace_entries)
    grade = grade_trace(task, normalized_trace)
    grade.update(
        {
            "nl_parse_error_count": loop_result["decision_summary"]["parse_error_count"],
            "nl_first_parse_error_step": loop_result["decision_summary"]["first_parse_error_step"],
        }
    )
    _write_json(task_output_dir / "grade.json", grade)

    score = _score_payload(task, grade)
    score["details"].update(
        {
            "nl_parse_error_count": grade["nl_parse_error_count"],
            "nl_first_parse_error_step": grade["nl_first_parse_error_step"],
            "conversation_turn_count": loop_result["decision_summary"]["conversation_turn_count"],
        }
    )
    _write_json(output_dir / "score.json", score)
    usage = _default_usage(time.time() - start_time, False, grade)
    usage["conversation_turn_count"] = loop_result["decision_summary"]["conversation_turn_count"]
    usage["backend"] = backend.name
    _write_json(output_dir / "usage.json", usage)
    (output_dir / "agent.log").write_text(
        json.dumps(
            {
                "backend": backend.name,
                "decision_log_path": loop_result["decision_log_path"],
                "conversation_log_path": loop_result["conversation_log_path"],
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
