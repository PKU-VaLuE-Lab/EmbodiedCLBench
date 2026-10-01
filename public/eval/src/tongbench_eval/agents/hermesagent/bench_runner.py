from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BENCH_CONFIG_PATH = "/tmp/hermes_bench_config.json"
POST_TASK_RESULT_PATH = "/tmp/hermes_post_task_result.json"
PRIMARY_RESULT_PATH = "/tmp/hermes_primary_result.json"
HERMES_INSTALL_DIR = "/opt/hermes"


def runtime_path(container_path: str) -> str:
    """Resolve a container path when this script is piped through native exec."""

    if os.environ.get("TONGBENCH_RUNTIME", "docker").strip().lower() != "native":
        return container_path
    # Checkpoint restore may provide an already host-mapped absolute path.
    # Keep it unchanged instead of prefixing the native task root twice.
    direct_path = Path(container_path).expanduser()
    if direct_path.is_absolute() and direct_path.exists():
        return str(direct_path)
    task_root = os.environ.get("TONGBENCH_NATIVE_TASK_ROOT", "").strip()
    if not task_root:
        raise RuntimeError("TONGBENCH_NATIVE_TASK_ROOT is missing in native mode")
    return str(Path(task_root) / container_path.lstrip("/"))


def main() -> int:
    install_dir = runtime_path(HERMES_INSTALL_DIR)
    sys.path.insert(0, install_dir)
    os.chdir(install_dir)

    from run_agent import AIAgent  # imported after install dir is added to sys.path
    data = json.loads(open(runtime_path(BENCH_CONFIG_PATH), encoding="utf-8").read())
    cfg = data["config"]
    prompt = data["prompt"]
    post_task_prompt = str(data.get("post_task_prompt") or "").strip()
    resume_session_path = str(data.get("resume_session_path") or "").strip()
    resume_without_prompt = bool(data.get("resume_without_prompt", False))
    tools_enabled = bool(data.get("tools_enabled", True))
    context_mode = str(cfg.get("context_mode", "full") or "full")
    if context_mode not in {"full", "task_compact"}:
        raise ValueError(f"Unsupported context_mode: {context_mode!r}")
    disable_context_compression = bool(cfg.get("disable_context_compression", False))
    # MCP owns task-history compaction for every harness. Do not apply a
    # second Hermes-only history transform here.
    history_compactor = None

    if resume_without_prompt and not resume_session_path:
        raise ValueError("resume_without_prompt requires resume_session_path")

    agent = AIAgent(
        model=cfg["model"],
        api_key=cfg.get("api_key") or None,
        base_url=cfg.get("base_url", ""),
        max_iterations=cfg.get("max_iterations", 90),
        enabled_toolsets=cfg.get("enabled_toolsets"),
        disabled_toolsets=cfg.get("disabled_toolsets"),
        save_trajectories=True,
        verbose_logging=True,
        reasoning_config=cfg.get("reasoning_config"),
        request_overrides=cfg.get("request_overrides"),
        # This is a benchmark-owned switch. It prevents unrelated Hermes
        # project files from entering the system prompt without changing the
        # native Hermes message loop or the task history.
        skip_context_files=(context_mode == "task_compact"),
        history_transform=history_compactor.transform if history_compactor else None,
    )
    if disable_context_compression:
        # Skill summarization must see the complete saved L2 checkpoint. The
        # application dialogue may use task_compact, but the summary branch
        # must not let Hermes drop the middle of the learning history.
        agent.compression_enabled = False
    conversation_history = None
    if resume_session_path:
        with open(resume_session_path, encoding="utf-8") as handle:
            session_payload = json.load(handle)
        conversation_history = list(session_payload.get("messages") or [])
        stored_system_prompt = session_payload.get("system_prompt")
        if context_mode == "full" and isinstance(stored_system_prompt, str) and stored_system_prompt:
            agent._cached_system_prompt = stored_system_prompt

    if not tools_enabled:
        agent.tools = []
        agent.valid_tool_names = set()

    if conversation_history is None:
        result = agent.run_conversation(prompt)
    else:
        result = agent.run_conversation(
            prompt,
            conversation_history=conversation_history,
            append_user_message=not resume_without_prompt,
        )
    task_api_calls = int(result.get("api_calls") or 0)
    with open(runtime_path(PRIMARY_RESULT_PATH), "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt": prompt,
                "final_response": result.get("final_response"),
                "completed": bool(result.get("completed")),
                "api_calls": task_api_calls,
                "error": result.get("error"),
                "context_mode": context_mode,
                "usage": {
                    # These values are accumulated from provider response
                    # usage by the unchanged Hermes agent.
                    "input_tokens": result.get("prompt_tokens", 0),
                    "uncached_input_tokens": result.get("input_tokens", 0),
                    "prompt_tokens": result.get("prompt_tokens", 0),
                    "output_tokens": result.get("output_tokens", result.get("completion_tokens", 0)),
                    "completion_tokens": result.get("completion_tokens", result.get("output_tokens", 0)),
                    "total_tokens": result.get("total_tokens", 0),
                    "cache_read_tokens": result.get("cache_read_tokens", 0),
                    "cache_write_tokens": result.get("cache_write_tokens", 0),
                    "reasoning_tokens": result.get("reasoning_tokens", 0),
                    "request_count": task_api_calls,
                    "usage_source": "provider_response_summary",
                    "history_compaction": history_compactor.summary() if history_compactor else None,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    post_task_result = None
    if post_task_prompt:
        saved_tools = agent.tools
        saved_valid_tool_names = agent.valid_tool_names
        try:
            agent.tools = []
            agent.valid_tool_names = set()
            post_task_result = agent.run_conversation(
                post_task_prompt,
                conversation_history=list(result.get("messages") or []),
            )
        finally:
            agent.tools = saved_tools
            agent.valid_tool_names = saved_valid_tool_names
        with open(runtime_path(POST_TASK_RESULT_PATH), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prompt": post_task_prompt,
                    "final_response": post_task_result.get("final_response"),
                    "completed": bool(post_task_result.get("completed")),
                    "api_calls": int(post_task_result.get("api_calls") or 0),
                    "error": post_task_result.get("error"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        result = post_task_result
    print("Completed:", result.get("completed"))
    print("Task API calls:", task_api_calls)
    if post_task_result is not None:
        print("Post-task API calls:", post_task_result.get("api_calls"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
