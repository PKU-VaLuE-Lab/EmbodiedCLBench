from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from tongbench_eval.agents.base import AgentCheckpoint


NEXT_TASK_MESSAGE = (
    "The preceding learning example ended successfully. This result belongs to the learning "
    "context, not the new task. A new task is now available; call observe to begin it."
)
NEXT_TASK_AFTER_FAILURE_MESSAGE = (
    "The preceding learning example ended before completion. This result belongs to the learning "
    "context, not the new task. A new task is now available; call observe to begin it."
)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _latest_hermes_session(root: Path) -> Path:
    sessions = sorted(root.rglob("session_*.json"), key=lambda path: path.stat().st_mtime)
    if not sessions:
        raise FileNotFoundError(f"No Hermes native session JSON found under {root}")
    return sessions[-1]


def _assistant_has_visible_text(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(part, str) and part.strip()
            or isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and part["text"].strip()
            for part in content
        )
    return False


def _rewrite_completed_terminal_as_next_task(
    content: Any,
    *,
    task_completed: bool | None = None,
) -> Any:
    """Turn a sequence terminal result into a resumable next-task result.

    New protocol sessions end with the terminal action result itself. Older
    sessions may still contain a finish result, so this deliberately keys off
    the sequence marker rather than a tool name.
    """
    content_blocks = None
    if isinstance(content, list):
        content_blocks = content
        outer = None
        for block in content:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                continue
            try:
                candidate = json.loads(block["text"])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                outer = candidate
                break
        if outer is None:
            raise ValueError("Hermes terminal tool content has no JSON text block.")
    elif isinstance(content, dict):
        outer = dict(content)
    elif isinstance(content, str):
        try:
            outer = json.loads(content)
        except Exception as exc:
            raise ValueError("Hermes terminal tool content must be valid JSON.") from exc
    else:
        raise ValueError("Hermes terminal tool content must be a JSON object or string.")
    if not isinstance(outer, dict):
        raise ValueError("Hermes terminal tool content is not a JSON object.")

    structured = outer.get("structuredContent")
    result = outer.get("result")
    if isinstance(structured, dict):
        terminal = structured
    elif isinstance(result, dict):
        terminal = result
    elif isinstance(result, str):
        try:
            parsed_result = json.loads(result)
        except Exception:
            parsed_result = None
        terminal = parsed_result if isinstance(parsed_result, dict) else None
    else:
        terminal = None
    if terminal is None and (
        outer.get("all_tasks_complete") is True or outer.get("done") is True
    ):
        terminal = outer
    if not isinstance(terminal, dict):
        raise ValueError("Hermes checkpoint does not end at a terminal result.")
    if terminal.get("all_tasks_complete") is not True and terminal.get("done") is not True:
        raise ValueError("Hermes checkpoint does not end at a completed or ended task result.")

    if task_completed is None:
        task_completed = terminal.get("all_tasks_complete") is True

    visible = {
        key: value
        for key, value in terminal.items()
        if key not in {"reached_goal", "confidence"}
    }
    visible.update(
        {
            # `done=true` means the preceding simulator episode ended.  Once
            # this record is used as a continuation boundary, leaving it true
            # makes some models stop instead of starting the next task.
            "done": False,
            "task_complete": bool(task_completed),
            "all_tasks_complete": False,
            "message": (
                NEXT_TASK_MESSAGE
                if task_completed
                else NEXT_TASK_AFTER_FAILURE_MESSAGE
            ),
        }
    )
    if "structuredContent" in outer:
        outer["structuredContent"] = visible
    if "result" in outer or "structuredContent" in outer:
        outer["result"] = json.dumps(visible, indent=2, ensure_ascii=False)
    else:
        outer = visible
    rewritten = json.dumps(outer, ensure_ascii=False)
    if content_blocks is None:
        return rewritten
    rewritten_blocks = [dict(block) if isinstance(block, dict) else block for block in content_blocks]
    for block in rewritten_blocks:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] = rewritten
            break
    return rewritten_blocks


def _contains_terminal_marker(content: Any) -> bool:
    """Return whether a tool message contains a completed/ended task result."""
    candidates: list[Any] = []
    if isinstance(content, dict):
        candidates.append(content)
    elif isinstance(content, str):
        try:
            candidates.append(json.loads(content))
        except json.JSONDecodeError:
            return False
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                try:
                    candidates.append(json.loads(block["text"]))
                except json.JSONDecodeError:
                    continue
    return any(
        isinstance(candidate, dict)
        and (candidate.get("done") is True or candidate.get("all_tasks_complete") is True)
        for candidate in candidates
    )


def _checkpoint_last_task_completed(source: AgentCheckpoint) -> bool | None:
    """Read the completion flag for the final independent source task, if present."""
    for filename in ("task_outcomes.json", "task_outcome.json"):
        path = source.path / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if filename == "task_outcomes.json":
            if isinstance(payload, list) and payload and isinstance(payload[-1], dict):
                value = payload[-1].get("completed")
                if isinstance(value, bool):
                    return value
        elif isinstance(payload, dict):
            value = payload.get("completed")
            if isinstance(value, bool):
                return value
    return None


def build_hermes_in_context_checkpoint(
    *,
    source: AgentCheckpoint,
    destination: Path,
) -> AgentCheckpoint:
    """Create the ICL continuation view without changing the learning trajectory."""
    if source.backend != "hermesagent":
        raise ValueError(f"Expected a Hermes checkpoint, got {source.backend!r}.")
    destination = destination.resolve()
    if destination.exists():
        try:
            existing = AgentCheckpoint.load(destination, expected_backend="hermesagent")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            shutil.rmtree(destination)
        else:
            if existing.metadata.get("source_checkpoint_payload_sha256") == source.payload_sha256:
                return existing
            shutil.rmtree(destination)

    shutil.copytree(source.path / "native", destination / "native")
    session_path = _latest_hermes_session(destination / "native" / "sessions")
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    messages = list(payload.get("messages") or [])
    if len(messages) < 2:
        raise ValueError("Hermes learning checkpoint is too short to derive an ICL continuation.")
    final_message = messages[-1]
    if isinstance(final_message, dict) and final_message.get("role") == "assistant":
        if not _assistant_has_visible_text(final_message):
            raise ValueError("Hermes learning checkpoint has no completion result.")
        messages.pop()
    terminal_message = messages[-1]
    if not isinstance(terminal_message, dict) or terminal_message.get("role") != "tool":
        raise ValueError("Hermes learning checkpoint has no final terminal tool result.")

    task_completed = _checkpoint_last_task_completed(source)
    terminal_content = terminal_message.get("content")
    # A failed learning example may end on an ordinary observation containing
    # text and an image rather than a terminal JSON result. Preserve that
    # observation verbatim; the next-task boundary is added by the continuation
    # runtime and the learning history remains intact.
    if task_completed is not False or _contains_terminal_marker(terminal_content):
        terminal_message["content"] = _rewrite_completed_terminal_as_next_task(
            terminal_content,
            task_completed=task_completed,
        )
    payload["messages"] = messages
    payload["message_count"] = len(messages)
    session_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return AgentCheckpoint.create(
        backend="hermesagent",
        path=destination,
        session_id=source.session_id,
        metadata={
            **source.metadata,
            "checkpoint_variant": "in_context_continuation",
            "source_checkpoint_payload_sha256": source.payload_sha256,
            "boundary_transform": "drop_completion_summary_and_expose_next_task",
        },
    )


def validate_hermes_branch_prefix(
    *,
    checkpoint: AgentCheckpoint,
    branch_output_dir: Path,
    branch_prompt: str | None,
    tools_enabled: bool = True,
) -> dict[str, Any]:
    checkpoint_session = _latest_hermes_session(checkpoint.path / "native" / "sessions")
    branch_session = _latest_hermes_session(branch_output_dir / "hermes_session")
    expected = json.loads(checkpoint_session.read_text(encoding="utf-8"))
    actual = json.loads(branch_session.read_text(encoding="utf-8"))
    expected_messages = list(expected.get("messages") or [])
    actual_messages = list(actual.get("messages") or [])
    actual_prefix = actual_messages[: len(expected_messages)]
    appended_message = (
        actual_messages[len(expected_messages)]
        if len(actual_messages) > len(expected_messages)
        else None
    )
    expected_history = {
        "system_prompt": expected.get("system_prompt"),
        "messages": expected_messages,
    }
    actual_history = {
        "system_prompt": actual.get("system_prompt"),
        "messages": actual_prefix,
    }
    expected_tools = (expected.get("tools") or []) if tools_enabled else []
    actual_tools = actual.get("tools") or []
    history_equal = expected_history == actual_history
    tools_equal = expected_tools == actual_tools
    expected_input = {**expected_history, "tools": expected_tools}
    actual_input = {**actual_history, "tools": actual_tools}
    prefix_equal = history_equal and tools_equal
    continuation_started = appended_message is not None
    if branch_prompt is None:
        prompt_equal = (
            isinstance(appended_message, dict)
            and appended_message.get("role") != "user"
        )
    else:
        prompt_equal = (
            isinstance(appended_message, dict)
            and appended_message.get("role") == "user"
            and appended_message.get("content") == branch_prompt
        )
    return {
        "schema_version": "tongbench_checkpoint_branch_validation_v1",
        "backend": "hermesagent",
        "checkpoint_payload_sha256": checkpoint.payload_sha256,
        "checkpoint_session": str(checkpoint_session.resolve()),
        "branch_session": str(branch_session.resolve()),
        "checkpoint_message_count": len(expected_messages),
        "branch_message_count": len(actual_messages),
        "expected_model_input_prefix_sha256": _canonical_sha256(expected_input),
        "actual_model_input_prefix_sha256": _canonical_sha256(actual_input),
        "history_prefix_exactly_equal": history_equal,
        "tools_exactly_equal": tools_equal,
        "tools_enabled": bool(tools_enabled),
        "prefix_exactly_equal": prefix_equal,
        "appended_prompt_exactly_equal": prompt_equal,
        "continued_without_user_message": bool(branch_prompt is None and prompt_equal),
        "equivalent": bool(prefix_equal and prompt_equal and continuation_started),
    }


def validate_checkpoint_branch(
    *,
    checkpoint: AgentCheckpoint,
    branch_output_dir: Path,
    branch_prompt: str | None,
    tools_enabled: bool = True,
) -> dict[str, Any]:
    if checkpoint.backend == "hermesagent":
        return validate_hermes_branch_prefix(
            checkpoint=checkpoint,
            branch_output_dir=branch_output_dir,
            branch_prompt=branch_prompt,
            tools_enabled=tools_enabled,
        )
    return {
        "schema_version": "tongbench_checkpoint_branch_validation_v1",
        "backend": checkpoint.backend,
        "checkpoint_payload_sha256": checkpoint.payload_sha256,
        "equivalent": None,
        "verification": "native_checkpoint_hash_and_explicit_session_resume",
        "note": "Exact message-prefix inspection is currently available for Hermes native sessions.",
    }
