from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

HERMES_HOME = "/root/.hermes"
HERMES_INSTALL_DIR = "/opt/hermes"
OUTPUT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"


def runtime_path(container_path: str) -> str:
    if os.environ.get("TONGBENCH_RUNTIME", "docker").strip().lower() != "native":
        return container_path
    task_root = os.environ.get("TONGBENCH_NATIVE_TASK_ROOT", "").strip()
    if not task_root:
        raise RuntimeError("TONGBENCH_NATIVE_TASK_ROOT is missing in native mode")
    return str(Path(task_root) / container_path.lstrip("/"))

def _load_messages(sessions_dir: str, trajectory_jsonl: str) -> list[dict[str, Any]]:
    session_files = sorted(glob.glob(os.path.join(sessions_dir, "session_*.json")), key=os.path.getmtime)
    if session_files:
        merged: list[dict[str, Any]] = []
        for session_file in session_files:
            try:
                payload = json.loads(Path(session_file).read_text(encoding="utf-8"))
            except Exception:
                continue
            messages = payload.get("messages", [])
            if isinstance(messages, list):
                merged.extend(item for item in messages if isinstance(item, dict))
        return merged

    trajectory_path = Path(trajectory_jsonl)
    if not trajectory_path.exists():
        return []
    try:
        lines = trajectory_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    if not lines:
        return []
    try:
        payload = json.loads(lines[-1])
    except Exception:
        return []
    conversations = payload.get("conversations", [])
    if isinstance(conversations, list):
        return [item for item in conversations if isinstance(item, dict)]
    return []


def _assistant_entry(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    tool_calls = message.get("tool_calls")
    usage = message.get("usage", {})

    if not isinstance(tool_calls, list) or not tool_calls:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": content,
                "usage": usage if isinstance(usage, dict) else {},
            },
        }

    content_blocks: list[dict[str, Any]] = []
    if isinstance(content, str) and content.strip():
        content_blocks.append({"type": "text", "text": content})

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {}
        raw_arguments = function_payload.get("arguments", {})
        parsed_arguments: Any = raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except Exception:
                parsed_arguments = raw_arguments
        content_blocks.append(
            {
                "type": "tool_use",
                "name": str(function_payload.get("name", "")),
                "input": parsed_arguments,
                "id": str(tool_call.get("id", "")),
            }
        )

    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": content_blocks,
            "usage": usage if isinstance(usage, dict) else {},
        },
    }


def _user_entry(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "type": "message",
        "message": {
            "role": "user",
            "content": content,
        },
    }


def _tool_entry(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "type": "toolResult",
        "toolResult": {
            "content": content,
            "tool_call_id": str(message.get("tool_call_id", "")),
        },
    }


def _to_openclaw_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", ""))
        if role == "assistant":
            converted.append(_assistant_entry(message))
            continue
        if role == "user":
            converted.append(_user_entry(message))
            continue
        if role == "tool":
            converted.append(_tool_entry(message))
    return converted


def main() -> int:
    output_path = Path(runtime_path(OUTPUT_TRANSCRIPT_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_messages = _load_messages(
        sessions_dir=runtime_path(f"{HERMES_HOME}/sessions"),
        trajectory_jsonl=runtime_path(f"{HERMES_INSTALL_DIR}/trajectory_samples.jsonl"),
    )
    converted = _to_openclaw_messages(source_messages)
    payload = ""
    if converted:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in converted) + "\n"
    output_path.write_text(payload, encoding="utf-8")

    print(
        f"Wrote compat transcript to {output_path} "
        f"({len(converted)} messages from {len(source_messages)} source items)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
