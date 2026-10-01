from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _message(role: str) -> dict[str, Any]:
    return {
        "type": "message",
        "message": {
            "role": role or "assistant",
            "content": [],
        },
    }


def _is_openclaw_message(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "message"
        and isinstance(item.get("message"), dict)
    )


def _read_rows(chat_path: Path) -> list[Any]:
    try:
        raw = chat_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    parsed = _safe_json_loads(raw)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]

    rows: list[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed_line = _safe_json_loads(line)
        if parsed_line is not None:
            rows.append(parsed_line)
    return rows


def _normalize_role_content_message(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    role = item.get("role")
    if not isinstance(role, str):
        return None

    normalized = _message(role)
    content = item.get("content", "")
    blocks = normalized["message"]["content"]

    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return normalized

    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(block.get("text", ""))})
                continue
            if block_type in ("tool_use", "toolCall"):
                tool_input = block.get("input", block.get("arguments", {}))
                if isinstance(tool_input, str):
                    parsed = _safe_json_loads(tool_input)
                    if parsed is not None:
                        tool_input = parsed
                blocks.append(
                    {
                        "type": "tool_use",
                        "name": str(block.get("name", block.get("tool_name", ""))),
                        "input": tool_input,
                    }
                )
    return normalized


def _extract_stream_event(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("event") != "query_yield":
        return None
    payload = row.get("payload")
    if not isinstance(payload, dict):
        return None
    payload_message = payload.get("message")
    if not isinstance(payload_message, dict):
        return None
    if payload_message.get("type") != "stream_event":
        return None
    stream_event = payload_message.get("event")
    if isinstance(stream_event, dict):
        return stream_event
    return None


def _to_openclaw_usage(usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": int(usage.get("input_tokens", 0) or 0),
        "output": int(usage.get("output_tokens", 0) or 0),
        "cacheRead": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cacheWrite": int(usage.get("cache_creation_input_tokens", 0) or 0),
        "totalTokens": (
            int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0)
            + int(usage.get("cache_read_input_tokens", 0) or 0)
            + int(usage.get("cache_creation_input_tokens", 0) or 0)
        ),
        "cost": {"total": float(usage.get("cost_usd", usage.get("total_cost_usd", 0.0)) or 0.0)},
    }


def _normalize_claude_event_rows(rows: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    current_role: str | None = None
    blocks_by_index: dict[int, dict[str, Any]] = {}
    tool_input_deltas: dict[int, str] = {}
    last_assistant_index: int | None = None

    def flush_current() -> None:
        nonlocal current_role, blocks_by_index, tool_input_deltas, last_assistant_index
        if current_role is None:
            blocks_by_index = {}
            tool_input_deltas = {}
            return

        message = _message(current_role)
        content = message["message"]["content"]
        for index in sorted(blocks_by_index.keys()):
            block = blocks_by_index[index]
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if isinstance(text, str) and text:
                    content.append({"type": "text", "text": text})
            elif block_type in ("tool_use", "toolCall"):
                content.append(
                    {
                        "type": "tool_use",
                        "name": str(block.get("name", "")),
                        "input": block.get("input", {}),
                    }
                )
            elif block_type == "tool_result":
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(block.get("tool_use_id", "")),
                        "content": block.get("content", ""),
                    }
                )

        if content:
            normalized.append(message)
            if current_role == "assistant":
                last_assistant_index = len(normalized) - 1

        current_role = None
        blocks_by_index = {}
        tool_input_deltas = {}

    for row in rows:
        if not isinstance(row, dict):
            continue

        if row.get("event") == "query_end":
            payload = row.get("payload")
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("usage"), dict)
                and last_assistant_index is not None
            ):
                normalized[last_assistant_index]["message"]["usage"] = _to_openclaw_usage(payload["usage"])
            continue

        stream_event = _extract_stream_event(row)
        if not isinstance(stream_event, dict):
            continue

        event_type = stream_event.get("type")
        if event_type == "message_start":
            flush_current()
            message_payload = stream_event.get("message")
            if isinstance(message_payload, dict):
                current_role = str(message_payload.get("role", "assistant"))
            else:
                current_role = "assistant"
            continue

        if current_role is None:
            continue

        if event_type == "content_block_start":
            index = int(stream_event.get("index", 0))
            content_block = stream_event.get("content_block")
            if not isinstance(content_block, dict):
                continue
            block_type = content_block.get("type")
            if block_type == "text":
                blocks_by_index[index] = {"type": "text", "text": str(content_block.get("text", ""))}
            elif block_type == "tool_use":
                blocks_by_index[index] = {
                    "type": "tool_use",
                    "name": str(content_block.get("name", "")),
                    "input": content_block.get("input", {}),
                }
                if "input" not in content_block or content_block.get("input") in ({}, None, ""):
                    tool_input_deltas[index] = ""
            elif block_type == "tool_result":
                blocks_by_index[index] = {
                    "type": "tool_result",
                    "tool_use_id": str(content_block.get("tool_use_id", "")),
                    "content": content_block.get("content", ""),
                }
            continue

        if event_type == "content_block_delta":
            index = int(stream_event.get("index", 0))
            delta = stream_event.get("delta")
            if not isinstance(delta, dict):
                continue
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block = blocks_by_index.get(index)
                if not isinstance(block, dict) or block.get("type") != "text":
                    block = {"type": "text", "text": ""}
                    blocks_by_index[index] = block
                block["text"] = str(block.get("text", "")) + str(delta.get("text", ""))
            elif delta_type == "input_json_delta":
                block = blocks_by_index.get(index)
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    block = {"type": "tool_use", "name": "", "input": {}}
                    blocks_by_index[index] = block
                tool_input_deltas[index] = tool_input_deltas.get(index, "") + str(
                    delta.get("partial_json", "")
                )
            continue

        if event_type == "content_block_stop":
            index = int(stream_event.get("index", 0))
            partial = tool_input_deltas.pop(index, "")
            if not partial:
                continue
            parsed = _safe_json_loads(partial)
            if parsed is None:
                parsed = {"raw": partial}
            block = blocks_by_index.get(index)
            if isinstance(block, dict) and block.get("type") == "tool_use":
                block["input"] = parsed
            continue

        if event_type == "message_stop":
            flush_current()

    flush_current()
    return normalized


def convert_claudecode_chat_to_openclaw_jsonl(chat_path: Path, output_path: Path) -> int:
    rows = _read_rows(chat_path)
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return 0

    openclaw_rows = [row for row in rows if _is_openclaw_message(row)]
    if openclaw_rows:
        normalized = openclaw_rows
    else:
        normalized = _normalize_claude_event_rows(rows)
        if not normalized:
            role_messages: list[dict[str, Any]] = []
            for row in rows:
                normalized_row = _normalize_role_content_message(row)
                if normalized_row is not None:
                    role_messages.append(normalized_row)
            normalized = role_messages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ""
    if normalized:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in normalized) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    return len(normalized)
