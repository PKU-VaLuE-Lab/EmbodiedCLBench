#!/usr/bin/env python3
"""Small stdlib-only Responses/Anthropic-to-Chat gateway for TongBench.

The gateway is intentionally limited to the text/image/function-call surface
used by the benchmark. It does not add prompts, retry requests, or route to a
different provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable


TONGSIM_TOOL_NAMES = {
    "observe",
    "select_option",
    "status",
    "observe_current_state",
    "choose_action",
    "check_task_status",
}


def _json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _content_parts(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    if content is None:
        return []
    return [{"type": "text", "text": _text(content)}]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_tongsim_tool_name(name: Any) -> bool:
    normalized = _text(name).strip().lower().replace("-", "_")
    if normalized in TONGSIM_TOOL_NAMES:
        return True
    return any(
        normalized.endswith(separator + candidate)
        for candidate in TONGSIM_TOOL_NAMES
        for separator in ("__", ".", "/")
    )


def _is_tongsim_source_tool(tool: Any) -> bool:
    if not isinstance(tool, dict):
        return False
    name = tool.get("name") or (tool.get("function") or {}).get("name")
    if _is_tongsim_tool_name(name):
        return True
    return tool.get("type") == "namespace" and _text(name).startswith("mcp__")


def _chat_content(content: Any, input_mode: bool = False) -> Any:
    parts = []
    for block in _content_parts(content):
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            raise ValueError(f"Unsupported content value: {type(block).__name__}")
        kind = str(block.get("type", "")).lower().replace("-", "_")
        if kind in {"text", "input_text", "output_text"}:
            parts.append({"type": "text", "text": _text(block.get("text", ""))})
        elif kind in {"image", "image_url", "input_image"}:
            url = block.get("image_url") or block.get("url")
            detail = block.get("detail")
            if isinstance(url, dict):
                detail = detail or url.get("detail")
                url = url.get("url")
            source = block.get("source")
            if not url and isinstance(source, dict) and source.get("type") == "url":
                url = source.get("url")
            if not url and isinstance(source, dict) and source.get("type") == "base64":
                media_type = source.get("media_type") or "image/jpeg"
                data = source.get("data") or ""
                url = f"data:{media_type};base64,{data}" if data else ""
            if url:
                image_url = {"url": url}
                if detail:
                    image_url["detail"] = detail
                parts.append({"type": "image_url", "image_url": image_url})
            else:
                raise ValueError("Image content has no usable URL or base64 source")
        else:
            raise ValueError(f"Unsupported content block: {kind}")
    if len(parts) == 1 and parts[0].get("type") == "text":
        return parts[0]["text"]
    return parts


def _tool_name(item: dict[str, Any]) -> str:
    return _text(item.get("name") or item.get("tool_name") or "")


def _responses_history_tool_name(item: dict[str, Any]) -> str:
    """Restore the flat Chat tool name from a Responses namespace call."""
    name = _tool_name(item)
    namespace = _text(item.get("namespace"))
    if namespace and not name.startswith(namespace):
        return namespace + name
    return name


def _responses_tool_identity(name: Any) -> tuple[str, str]:
    """Split a flat TongSIM Chat tool into Codex's Responses namespace form."""
    value = _text(name)
    if not value.startswith("mcp__"):
        return value, ""
    for candidate in sorted(TONGSIM_TOOL_NAMES, key=len, reverse=True):
        if value.endswith(candidate):
            namespace = value[: -len(candidate)]
            if namespace.endswith("__"):
                return candidate, namespace
    return value, ""


def _tool_arguments(item: dict[str, Any]) -> str:
    args = item.get("arguments")
    if args is None:
        args = item.get("input")
    if isinstance(args, str):
        return args
    return json.dumps(args or {}, ensure_ascii=False, separators=(",", ":"))


def _chat_role(role: Any) -> str:
    # DashScope compatible mode accepts the Chat Completions roles, but not
    # the Responses-only ``developer`` role.
    value = _text(role or "user").lower()
    return "system" if value == "developer" else value


def _chat_tools(tools: Any) -> list[dict[str, Any]]:
    result = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise ValueError(f"Unsupported tool value: {type(tool).__name__}")
        if tool.get("type") == "namespace":
            namespace = _text(tool.get("name"))
            nested_tools = tool.get("tools")
            if not namespace or not isinstance(nested_tools, list):
                raise ValueError(f"Unsupported namespace tool schema: {tool!r}")
            expanded = []
            for nested in nested_tools:
                if not isinstance(nested, dict) or not _tool_name(nested):
                    raise ValueError(f"Unsupported nested namespace tool: {nested!r}")
                expanded_tool = dict(nested)
                expanded_tool["name"] = namespace + _tool_name(nested)
                expanded.append(expanded_tool)
            result.extend(_chat_tools(expanded))
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            copied = {"type": "function", "function": dict(tool["function"])}
            if "strict" in tool and "strict" not in copied["function"]:
                copied["function"]["strict"] = tool["strict"]
            result.append(copied)
            continue
        name = tool.get("name")
        if name:
            function = {
                "name": name,
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or tool.get("input_schema") or {"type": "object"},
            }
            if "strict" in tool:
                function["strict"] = tool["strict"]
            result.append({
                "type": "function",
                "function": function,
            })
            continue
        raise ValueError(f"Unsupported tool schema: {tool!r}")
    return result


def _responses_reasoning_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in list(item.get("content") or []) + list(item.get("summary") or []):
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("reasoning_text") or block.get("summary_text")
            if text:
                parts.append(_text(text))
    if not parts and item.get("encrypted_content"):
        raise ValueError("Cannot translate opaque encrypted Responses reasoning to Chat Completions")
    return "\n".join(parts)


def _append_pending_reasoning(
    messages: list[dict[str, Any]],
    reasoning_parts: list[str],
) -> None:
    if not reasoning_parts:
        return
    messages.append({
        "role": "assistant",
        "content": None,
        "reasoning_content": "\n".join(reasoning_parts),
    })
    reasoning_parts.clear()


def _response_input_to_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []
    instructions = data.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": _chat_content(instructions)})
    value = data.get("input", "")
    if isinstance(value, str):
        return messages + [{"role": "user", "content": value}]
    for item in value or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "")).lower().replace("-", "_")
        if not kind and item.get("role") is not None:
            # Responses callers may use the role/content message shape
            # without an explicit ``type`` discriminator.
            kind = "message"
        if kind in {"message", "input_message", "output_message"}:
            role = _chat_role(item.get("role", "user"))
            if role != "assistant":
                _append_pending_reasoning(messages, pending_reasoning)
            content = _chat_content(item.get("content", ""), input_mode=True)
            message = {"role": role, "content": content}
            if role == "assistant" and pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning.clear()
            messages.append(message)
        elif kind in {"input_text", "input_image"}:
            _append_pending_reasoning(messages, pending_reasoning)
            if not messages or messages[-1].get("role") != "user":
                messages.append({"role": "user", "content": []})
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}]
            converted = _chat_content([item], input_mode=True)
            if isinstance(converted, str):
                converted = {"type": "text", "text": converted}
            if isinstance(converted, dict):
                messages[-1]["content"].append(converted)
            else:
                messages[-1]["content"].extend(converted)
        elif kind == "reasoning":
            reasoning = _responses_reasoning_text(item)
            if reasoning:
                pending_reasoning.append(reasoning)
        elif kind == "function_call":
            call = {
                "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": _responses_history_tool_name(item), "arguments": _tool_arguments(item)},
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
                if pending_reasoning:
                    prior = _text(messages[-1].get("reasoning_content"))
                    combined = [part for part in [prior, *pending_reasoning] if part]
                    messages[-1]["reasoning_content"] = "\n".join(combined)
                    pending_reasoning.clear()
            else:
                message = {"role": "assistant", "content": None, "tool_calls": [call]}
                if pending_reasoning:
                    message["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning.clear()
                messages.append(message)
        elif kind == "function_call_output":
            _append_pending_reasoning(messages, pending_reasoning)
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id") or item.get("id") or "tool_call",
                "content": _chat_content(item.get("output", "")),
            })
        else:
            raise ValueError(f"Unsupported Responses input item: {kind or '<missing type>'}")
    _append_pending_reasoning(messages, pending_reasoning)
    return messages


def _anthropic_to_messages(data: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = data.get("system")
    if system:
        messages.append({"role": "system", "content": _chat_content(system)})
    for item in data.get("messages", []) or []:
        if not isinstance(item, dict):
            continue
        role = _chat_role(item.get("role", "user"))
        content = _content_parts(item.get("content", ""))
        normal: list[Any] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                normal.append(block)
                continue
            kind = str(block.get("type", "")).lower().replace("-", "_")
            if kind == "tool_use":
                tool_calls.append({
                    "id": block.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": _tool_name(block), "arguments": _tool_arguments(block)},
                })
            elif kind == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") or "tool_call",
                    "content": _chat_content(block.get("content", "")),
                })
            elif kind == "thinking":
                reasoning_parts.append(_text(block.get("thinking", "")))
            elif kind == "redacted_thinking":
                raise ValueError("Cannot translate opaque redacted Anthropic thinking to Chat Completions")
            else:
                normal.append(block)
        if normal or tool_calls or reasoning_parts:
            message: dict[str, Any] = {"role": role, "content": _chat_content(normal)}
            if tool_calls:
                message["tool_calls"] = tool_calls
            if reasoning_parts:
                message["reasoning_content"] = "\n".join(reasoning_parts)
            messages.append(message)
    return messages


def _base_chat_request(data: dict[str, Any], messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
    if not model:
        raise ValueError("Configured gateway model is empty")
    result: dict[str, Any] = {"model": model, "messages": messages}
    for source, target in (("max_output_tokens", "max_tokens"), ("max_tokens", "max_tokens"), ("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop"), ("stop", "stop")):
        if source in data and data[source] is not None:
            result[target] = data[source]
    if data.get("stream") is not None:
        result["stream"] = bool(data["stream"])
    source_tools = data.get("tools") or []
    if _env_flag("TONGBENCH_GATEWAY_MCP_ONLY_TOOLS"):
        source_tools = [
            tool for tool in source_tools
            if _is_tongsim_source_tool(tool)
        ]
    tools = _chat_tools(source_tools)
    if _env_flag("TONGBENCH_GATEWAY_MCP_ONLY_TOOLS"):
        result["parallel_tool_calls"] = False
    elif data.get("parallel_tool_calls") is not None:
        result["parallel_tool_calls"] = bool(data["parallel_tool_calls"])
    if tools:
        result["tools"] = tools
    if data.get("tool_choice") is not None:
        choice = data["tool_choice"]
        if isinstance(choice, dict):
            choice_type = choice.get("type")
            if choice_type == "auto":
                choice = "auto"
            elif choice_type == "any":
                choice = "required"
            elif choice_type == "tool":
                choice = {"type": "function", "function": {"name": choice.get("name", "")}}
        if (
            _env_flag("TONGBENCH_GATEWAY_MCP_ONLY_TOOLS")
            and isinstance(choice, dict)
            and not _is_tongsim_tool_name((choice.get("function") or {}).get("name"))
        ):
            choice = "auto"
        result["tool_choice"] = choice
    effort = None
    reasoning = data.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    thinking = data.get("thinking")
    if isinstance(thinking, dict):
        effort = effort or thinking.get("effort")
    output_config = data.get("output_config")
    if isinstance(output_config, dict):
        effort = effort or output_config.get("effort")
    effort = os.environ.get("TONGBENCH_GATEWAY_REASONING_EFFORT") or effort
    if effort:
        # ``extra_body`` is an SDK-side convenience. The upstream HTTP body
        # must contain the provider field itself, otherwise DashScope receives
        # a literal unknown ``extra_body`` object.
        result["reasoning_effort"] = effort
        if str(result["model"]).lower().startswith("zhipu/"):
            result["enable_thinking"] = True
            result["thinking"] = {"type": "enabled", "clear_thinking": False}
    return result


def _usage(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    result = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": usage.get("total_tokens", prompt + completion),
    }
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if prompt_details is not None:
        result["input_tokens_details"] = prompt_details
    if completion_details is not None:
        result["output_tokens_details"] = completion_details
    return result


def _request_tool_names(request: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else tool.get("name")
        if name:
            names.append(_text(name))
    return names


def _source_tool_descriptors(request: dict[str, Any]) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    for tool in request.get("tools") or []:
        if not isinstance(tool, dict):
            descriptors.append({"value_type": type(tool).__name__})
            continue
        function = tool.get("function")
        nested_tools = tool.get("tools") if isinstance(tool.get("tools"), list) else []
        if not _is_tongsim_source_tool(tool):
            continue
        descriptors.append({
            "type": _text(tool.get("type")),
            "name": _text(tool.get("name")),
            "function_name": _text(function.get("name")) if isinstance(function, dict) else "",
            "namespace": _text(tool.get("namespace")),
            "nested_tools": ",".join(_tool_name(nested) for nested in nested_tools),
        })
    return descriptors


def _response_tool_names(data: dict[str, Any]) -> list[str]:
    names: list[str] = []
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name"):
            names.append(_text(function["name"]))
    return names


def _chat_to_responses(data: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        output.append({
            "type": "reasoning",
            "id": f"rs_{uuid.uuid4().hex}",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": _text(reasoning)}],
        })
    content = message.get("content")
    if content:
        output.append({"type": "message", "id": f"msg_{uuid.uuid4().hex}", "role": "assistant", "content": [{"type": "output_text", "text": _text(content), "annotations": []}]})
    calls = list(message.get("tool_calls", []) or [])
    if _env_flag("TONGBENCH_GATEWAY_SINGLE_TONGSIM_CALL") and calls:
        calls = [call for call in calls if _is_tongsim_tool_name((call.get("function") or {}).get("name"))]
        calls = calls[:1]
    for call in calls:
        fn = call.get("function") or {}
        call_id = call.get("id") or f"call_{uuid.uuid4().hex[:12]}"
        name, namespace = _responses_tool_identity(fn.get("name", ""))
        item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}", "call_id": call_id, "name": name, "arguments": _text(fn.get("arguments", "{}"))}
        if namespace:
            item["namespace"] = namespace
        output.append(item)
    return {"id": f"resp_{uuid.uuid4().hex}", "object": "response", "created_at": int(time.time()), "status": "completed", "model": model, "output": output, "usage": _usage(data)}


def _responses_stream_events(data: dict[str, Any], model: str) -> list[tuple[str, dict[str, Any]]]:
    """Materialize the Responses item/content lifecycle expected by Codex."""
    response = _chat_to_responses(data, model)
    response_id = response["id"]
    events: list[tuple[str, dict[str, Any]]] = []
    seq = 0
    def add(name: str, body: dict[str, Any]) -> None:
        nonlocal seq
        seq += 1
        body.setdefault("sequence_number", seq)
        events.append((name, body))

    add("response.created", {"type": "response.created", "response": {**response, "status": "in_progress", "output": []}})
    add("response.in_progress", {"type": "response.in_progress", "response": {**response, "status": "in_progress", "output": []}})
    for output_index, item in enumerate(response.get("output", [])):
        item_id = item.get("id") or f"item_{uuid.uuid4().hex}"
        item = {**item, "id": item_id, "status": "in_progress"}
        add("response.output_item.added", {"type": "response.output_item.added", "response_id": response_id, "output_index": output_index, "item": item})
        if item.get("type") == "message":
            part = {"type": "output_text", "text": "", "annotations": []}
            add("response.content_part.added", {"type": "response.content_part.added", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "part": part})
            text = _text((item.get("content") or [{}])[0].get("text", ""))
            if text:
                add("response.output_text.delta", {"type": "response.output_text.delta", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": text})
            add("response.output_text.done", {"type": "response.output_text.done", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text})
            add("response.content_part.done", {"type": "response.content_part.done", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "part": {**part, "text": text}})
        elif item.get("type") == "reasoning":
            text = _text((item.get("content") or [{}])[0].get("text", ""))
            if text:
                add("response.reasoning_text.delta", {"type": "response.reasoning_text.delta", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": text})
            add("response.reasoning_text.done", {"type": "response.reasoning_text.done", "response_id": response_id, "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text})
        elif item.get("type") == "function_call":
            raw = _text(item.get("arguments", "{}"))
            if raw:
                add("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "response_id": response_id, "item_id": item_id, "output_index": output_index, "delta": raw})
            add("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "response_id": response_id, "item_id": item_id, "output_index": output_index, "arguments": raw})
        add("response.output_item.done", {"type": "response.output_item.done", "response_id": response_id, "output_index": output_index, "item": {**item, "status": "completed"}})
    add("response.completed", {"type": "response.completed", "response": response})
    return events


def _chat_to_anthropic(data: dict[str, Any], model: str, request: dict[str, Any]) -> dict[str, Any]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": _text(message["content"])})
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning:
        content.insert(0, {"type": "thinking", "thinking": _text(reasoning), "signature": ""})
    for call in message.get("tool_calls", []) or []:
        fn = call.get("function") or {}
        args = fn.get("arguments", "{}")
        args = json.loads(args) if isinstance(args, str) else args
        if not isinstance(args, dict):
            raise ValueError("Upstream tool arguments must be a JSON object")
        content.append({"type": "tool_use", "id": call.get("id") or f"toolu_{uuid.uuid4().hex}", "name": fn.get("name", ""), "input": args})
    stop = (choice.get("finish_reason") or "stop")
    stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(stop, stop)
    return {"id": f"msg_{uuid.uuid4().hex}", "type": "message", "role": "assistant", "model": model, "content": content, "stop_reason": stop, "stop_sequence": None, "usage": _usage(data)}


def _anthropic_stream_events(data: dict[str, Any], model: str) -> list[tuple[str, dict[str, Any]]]:
    """Materialize a valid Anthropic stream from one completed Chat response.

    Claude Code consumes the lifecycle events, not only text deltas. Buffering
    one upstream response here keeps the adapter protocol-correct while
    preserving the exact model output and usage values.
    """
    message = _chat_to_anthropic(data, model, {})
    message_id = message["id"]
    blocks = message.get("content") or []
    usage = message.get("usage") or {}
    events: list[tuple[str, dict[str, Any]]] = []
    events.append(("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
        },
    }))
    for index, block in enumerate(blocks):
        kind = block.get("type")
        if kind == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}))
            text = _text(block.get("text", ""))
            if text:
                events.append(("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}}))
        elif kind == "thinking":
            events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": "", "signature": block.get("signature", "")}}))
            thinking = _text(block.get("thinking", ""))
            if thinking:
                events.append(("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": thinking}}))
        elif kind == "tool_use":
            events.append(("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "tool_use", "id": block.get("id", f"toolu_{uuid.uuid4().hex}"), "name": block.get("name", ""), "input": {}}}))
            raw = json.dumps(block.get("input") or {}, ensure_ascii=False, separators=(",", ":"))
            if raw:
                events.append(("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": raw}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    events.append(("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": message.get("stop_reason"), "stop_sequence": None},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }))
    events.append(("message_stop", {"type": "message_stop"}))
    return events


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, flush=True)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        raw = _json(body)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self) -> None:
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        if path in {"/health", "/v1/health"}:
            self._send_json(200, {"ok": True})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            path = urllib.parse.urlsplit(self.path).path
            if path.endswith("/responses"):
                protocol = "responses"
                messages = _response_input_to_messages(payload)
            elif path.endswith("/messages"):
                protocol = "anthropic"
                messages = _anthropic_to_messages(payload)
            else:
                self._send_json(404, {"error": "unsupported_path", "path": self.path})
                return
            request = _base_chat_request(payload, messages, os.environ.get("TONGBENCH_GATEWAY_MODEL", "ZHIPU/GLM-5.3-Flash"))
            self._call_upstream(protocol, payload, request)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(json.dumps({"event": "gateway_error", "type": "HTTPError", "status": exc.code, "body": body[:2000]}), flush=True)
            self._send_json(exc.code, {"error": "upstream_error", "body": body[:8000]})
        except Exception as exc:
            print(json.dumps({"event": "gateway_error", "type": type(exc).__name__, "message": str(exc)}), flush=True)
            self._send_json(500, {"error": type(exc).__name__, "message": str(exc)})

    def _call_upstream(self, protocol: str, original: dict[str, Any], request: dict[str, Any]) -> None:
        base = (
            os.environ.get("TONGBENCH_GATEWAY_BASE_URL")
            or os.environ.get("CODEX_BASE_URL")
            or os.environ.get("ANTHROPIC_TARGET_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or ""
        ).rstrip("/")
        key = (
            os.environ.get("TONGBENCH_GATEWAY_API_KEY")
            or os.environ.get("CODEX_API_KEY")
            or os.environ.get("ANTHROPIC_TARGET_API_KEY")
            or os.environ.get("DASHSCOPE_API_KEY")
            or ""
        )
        if not base or not key:
            raise RuntimeError("gateway target base URL or API key is missing")
        # The Anthropic SDK requires a complete named-event lifecycle. Ask
        # DashScope for one completed response and materialize that lifecycle
        # locally; this avoids dropping tool/input state in a partial adapter.
        upstream_request = dict(request)
        if protocol in {"anthropic", "responses"} and request.get("stream"):
            upstream_request["stream"] = False
        request_sha256 = hashlib.sha256(_json(upstream_request)).hexdigest()
        print(json.dumps({"event": "upstream_request", "protocol": protocol,
                          "request_sha256": request_sha256,
                          "model": upstream_request.get("model"),
                          "reasoning_effort": upstream_request.get("reasoning_effort"),
                          "enable_thinking": upstream_request.get("enable_thinking"),
                          "thinking": upstream_request.get("thinking"),
                          "message_count": len(upstream_request["messages"]),
                          "source_tools": _source_tool_descriptors(original),
                          "tool_names": _request_tool_names(upstream_request),
                          "image_count": sum(1 for m in upstream_request["messages"]
                                             for b in _content_parts(m.get("content"))
                                             if isinstance(b, dict) and b.get("type") == "image_url")}), flush=True)
        req = urllib.request.Request(base + "/chat/completions", data=_json(upstream_request), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=None) as response:
            if protocol in {"anthropic", "responses"} and request.get("stream"):
                data = json.loads(response.read().decode("utf-8"))
                print(json.dumps({"event": "upstream_usage", "id": data.get("id"), "usage": data.get("usage"), "tool_call_names": _response_tool_names(data)}), flush=True)
                if protocol == "anthropic":
                    self._send_anthropic_stream(data, request.get("model", ""))
                else:
                    self._send_responses_stream(data, request.get("model", ""))
            elif request.get("stream"):
                self._stream_response(protocol, response, request.get("model", ""))
            else:
                data = json.loads(response.read().decode("utf-8"))
                print(json.dumps({"event": "upstream_usage", "id": data.get("id"), "usage": data.get("usage"), "tool_call_names": _response_tool_names(data)}), flush=True)
                body = _chat_to_responses(data, request.get("model", "")) if protocol == "responses" else _chat_to_anthropic(data, request.get("model", ""), original)
                self._send_json(200, body)

    def _send_anthropic_stream(self, data: dict[str, Any], model: str) -> None:
        raw = b"".join(
            f"event: {event_name}\n".encode() + b"data: " + _json(event) + b"\n\n"
            for event_name, event in _anthropic_stream_events(data, model)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)
        self.wfile.flush()

    def _send_responses_stream(self, data: dict[str, Any], model: str) -> None:
        raw = b"".join(
            f"event: {event_name}\n".encode() + b"data: " + _json(event) + b"\n\n"
            for event_name, event in _responses_stream_events(data, model)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)
        self.wfile.flush()

    def _stream_response(self, protocol: str, response: Any, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            try:
                chunk = json.loads(raw)
            except Exception:
                continue
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if protocol == "responses":
                    event = {"type": "response.output_text.delta", "delta": text, "response_id": "resp_gateway"}
                else:
                    event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
                self.wfile.write(b"data: " + _json(event) + b"\n\n")
                self.wfile.flush()
            for call in delta.get("tool_calls", []) or []:
                fn = call.get("function") or {}
                if protocol == "responses":
                    event = {"type": "response.function_call_arguments.delta", "delta": _text(fn.get("arguments", "")), "item_id": call.get("id", "call_gateway")}
                else:
                    event = {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": _text(fn.get("arguments", ""))}}
                self.wfile.write(b"data: " + _json(event) + b"\n\n")
                self.wfile.flush()
        terminal = {"type": "response.completed", "response": {"id": "resp_gateway", "status": "completed", "model": model}} if protocol == "responses" else {"type": "message_stop"}
        self.wfile.write(b"data: " + _json(terminal) + b"\n\ndata: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), GatewayHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
