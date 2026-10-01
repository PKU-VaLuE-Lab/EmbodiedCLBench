#!/usr/bin/env python3
"""Translate Chat/Anthropic clients to an OpenAI Responses upstream.

This gateway is deliberately protocol-only. It preserves benchmark messages,
images, tool schemas, tool calls, tool results, reasoning text, and usage. It
does not add prompts, retry requests, or alter evaluation behavior.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tongbench_eval.agents.protocol_gateway import (
    _anthropic_stream_events,
    _anthropic_to_messages,
    _chat_to_anthropic,
    _content_parts,
    _json,
    _text,
    _usage,
)


_REASONING_STATE_PREFIX = "__TONGBENCH_RESPONSES_REASONING_STATE_V1__:"
_LOCAL_IMAGE_PATH_RE = re.compile(
    r"(?:file://)?(/[^\s\"'\\]+?\.(?:jpe?g|png|webp))",
    re.IGNORECASE,
)


def _encoded_reasoning_items(items: list[dict[str, Any]]) -> str:
    return _REASONING_STATE_PREFIX + json.dumps(
        items, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _decoded_reasoning_items(value: str) -> list[dict[str, Any]] | None:
    if not value.startswith(_REASONING_STATE_PREFIX):
        return None
    decoded = json.loads(value[len(_REASONING_STATE_PREFIX):])
    if not isinstance(decoded, list) or any(
        not isinstance(item, dict) or item.get("type") != "reasoning"
        for item in decoded
    ):
        raise ValueError("Invalid preserved Responses reasoning state")
    return decoded


def _responses_content(content: Any, *, assistant: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in _content_parts(content):
        if isinstance(block, str):
            result.append({"type": "output_text" if assistant else "input_text", "text": block})
            continue
        if not isinstance(block, dict):
            raise ValueError(f"Unsupported content value: {type(block).__name__}")
        kind = str(block.get("type", "")).lower().replace("-", "_")
        if kind in {"text", "input_text", "output_text"}:
            result.append({
                "type": "output_text" if assistant else "input_text",
                "text": _text(block.get("text", "")),
            })
            continue
        if kind in {"image", "image_url", "input_image"}:
            if assistant:
                raise ValueError("Assistant image output cannot be represented as Responses input")
            image_url = block.get("image_url") or block.get("url")
            detail = block.get("detail")
            if isinstance(image_url, dict):
                detail = detail or image_url.get("detail")
                image_url = image_url.get("url")
            source = block.get("source")
            if not image_url and block.get("data"):
                media_type = block.get("mimeType") or block.get("mime_type") or "image/jpeg"
                image_url = f"data:{media_type};base64,{block['data']}"
            if not image_url and isinstance(source, dict) and source.get("type") == "url":
                image_url = source.get("url")
            if not image_url and isinstance(source, dict) and source.get("type") == "base64":
                media_type = source.get("media_type") or "image/jpeg"
                raw = source.get("data") or ""
                image_url = f"data:{media_type};base64,{raw}" if raw else ""
            if not image_url:
                raise ValueError("Image content has no usable URL or base64 source")
            converted: dict[str, Any] = {"type": "input_image", "image_url": image_url}
            if detail:
                converted["detail"] = detail
            result.append(converted)
            continue
        raise ValueError(f"Unsupported content block: {kind}")
    return result


def _multimodal_blocks(value: Any) -> list[dict[str, Any]] | None:
    """Find MCP text/image blocks even when a client JSON-wraps them."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return _multimodal_blocks(decoded)
    if isinstance(value, list):
        direct = [item for item in value if isinstance(item, dict)]
        if any(
            str(item.get("type", "")).lower().replace("-", "_")
            in {"image", "image_url", "input_image"}
            for item in direct
        ):
            return direct
        for item in value:
            nested = _multimodal_blocks(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, dict):
        if value.get("type") in {"text", "input_text", "output_text"}:
            nested = _multimodal_blocks(value.get("text"))
            if nested is not None:
                return nested
        for key in ("structuredContent", "structured_content", "result", "content"):
            if key in value:
                nested = _multimodal_blocks(value[key])
                if nested is not None:
                    return nested
    return None


def _tool_output_content(content: Any) -> list[dict[str, Any]]:
    """Recover model content from direct or JSON-wrapped MCP image blocks."""
    blocks = _multimodal_blocks(content)
    if blocks is not None:
        return _responses_content(blocks, assistant=False)
    return _responses_content(content, assistant=False)


def _reasoning_item(text: str) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4().hex}",
        "summary": [{"type": "summary_text", "text": text}],
    }


def _chat_messages_to_responses(messages: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages or []:
        if not isinstance(message, dict):
            raise ValueError(f"Unsupported message value: {type(message).__name__}")
        role = _text(message.get("role") or "user").lower()
        if role == "tool":
            output = _tool_output_content(message.get("content", ""))
            result.append({
                "type": "function_call_output",
                "call_id": message.get("tool_call_id") or "tool_call",
                "output": output if len(output) != 1 or output[0].get("type") != "input_text" else output[0]["text"],
            })
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"Unsupported message role: {role}")
        reasoning = _text(message.get("reasoning_content") or message.get("reasoning"))
        if reasoning:
            preserved = _decoded_reasoning_items(reasoning)
            if preserved is not None:
                # Some OpenAI-compatible relays route consecutive requests to
                # different Azure resources.  Their encrypted reasoning item
                # ids are then unusable on the next request.  For those
                # explicitly opted-in gateways, keep the normal conversation
                # (messages, images, tool calls, and tool results) but omit
                # only the provider-owned encrypted reasoning state.
                if os.environ.get("TONGBENCH_GATEWAY_DROP_ENCRYPTED_REASONING") != "1":
                    result.extend(preserved)
            else:
                result.append(_reasoning_item(reasoning))
        content = _responses_content(message.get("content", ""), assistant=role == "assistant")
        if content:
            result.append({"type": "message", "role": role, "content": content})
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                raise ValueError("Chat tool call must be an object")
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            result.append({
                "type": "function_call",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "name": function.get("name") or "",
                "arguments": arguments,
            })
    return result


def _responses_tools(tools: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise ValueError(f"Unsupported tool value: {type(tool).__name__}")
        if tool.get("type") == "namespace":
            result.append(dict(tool))
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not name:
            raise ValueError(f"Tool is missing a name: {tool!r}")
        converted: dict[str, Any] = {
            "type": "function",
            "name": name,
            "description": function.get("description", ""),
            "parameters": function.get("parameters") or function.get("input_schema") or {"type": "object"},
        }
        if "strict" in function:
            converted["strict"] = function["strict"]
        elif "strict" in tool:
            converted["strict"] = tool["strict"]
        result.append(converted)
    return result


def _input_image_count(items: Any) -> int:
    def count(value: Any) -> int:
        if isinstance(value, str):
            try:
                return count(json.loads(value))
            except json.JSONDecodeError:
                return 0
        if isinstance(value, list):
            return sum(count(item) for item in value)
        if not isinstance(value, dict):
            return 0
        kind = str(value.get("type", "")).lower().replace("-", "_")
        total = int(kind in {"image", "image_url", "input_image"})
        for key in ("content", "output", "structuredContent", "structured_content", "result"):
            if key in value:
                total += count(value[key])
        return total

    return count(items or [])


def _request_content_summary(value: Any) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    data_image_strings = 0
    image_path_strings = 0

    def visit(item: Any) -> None:
        nonlocal data_image_strings, image_path_strings
        if isinstance(item, str):
            stripped = item.lstrip()
            if stripped.startswith(("{", "[")):
                try:
                    visit(json.loads(item))
                    return
                except json.JSONDecodeError:
                    pass
            data_image_strings += item.count("data:image/")
            lowered = item.lower()
            is_image_path = any(suffix in lowered for suffix in (".jpg", ".jpeg", ".png", ".webp"))
            image_path_strings += int(is_image_path)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        kind = item.get("type")
        if isinstance(kind, str) and kind:
            normalized = kind.lower().replace("-", "_")
            type_counts[normalized] = type_counts.get(normalized, 0) + 1
        for child in item.values():
            visit(child)

    visit(value)
    return {
        "content_type_counts": dict(sorted(type_counts.items())),
        "data_image_string_count": data_image_strings,
        "image_path_string_count": image_path_strings,
    }


def _image_paths_from_tool_output(value: Any) -> list[Path]:
    paths: list[Path] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            stripped = item.lstrip()
            if stripped.startswith(("{", "[")):
                try:
                    visit(json.loads(item))
                except json.JSONDecodeError:
                    pass
            for match in _LOCAL_IMAGE_PATH_RE.finditer(item):
                paths.append(Path(match.group(1)))
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, dict):
            return
        image_paths = item.get("image_paths")
        if isinstance(image_paths, list):
            for raw_path in image_paths:
                if isinstance(raw_path, str) and raw_path.strip():
                    paths.append(Path(raw_path))
        for key, child in item.items():
            if key != "image_paths":
                visit(child)

    visit(value)
    return paths


def _attach_local_tool_output_images(request: dict[str, Any]) -> int:
    if os.environ.get("TONGBENCH_GATEWAY_ATTACH_LOCAL_IMAGES", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return 0

    attached = 0
    seen: set[Path] = set()
    for item in request.get("input") or []:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        output = item.get("output")
        if _input_image_count(output):
            continue
        image_blocks: list[dict[str, Any]] = []
        for path in _image_paths_from_tool_output(output):
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError):
                continue
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            mime_type = mimetypes.guess_type(resolved.name)[0] or "image/jpeg"
            encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
            image_blocks.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
            })
        if not image_blocks:
            continue
        if isinstance(output, list):
            item["output"] = [*output, *image_blocks]
        else:
            item["output"] = [{"type": "input_text", "text": str(output or "")}, *image_blocks]
        attached += len(image_blocks)
    return attached


def _responses_tool_choice(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported tool_choice: {value!r}")
    kind = value.get("type")
    if kind == "any":
        return "required"
    if kind in {"auto", "none", "required"}:
        return kind
    if kind in {"tool", "function"}:
        function = value.get("function") or {}
        return {"type": "function", "name": value.get("name") or function.get("name") or ""}
    raise ValueError(f"Unsupported tool_choice type: {kind}")


def _responses_request(
    source: dict[str, Any],
    messages: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model,
        "input": _chat_messages_to_responses(messages),
        "stream": False,
    }
    max_tokens = source.get("max_output_tokens", source.get("max_tokens"))
    if max_tokens is not None:
        max_output_tokens = int(max_tokens)
        if max_output_tokens < 16:
            raise ValueError("Luna Responses requires max_output_tokens >= 16")
        result["max_output_tokens"] = max_output_tokens
    for key in ("temperature", "top_p", "parallel_tool_calls"):
        if source.get(key) is not None:
            result[key] = source[key]
    tools = _responses_tools(source.get("tools") or [])
    if tools:
        result["tools"] = tools
    tool_choice = _responses_tool_choice(source.get("tool_choice"))
    if tool_choice is not None:
        result["tool_choice"] = tool_choice
    effort = os.environ.get("TONGBENCH_GATEWAY_REASONING_EFFORT", "").strip()
    if not effort:
        reasoning = source.get("reasoning")
        if isinstance(reasoning, dict):
            effort = _text(reasoning.get("effort"))
        effort = effort or _text(source.get("reasoning_effort"))
    if effort and effort.lower() not in {"none", "off", "disabled"}:
        result["reasoning"] = {"effort": effort}
    return result


def _anthropic_request_to_responses(data: dict[str, Any], model: str) -> dict[str, Any]:
    messages = _anthropic_to_messages(data)
    source = dict(data)
    source["tools"] = [
        {
            "type": "function",
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object"},
            **({"strict": tool["strict"]} if "strict" in tool else {}),
        }
        for tool in data.get("tools") or []
        if isinstance(tool, dict)
    ]
    return _responses_request(source, messages, model)


def _responses_reasoning_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in list(item.get("summary") or []) + list(item.get("content") or []):
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("summary_text") or block.get("reasoning_text")
            if text:
                parts.append(_text(text))
    return "\n".join(parts)


def _responses_to_chat(data: dict[str, Any], model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_items: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            raise ValueError("Responses output item must be an object")
        kind = item.get("type")
        if kind == "reasoning":
            reasoning_items.append(dict(item))
            reasoning = _responses_reasoning_text(item)
            if reasoning:
                reasoning_parts.append(reasoning)
        elif kind == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    text_parts.append(_text(block.get("text")))
                else:
                    raise ValueError(f"Unsupported Responses message block: {block!r}")
        elif kind == "function_call":
            name = _text(item.get("name"))
            namespace = _text(item.get("namespace"))
            if namespace and not name.startswith(namespace):
                name = namespace + name
            tool_calls.append({
                "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": _text(item.get("arguments") or "{}")},
            })
        else:
            raise ValueError(f"Unsupported Responses output item: {kind}")
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
    if any(item.get("encrypted_content") for item in reasoning_items):
        message["reasoning_content"] = _encoded_reasoning_items(reasoning_items)
    elif reasoning_parts:
        message["reasoning_content"] = "\n".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    chat_usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
    }
    if usage.get("input_tokens_details") is not None:
        chat_usage["prompt_tokens_details"] = usage["input_tokens_details"]
    if usage.get("output_tokens_details") is not None:
        chat_usage["completion_tokens_details"] = usage["output_tokens_details"]
    return {
        "id": data.get("id") or f"chatcmpl_{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": chat_usage,
    }


def _chat_stream_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    base = {
        "id": data.get("id"),
        "object": "chat.completion.chunk",
        "created": data.get("created"),
        "model": data.get("model"),
    }
    events: list[dict[str, Any]] = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}]
    if message.get("reasoning_content"):
        events.append({**base, "choices": [{"index": 0, "delta": {"reasoning_content": message["reasoning_content"]}, "finish_reason": None}]})
    if message.get("content"):
        events.append({**base, "choices": [{"index": 0, "delta": {"content": message["content"]}, "finish_reason": None}]})
    for index, call in enumerate(message.get("tool_calls") or []):
        events.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, **call}]}, "finish_reason": None}]})
    events.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]})
    events.append({**base, "choices": [], "usage": data.get("usage") or {}})
    return events


class ResponsesGatewayHandler(BaseHTTPRequestHandler):
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
            path = urllib.parse.urlsplit(self.path).path.rstrip("/")
            model = os.environ.get("TONGBENCH_GATEWAY_MODEL", "gpt-5.6-luna")
            if path.endswith("/chat/completions"):
                protocol = "chat"
                request = _responses_request(payload, list(payload.get("messages") or []), model)
            elif path.endswith("/messages"):
                protocol = "anthropic"
                request = _anthropic_request_to_responses(payload, model)
            elif path.endswith("/responses"):
                protocol = "responses"
                request = dict(payload)
                request["model"] = model
            else:
                self._send_json(404, {"error": "unsupported_path", "path": self.path})
                return
            wants_stream = bool(payload.get("stream"))
            request["stream"] = protocol == "responses" and wants_stream
            self._call_upstream(protocol, payload, request, wants_stream)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(json.dumps({"event": "gateway_error", "type": "HTTPError", "status": exc.code, "body": body[:2000]}), flush=True)
            self._send_json(exc.code, {"error": "upstream_error", "body": body[:8000]})
        except Exception as exc:
            print(json.dumps({"event": "gateway_error", "type": type(exc).__name__, "message": str(exc)}), flush=True)
            self._send_json(500, {"error": type(exc).__name__, "message": str(exc)})

    def _call_upstream(
        self,
        protocol: str,
        original: dict[str, Any],
        request: dict[str, Any],
        wants_stream: bool,
    ) -> None:
        base = os.environ.get("TONGBENCH_GATEWAY_BASE_URL", "").rstrip("/")
        key = os.environ.get("TONGBENCH_GATEWAY_API_KEY", "")
        if not base or not key:
            raise RuntimeError("gateway target base URL or API key is missing")
        attached_image_count = _attach_local_tool_output_images(request)
        request_sha256 = hashlib.sha256(_json(request)).hexdigest()
        content_summary = _request_content_summary(request.get("input"))
        print(json.dumps({
            "event": "upstream_request",
            "protocol": protocol,
            "request_sha256": request_sha256,
            "model": request.get("model"),
            "reasoning": request.get("reasoning"),
            "message_count": len(request.get("input") or []),
            "tool_count": len(request.get("tools") or []),
            "image_count": _input_image_count(request.get("input")),
            "attached_local_image_count": attached_image_count,
            **content_summary,
        }), flush=True)
        upstream = urllib.request.Request(
            base + "/responses",
            data=_json(request),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(upstream, timeout=None) as response:
            if protocol == "responses" and wants_stream:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "text/event-stream"))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                while True:
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return
            data = json.loads(response.read().decode("utf-8"))
        print(json.dumps({
            "event": "upstream_usage",
            "id": data.get("id"),
            "usage": data.get("usage"),
            "output_types": [item.get("type") for item in data.get("output") or [] if isinstance(item, dict)],
        }), flush=True)
        model = request.get("model", "")
        if protocol == "responses":
            self._send_json(200, data)
            return
        chat = _responses_to_chat(data, model)
        if protocol == "chat":
            if wants_stream:
                self._send_chat_stream(chat)
            else:
                self._send_json(200, chat)
            return
        anthropic = _chat_to_anthropic(chat, model, original)
        if wants_stream:
            self._send_anthropic_stream(chat, model)
        else:
            self._send_json(200, anthropic)

    def _send_chat_stream(self, data: dict[str, Any]) -> None:
        raw = b"".join(b"data: " + _json(event) + b"\n\n" for event in _chat_stream_events(data)) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)
        self.wfile.flush()

    def _send_anthropic_stream(self, chat: dict[str, Any], model: str) -> None:
        raw = b"".join(
            f"event: {event_name}\n".encode() + b"data: " + _json(event) + b"\n\n"
            for event_name, event in _anthropic_stream_events(chat, model)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), ResponsesGatewayHandler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
