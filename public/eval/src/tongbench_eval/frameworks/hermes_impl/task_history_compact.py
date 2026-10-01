"""Benchmark-owned compaction for Hermes task histories.

The compact mode removes only redundant presentation and internal bookkeeping.
Task content, images, choices, actions, feedback, state, tool calls, and
assistant reasoning remain in the history.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any


_TASK_LINE_RE = re.compile(r"^([ \t]*Task:[ \t]*)(.+?)[ \t]*$", re.MULTILINE)
_DUPLICATE_RESULT_KEYS = {
    "accepted",
    "action_feedback",
    "action_rejected",
    "completed",
    "done",
    "feedback",
    "observation",
    "rejected",
    "state",
    "state_id",
    "status",
    "step_count",
    "task_complete",
}


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compact_text(text: str, seen_task_descriptions: set[str]) -> str:
    """Remove duplicate Task lines while preserving all other text."""

    def replace_task(match: re.Match[str]) -> str:
        prefix = match.group(1)
        description = match.group(2).strip()
        if description in seen_task_descriptions:
            return ""
        seen_task_descriptions.add(description)
        return f"{prefix}{description}"

    return _TASK_LINE_RE.sub(replace_task, text)


def _collect_task_descriptions(content: Any, seen_task_descriptions: set[str]) -> None:
    blocks = content if isinstance(content, list) else [content]
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) and block.get("type") == "text" else block
        if not isinstance(text, str):
            continue
        for match in _TASK_LINE_RE.finditer(text):
            seen_task_descriptions.add(match.group(2).strip())


def _compact_observation_text(text: str, seen_task_descriptions: set[str]) -> str:
    """Keep task/state context but remove the choices text duplicated by choices."""

    lines = text.splitlines()
    compacted_lines: list[str] = []
    for line in lines:
        if line.strip().lower() == "choices:":
            break
        compacted_lines.append(line)
    return _compact_text("\n".join(compacted_lines), seen_task_descriptions).strip()


def _compact_result_payload(payload: Any, seen_task_descriptions: set[str]) -> Any:
    """Compact one decoded tool result while preserving decision fields."""

    if not isinstance(payload, dict):
        return payload

    result = copy.deepcopy(payload)
    if isinstance(result.get("observation_text"), str):
        result["observation_text"] = _compact_observation_text(
            result["observation_text"], seen_task_descriptions
        )

    next_observation = result.get("next_observation")
    if isinstance(next_observation, dict):
        next_observation = copy.deepcopy(next_observation)
        if isinstance(next_observation.get("observation_text"), str):
            next_observation["observation_text"] = _compact_observation_text(
                next_observation["observation_text"], seen_task_descriptions
            )
        result["next_observation"] = next_observation

    # Structured feedback is authoritative. The display-only result_text is
    # redundant only when the same response has structured state or feedback.
    if result.get("result_text") and _DUPLICATE_RESULT_KEYS.intersection(result):
        result.pop("result_text", None)
    return result


def _compact_tool_text(text: str, seen_task_descriptions: set[str]) -> str:
    """Compact direct and MCP-wrapper JSON tool text."""

    compacted = _compact_text(text, seen_task_descriptions)
    try:
        payload = json.loads(compacted)
    except (TypeError, json.JSONDecodeError):
        return compacted

    if not isinstance(payload, dict):
        return compacted

    # MCP multimodal results are exposed to the model as an outer JSON object
    # whose result value is another JSON object. Compact the inner object
    # rather than treating the wrapper as opaque.
    nested_result = payload.get("result")
    if isinstance(nested_result, str):
        try:
            nested_payload = json.loads(nested_result)
        except (TypeError, json.JSONDecodeError):
            nested_payload = None
        if isinstance(nested_payload, dict):
            payload["result"] = json.dumps(
                _compact_result_payload(nested_payload, seen_task_descriptions),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    payload = _compact_result_payload(payload, seen_task_descriptions)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _compact_content(content: Any, seen_task_descriptions: set[str]) -> Any:
    if isinstance(content, str):
        return _compact_tool_text(content, seen_task_descriptions)
    if not isinstance(content, list):
        return content

    result = copy.deepcopy(content)
    for block in result:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        if isinstance(block.get("text"), str):
            block["text"] = _compact_tool_text(block["text"], seen_task_descriptions)
    return result


class TaskHistoryCompactor:
    """Callable history transform shared by Basic, Zero-shot, ICL, and Skill."""

    def __init__(self) -> None:
        self._summary = {
            "calls": 0,
            "before_chars": 0,
            "after_chars": 0,
            "removed_chars": 0,
            "images_before": 0,
            "images_after": 0,
            "images_unchanged": True,
            "reasoning_preserved": True,
        }
        self.last_audit: dict[str, Any] = {}

    @staticmethod
    def _image_fingerprints(messages: list[dict[str, Any]]) -> list[str]:
        fingerprints: list[str] = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            blocks = content if isinstance(content, list) else [content]
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") not in {"image", "image_url", "input_image"}:
                    continue
                if "image_url" in block:
                    fingerprints.append(_json_fingerprint(block["image_url"]))
                elif "source" in block:
                    fingerprints.append(_json_fingerprint(block["source"]))
                else:
                    fingerprints.append(_json_fingerprint(block))
        return fingerprints

    @staticmethod
    def _serialized_chars(messages: list[dict[str, Any]]) -> int:
        return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))

    def transform(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(messages, list):
            raise TypeError("history_transform expects a list of messages")

        before = copy.deepcopy(messages)
        before_images = self._image_fingerprints(before)
        seen_task_descriptions: set[str] = set()
        for message in before:
            if isinstance(message, dict) and message.get("role") != "tool":
                _collect_task_descriptions(message.get("content"), seen_task_descriptions)

        transformed: list[dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                transformed.append(copy.deepcopy(message))
                continue

            item = copy.deepcopy(message)
            if item.get("role") == "assistant":
                # These are internal transcript bookkeeping fields. Reasoning,
                # content, and tool_calls are intentionally retained.
                item.pop("finish_reason", None)
                item.pop("_thinking_prefill", None)
            if item.get("role") == "tool":
                item["content"] = _compact_content(item.get("content"), seen_task_descriptions)
            transformed.append(item)

        after_images = self._image_fingerprints(transformed)
        audit = {
            "before_chars": self._serialized_chars(before),
            "after_chars": self._serialized_chars(transformed),
            "removed_chars": max(
                0,
                self._serialized_chars(before) - self._serialized_chars(transformed),
            ),
            "message_count": len(transformed),
            "images_before": len(before_images),
            "images_after": len(after_images),
            "images_unchanged": before_images == after_images,
            "reasoning_preserved": all(
                not isinstance(old, dict)
                or old.get("role") != "assistant"
                or old.get("reasoning") == new.get("reasoning")
                for old, new in zip(before, transformed)
            ),
        }
        if not audit["images_unchanged"] or not audit["reasoning_preserved"]:
            raise ValueError("task history compaction changed protected content")

        messages[:] = transformed
        self.last_audit = audit
        self._summary["calls"] += 1
        self._summary["before_chars"] += audit["before_chars"]
        self._summary["after_chars"] += audit["after_chars"]
        self._summary["removed_chars"] += audit["removed_chars"]
        self._summary["images_before"] += audit["images_before"]
        self._summary["images_after"] += audit["images_after"]
        self._summary["images_unchanged"] = (
            self._summary["images_unchanged"] and audit["images_unchanged"]
        )
        self._summary["reasoning_preserved"] = (
            self._summary["reasoning_preserved"] and audit["reasoning_preserved"]
        )
        return messages

    def summary(self) -> dict[str, Any]:
        return dict(self._summary)
