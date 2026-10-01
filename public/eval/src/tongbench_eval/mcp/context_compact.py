"""Shared benchmark-owned compaction for MCP results.

The compactor removes repeated presentation text only. Task semantics,
feedback, option mappings, and image paths are protected.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


_TASK_LINE_RE = re.compile(r"^([ \\t]*Task:[ \\t]*)(.+?)[ \\t]*$", re.MULTILINE)
_DUPLICATE_RESULT_KEYS = {
    "accepted",
    "action_feedback",
    "action_rejected",
    "completed",
    "done",
    "feedback",
    "next_observation",
    "observation",
    "rejected",
    "state",
    "state_id",
    "status",
    "step_count",
    "task_complete",
}


def _serialized_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _compact_task_lines(text: str, seen: set[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        description = match.group(2).strip()
        if description in seen:
            return ""
        seen.add(description)
        return f"{match.group(1)}{description}"

    return _TASK_LINE_RE.sub(replace, text)


def _compact_observation_text(text: str, seen: set[str]) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if line.strip().lower() == "choices:":
            break
        kept.append(line)
    return _compact_task_lines("\n".join(kept), seen).strip()


def _compact_result(value: Any, seen: set[str]) -> Any:
    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    if isinstance(result.get("observation_text"), str):
        result["observation_text"] = _compact_observation_text(
            result["observation_text"], seen
        )
    if isinstance(result.get("result_text"), str):
        if _DUPLICATE_RESULT_KEYS.intersection(result):
            result.pop("result_text", None)
        else:
            result["result_text"] = _compact_task_lines(result["result_text"], seen)
    next_observation = result.get("next_observation")
    if isinstance(next_observation, dict):
        result["next_observation"] = _compact_result(next_observation, seen)
    return result


def _image_paths(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    paths = [str(item) for item in value.get("image_paths", []) or []]
    next_observation = value.get("next_observation")
    if isinstance(next_observation, dict):
        paths.extend(str(item) for item in next_observation.get("image_paths", []) or [])
    return paths


class SharedMcpCompactor:
    """Stateful per-task compactor shared by every harness backend."""

    def __init__(self, audit_path: Path | None = None) -> None:
        self.audit_path = audit_path
        self.seen_task_descriptions: set[str] = set()
        self.summary: dict[str, Any] = {
            "mode": "task_compact",
            "calls": 0,
            "before_chars": 0,
            "after_chars": 0,
            "removed_chars": 0,
            "images_before": 0,
            "images_after": 0,
            "images_unchanged": True,
            "protected_content_unchanged": True,
        }
        self.last_audit: dict[str, Any] = {}

    def transform(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise TypeError("MCP compaction expects a result object")
        before = copy.deepcopy(payload)
        transformed = _compact_result(payload, self.seen_task_descriptions)
        before_images = _image_paths(before)
        after_images = _image_paths(transformed)
        audit = {
            "before_chars": _serialized_size(before),
            "after_chars": _serialized_size(transformed),
            "removed_chars": max(
                0,
                _serialized_size(before) - _serialized_size(transformed),
            ),
            "images_before": len(before_images),
            "images_after": len(after_images),
            "images_unchanged": before_images == after_images,
            "protected_content_unchanged": all(
                transformed.get(key) == before.get(key)
                for key in (
                    "state",
                    "state_id",
                    "choices",
                    "options",
                    "action_feedback",
                    "feedback",
                    "accepted",
                    "rejected",
                    "done",
                    "task_complete",
                    "step_count",
                )
                if key in before
            ),
        }
        if not audit["images_unchanged"] or not audit["protected_content_unchanged"]:
            raise ValueError("MCP compaction changed protected task content")
        self.last_audit = audit
        self.summary["calls"] += 1
        self.summary["before_chars"] += audit["before_chars"]
        self.summary["after_chars"] += audit["after_chars"]
        self.summary["removed_chars"] += audit["removed_chars"]
        self.summary["images_before"] += audit["images_before"]
        self.summary["images_after"] += audit["images_after"]
        self.summary["images_unchanged"] = (
            self.summary["images_unchanged"] and audit["images_unchanged"]
        )
        self.summary["protected_content_unchanged"] = (
            self.summary["protected_content_unchanged"]
            and audit["protected_content_unchanged"]
        )
        self._write_audit()
        return transformed

    def _write_audit(self) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(
                {"summary": self.summary, "last": self.last_audit},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
