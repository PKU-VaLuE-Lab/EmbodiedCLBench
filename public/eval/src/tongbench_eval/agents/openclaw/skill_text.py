"""OpenClaw-specific extraction of a post-task Skill response."""

from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any


_TEXT_KEYS = {"text", "content", "message", "reply", "response", "result"}


def _collect_candidates(value: Any, candidates: list[tuple[int, str]], key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_candidates(child, candidates, str(child_key))
        return
    if isinstance(value, list):
        for child in value:
            _collect_candidates(child, candidates, key)
        return
    if isinstance(value, str) and key in _TEXT_KEYS and value.strip():
        priority = 3 if key == "text" else 2 if key in {"content", "result"} else 1
        candidates.append((priority, value.strip()))


def _parse_embedded_json(raw: str) -> list[Any]:
    parsed: list[Any] = []
    decoder = JSONDecoder()
    for index, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        parsed.append(value)
    return parsed


def _strip_outer_markdown_fence(text: str) -> str:
    lines = text.strip().splitlines()
    fence = chr(96) * 3
    if len(lines) >= 2 and lines[0].strip().startswith(fence) and lines[-1].strip() == fence:
        return "\n".join(lines[1:-1]).strip()
    return text.strip()


def extract_skill_text(raw_output: str | None) -> str:
    """Extract only OpenClaw assistant text from JSON and gateway-log output."""

    raw = str(raw_output or "").strip()
    if not raw:
        return ""

    candidates: list[tuple[int, str]] = []
    try:
        _collect_candidates(json.loads(raw), candidates)
    except json.JSONDecodeError:
        for value in _parse_embedded_json(raw):
            _collect_candidates(value, candidates)
    if not candidates:
        return raw

    highest_priority = max(priority for priority, _ in candidates)
    selected = next(text for priority, text in reversed(candidates) if priority == highest_priority)
    return _strip_outer_markdown_fence(selected)
