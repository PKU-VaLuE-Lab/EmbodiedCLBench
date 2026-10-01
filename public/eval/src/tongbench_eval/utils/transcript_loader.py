from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OPENCLAW_FALLBACK_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"


def _safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_json_lines(raw: str) -> list[Any]:
    rows: list[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _safe_json_loads(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def _read_transcript_file(path: Path) -> list[Any]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    parsed = _safe_json_loads(raw)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("transcript", "messages", "chat"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]

    return _parse_json_lines(raw)


def load_transcript(path_str: str = "") -> list[Any]:
    candidates: list[str] = []
    if path_str:
        candidates.append(path_str)
    candidates.append(OPENCLAW_FALLBACK_PATH)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        loaded = _read_transcript_file(Path(candidate))
        if loaded:
            return loaded
    return []
