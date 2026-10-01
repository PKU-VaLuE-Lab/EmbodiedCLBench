"""Codex-specific extraction of the post-task Skill response."""

from __future__ import annotations


def extract_skill_text(raw_output: str | None) -> str:
    """Return Codex assistant text without changing its Markdown content."""

    return str(raw_output or "").strip()
