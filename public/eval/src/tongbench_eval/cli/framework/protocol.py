from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


LEGACY_MCP_TOOL_NAMES = (
    "observe_current_state",
    "choose_action",
    "check_task_status",
)

SAFE_CHOICE_MCP_TOOL_NAMES = (
    "observe",
    "select_option",
    "status",
)


def protocol_tool_names(protocol_surface: str) -> tuple[str, ...]:
    if protocol_surface == "safe_choice":
        return SAFE_CHOICE_MCP_TOOL_NAMES
    return LEGACY_MCP_TOOL_NAMES


def mcp_image_env(
    *,
    backend: str,
    no_image_input: bool,
    direct_image_input: bool,
    compress_images: bool,
    context_mode: str = "full",
    image_max_side: int | None = None,
    image_max_base64_chars: int | None = None,
    image_jpeg_quality: int | None = None,
) -> dict[str, str]:
    # Explicitly disable compression unless the CLI opt-in is present. This
    # prevents a stale parent environment variable from changing the protocol.
    env: dict[str, str] = {
        "TONGSIM_MCP_COMPRESS_IMAGES": "0",
        "TONGSIM_MCP_CONTEXT_MODE": str(context_mode),
    }
    canonical_image_content = os.environ.get(
        "TONGBENCH_MCP_CANONICAL_IMAGE_CONTENT", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if no_image_input:
        env["TONGSIM_MCP_TEXT_ONLY"] = "1"
        env["TONGSIM_MCP_NO_IMAGE_INPUT"] = "1"
    elif canonical_image_content:
        # Responses-based compatibility routes require the protocol-standard
        # MCP image block. Keep this opt-in so existing providers retain their
        # established transport representation.
        env["TONGSIM_MCP_IMAGE_CONTENT_ONLY"] = "1"
        env["TONGSIM_MCP_PI_IMAGE_CONTENT"] = "1"
    elif backend == "codex":
        env["TONGSIM_MCP_EXPOSE_IMAGE_PATHS"] = "1"
        if not direct_image_input:
            env["TONGSIM_MCP_TEXT_ONLY"] = "1"
    elif backend == "openclaw":
        if direct_image_input:
            env["TONGSIM_MCP_IMAGE_CONTENT_ONLY"] = "1"
        else:
            env["TONGSIM_MCP_TEXT_ONLY"] = "1"
    elif backend == "claudecode":
        env["TONGSIM_MCP_ANTHROPIC_IMAGE_BLOCKS"] = "1"

    if compress_images:
        env["TONGSIM_MCP_COMPRESS_IMAGES"] = "1"
        if image_max_side is not None:
            env["TONGSIM_MCP_IMAGE_MAX_SIDE"] = str(int(image_max_side))
        if image_max_base64_chars is not None:
            env["TONGSIM_MCP_IMAGE_MAX_BASE64_CHARS"] = str(int(image_max_base64_chars))
        if image_jpeg_quality is not None:
            env["TONGSIM_MCP_IMAGE_JPEG_QUALITY"] = str(int(image_jpeg_quality))
    return env


def codex_mcp_servers(
    mcp_servers: dict[str, Any],
    *,
    tool_names: Iterable[str],
) -> dict[str, Any]:
    enabled_tools = [str(name) for name in tool_names]
    return {
        name: {
            **server,
            "env": {
                **(server.get("env", {}) if isinstance(server.get("env"), dict) else {}),
                "TONGSIM_MCP_EXPOSE_IMAGE_PATHS": "1",
            },
            "required": True,
            "default_tools_approval_mode": "approve",
            "enabled_tools": enabled_tools,
        }
        for name, server in mcp_servers.items()
    }


def claudecode_allowed_tools(
    mcp_servers: dict[str, Any],
    *,
    tool_names: Iterable[str],
) -> list[str]:
    return [
        f"mcp__{_safe_name(server_name)}__{tool_name}"
        for server_name in mcp_servers
        for tool_name in tool_names
    ]


def build_protocol_manifest(
    *,
    task_id: str,
    phase: str,
    self_evolution_mode: str,
    protocol_surface: str,
    prompt: str,
    post_task_prompt: str | None,
    choice_count: int,
    choice_seed: int,
    choice_bank_path: str | None,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_steps = [
        {
            "index": int(step["index"]),
            "task_id": str(step["task_id"]),
            "level": str(step["level"]),
            "max_steps": int(step["max_steps"]),
        }
        for step in steps
    ]
    payload = {
        "protocol_version": 2,
        "task_id": task_id,
        "phase": phase,
        "self_evolution_mode": self_evolution_mode,
        "protocol_surface": protocol_surface,
        "tool_names": list(protocol_tool_names(protocol_surface)),
        "prompt_sha256": _sha256_text(prompt),
        "post_task_prompt_sha256": _sha256_text(post_task_prompt or ""),
        "choice_count": int(choice_count),
        "choice_seed": int(choice_seed),
        "choice_bank_sha256": _sha256_file(Path(choice_bank_path)) if choice_bank_path else "",
        "steps": normalized_steps,
    }
    # The application prompt may contain a skill generated from this harness's
    # own learning trajectory. Keep its hash for provenance, but do not treat
    # model-generated text as a protocol configuration difference.
    protocol_payload = {key: value for key, value in payload.items() if key != "prompt_sha256"}
    canonical = json.dumps(protocol_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {**payload, "protocol_fingerprint": _sha256_text(canonical)}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
