from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _path_to_data_url(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_chat_image_urls(chat_path: Path) -> list[str]:
    image_urls: list[str] = []
    for row in _read_jsonl(chat_path):
        if row.get("type") != "toolResult":
            continue
        content = (row.get("toolResult") or {}).get("content")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        for block in parsed:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_url = (block.get("image_url") or {}).get("url")
            if isinstance(image_url, str) and image_url.startswith("data:image/"):
                image_urls.append(image_url)
    return image_urls


def _image_source_for_trace_entry(
    entry: dict[str, Any],
    chat_image_urls: list[str],
) -> tuple[str | None, str | None]:
    for raw_path in entry.get("observation_image_paths", []) or []:
        data_url = _path_to_data_url(Path(str(raw_path)))
        if data_url:
            return data_url, str(raw_path)
    if chat_image_urls:
        # Fallback for semantic-tool runs whose trace contains container/Linux
        # paths that are not readable from the host after export.
        return chat_image_urls[0], "chat.jsonl:first_attached_image"
    return None, None


def _image_cache_key(data_url: str) -> str:
    return hashlib.sha256(data_url.encode("utf-8")).hexdigest()


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _describe_image_with_model(
    *,
    data_url: str,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
) -> str:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for sidecar decision diagnostics") from exc

    prompt = (
        "Describe this TongSIM observation image for offline debugging. "
        "Do not choose or recommend actions. Mention visible objects, spatial "
        "relations, whether common task targets appear visible, and uncertainty. "
        "Keep it concise and factual."
    )
    response = requests.post(
        f"{_normalize_sidecar_base_url(base_url).rstrip('/')}/chat/completions",
        headers={
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return _extract_message_text(payload["choices"][0]["message"]["content"]).strip()


def _normalize_sidecar_base_url(base_url: str) -> str:
    override = os.environ.get("TONGSIM_DIAGNOSTICS_BASE_URL", "").strip()
    if override:
        return override
    parsed = urlsplit(base_url)
    if parsed.hostname != "host.docker.internal":
        return base_url
    netloc = "127.0.0.1"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _symbolic_diagnostic_summary(entry: dict[str, Any]) -> str:
    action = entry.get("selected_template_id") or entry.get("selected_action_id") or "<unknown>"
    role_bindings = entry.get("selected_role_bindings") or {}
    target_text = ", ".join(f"{key}={value}" for key, value in role_bindings.items())
    if entry.get("valid"):
        to_state = entry.get("to_state") or entry.get("next_state_id")
        return f"Action {action}({target_text}) was accepted and moved to {to_state}."
    missing = [str(item) for item in entry.get("missing_preconditions", []) if str(item).strip()]
    if missing:
        return f"Action {action}({target_text}) was rejected because missing preconditions: {', '.join(missing)}."
    diagnostic = entry.get("diagnostic") if isinstance(entry.get("diagnostic"), dict) else {}
    reason = str(diagnostic.get("reason", "") or entry.get("execution_status", "") or "unknown reason")
    return f"Action {action}({target_text}) was rejected: {reason}."


def write_sidecar_decision_diagnostics(
    *,
    output_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int = 256,
) -> Path:
    """Write offline diagnostics without feeding them back to the acting model."""

    output_dir = Path(output_dir)
    task_output_dir = output_dir / "task_output"
    trace_entries = _read_jsonl(task_output_dir / "trace.jsonl")
    chat_image_urls = _extract_chat_image_urls(output_dir / "chat.jsonl")
    diagnostics_path = task_output_dir / "sidecar_decision_diagnostics.jsonl"
    diagnostics_path.unlink(missing_ok=True)

    image_description_cache: dict[str, str] = {}
    image_error_cache: dict[str, str] = {}

    for entry in trace_entries:
        data_url, image_source = _image_source_for_trace_entry(entry, chat_image_urls)
        image_description = None
        image_error = None
        if data_url and base_url and model:
            key = _image_cache_key(data_url)
            if key not in image_description_cache and key not in image_error_cache:
                try:
                    image_description_cache[key] = _describe_image_with_model(
                        data_url=data_url,
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        max_tokens=max_tokens,
                    )
                except Exception as exc:
                    image_error_cache[key] = f"{type(exc).__name__}: {exc}"
            image_description = image_description_cache.get(key)
            image_error = image_error_cache.get(key)
        elif data_url:
            image_error = "Missing model/base_url for image description."
        else:
            image_error = "No readable observation image found."

        _append_jsonl(
            diagnostics_path,
            {
                "step_index": entry.get("step_index"),
                "current_state_id": entry.get("current_state_id"),
                "render_node_id": entry.get("render_node_id"),
                "selected_action": {
                    "template_id": entry.get("selected_template_id"),
                    "action_type": entry.get("selected_action_type"),
                    "role_bindings": entry.get("selected_role_bindings") or {},
                },
                "valid": bool(entry.get("valid", False)),
                "execution_status": entry.get("execution_status"),
                "missing_preconditions": list(entry.get("missing_preconditions", []) or []),
                "symbolic_diagnostic_summary": _symbolic_diagnostic_summary(entry),
                "image_source": image_source,
                "image_description": image_description,
                "image_description_error": image_error,
            },
        )

    return diagnostics_path
