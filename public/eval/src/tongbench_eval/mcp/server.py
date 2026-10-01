from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from tongbench_eval.mcp.context_compact import SharedMcpCompactor
except ModuleNotFoundError:
    # Native Codex/Claude/OpenClaw runners execute this copied server script
    # with a standalone harness Python, not the repository package environment.
    from context_compact import SharedMcpCompactor


SESSION_CONFIG_PATH: Path | None = None
_CONTEXT_COMPACTOR: SharedMcpCompactor | None = None


def _session_config_path() -> Path:
    if SESSION_CONFIG_PATH is not None:
        return SESSION_CONFIG_PATH
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / ".tongsim_session.json", Path.cwd() / ".tongsim_session.json"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Missing .tongsim_session.json next to tongsim_mcp_server.py or in the current directory.")


def _load_session_config() -> dict[str, Any]:
    return json.loads(_session_config_path().read_text(encoding="utf-8"))


def _request_paths(config: dict[str, Any], request_id: str) -> tuple[Path, Path]:
    bridge_mount = Path(str(config.get("bridge_mount", "/tongsim_bridge")))
    return bridge_mount / "requests" / f"{request_id}.json", bridge_mount / "responses" / f"{request_id}.json"


def _request_paths_for_bridge_mount(bridge_mount: str, request_id: str) -> tuple[Path, Path]:
    bridge_root = Path(str(bridge_mount or "/tongsim_bridge"))
    return bridge_root / "requests" / f"{request_id}.json", bridge_root / "responses" / f"{request_id}.json"


def _send_request(command: str, payload: dict[str, Any] | None = None, timeout_sec: float = 120.0) -> dict[str, Any]:
    config = _load_session_config()
    request_id = uuid.uuid4().hex
    request_path, response_path = _request_paths(config, request_id)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps({"command": command, "payload": payload or {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            if not response.get("ok", False):
                error = response.get("error", "Unknown TongSIM bridge error")
                traceback_text = response.get("traceback", "")
                raise RuntimeError(f"{error}\n{traceback_text}".strip())
            return dict(response.get("result", {}))
        time.sleep(0.1)
    raise TimeoutError(f"TongSIM bridge timed out waiting for response to {command}.")


def _send_sequence_request(
    step: dict[str, Any],
    command: str,
    payload: dict[str, Any] | None = None,
    timeout_sec: float = 120.0,
) -> dict[str, Any]:
    request_id = uuid.uuid4().hex
    request_path, response_path = _request_paths_for_bridge_mount(str(step.get("bridge_mount", "")), request_id)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps({"command": command, "payload": payload or {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if response_path.exists():
            response = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            if not response.get("ok", False):
                error = response.get("error", "Unknown task bridge error")
                traceback_text = response.get("traceback", "")
                raise RuntimeError(f"{error}\n{traceback_text}".strip())
            return dict(response.get("result", {}))
        time.sleep(0.1)
    raise TimeoutError(f"Task bridge timed out waiting for response to {command}.")


def _resolve_local_workspace_root() -> Path:
    return _session_config_path().parent


def _mirror_observation_images(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _redact_observation_action_types(dict(payload))
    config = _load_session_config()
    local_root = _resolve_local_workspace_root()
    observation_subdir = str(config.get("observation_subdir", "current_observation"))
    local_observation_dir = local_root / observation_subdir
    if local_observation_dir.exists():
        shutil.rmtree(local_observation_dir)
    local_observation_dir.mkdir(parents=True, exist_ok=True)

    rewritten: list[str] = []
    for raw_path in payload.get("image_paths", []) or []:
        source = Path(str(raw_path))
        if not source.exists():
            continue
        destination = local_observation_dir / source.name
        shutil.copy2(source, destination)
        rewritten.append(str(destination.resolve()))
    payload["image_paths"] = rewritten
    (local_root / "last_observation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def _content_response_with_images(payload: dict[str, Any]) -> Any:
    payload = _compact_for_model(payload)
    image_paths = _response_image_paths(payload)
    visible_payload = _model_visible_payload(payload)
    if _text_only_mcp_mode() or not image_paths:
        return visible_payload

    text_payload = json.dumps(visible_payload, indent=2, ensure_ascii=False)
    content_blocks: list[Any] = [_text_content(text_payload)]
    for image_path in image_paths:
        image_content = _image_content(Path(image_path))
        if image_content is not None:
            content_blocks.append(image_content)
    return content_blocks if len(content_blocks) > 1 else visible_payload


def _response_image_paths(payload: dict[str, Any]) -> list[str]:
    image_paths = [str(path) for path in payload.get("image_paths", []) or [] if str(path).strip()]
    next_observation = payload.get("next_observation")
    if isinstance(next_observation, dict):
        image_paths.extend(str(path) for path in next_observation.get("image_paths", []) or [] if str(path).strip())
    return image_paths


def _text_only_mcp_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_TEXT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _no_image_input_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_NO_IMAGE_INPUT", "").strip().lower() in {"1", "true", "yes", "on"}


def _expose_image_paths_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_EXPOSE_IMAGE_PATHS", "").strip().lower() in {"1", "true", "yes", "on"}


def _image_content_only_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_IMAGE_CONTENT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _anthropic_image_blocks_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_ANTHROPIC_IMAGE_BLOCKS", "").strip().lower() in {"1", "true", "yes", "on"}


def _compress_images_mode() -> bool:
    return os.environ.get("TONGSIM_MCP_COMPRESS_IMAGES", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _model_visible_payload(value: Any) -> Any:
    """Hide filesystem image paths from the model while preserving attached images."""
    if isinstance(value, dict):
        visible: dict[str, Any] = {}
        for key, item in value.items():
            if key == "image_paths" and not _expose_image_paths_mode():
                continue
            if key in {"reached_goal", "confidence"}:
                continue
            if key == "task_instruction":
                continue
            if key == "action_candidates" and isinstance(item, list):
                visible[key] = _visible_action_templates(item)
                continue
            if key == "object_candidates" and isinstance(item, list):
                visible[key] = _visible_object_ids(item)
                continue
            if key == "observation_text" and isinstance(item, str):
                visible[key] = _scrub_observation_text_image_paths(item)
            else:
                visible[key] = _model_visible_payload(item)
        return visible
    if isinstance(value, list):
        return [_model_visible_payload(item) for item in value]
    return value


def _compact_for_model(value: Any) -> Any:
    if _CONTEXT_COMPACTOR is None or not isinstance(value, dict):
        return value
    return _CONTEXT_COMPACTOR.transform(value)


def _visible_payload(value: dict[str, Any]) -> Any:
    return _model_visible_payload(_compact_for_model(value))


def _visible_action_templates(items: list[Any]) -> list[str]:
    templates: list[str] = []
    for item in items:
        if isinstance(item, dict):
            template_id = str(item.get("template_id", "") or "").strip()
            if template_id:
                templates.append(template_id)
        elif str(item).strip():
            templates.append(str(item).strip())
    return templates


def _visible_object_ids(items: list[Any]) -> list[str]:
    object_ids: list[str] = []
    for item in items:
        if isinstance(item, dict):
            object_id = str(item.get("object_id", "") or "").strip()
            if object_id:
                object_ids.append(object_id)
        elif str(item).strip():
            object_ids.append(str(item).strip())
    return object_ids


def _scrub_observation_text_image_paths(text: str) -> str:
    lines: list[str] = []
    attached_line_added = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _no_image_input_mode() and line == "Current image: attached.":
            lines.append("Current image: none.")
            continue
        if line.startswith("- Image "):
            continue
        if line.startswith("Images available:") and "none" not in line.lower():
            if _text_only_mcp_mode():
                continue
            if not attached_line_added:
                lines.append("Current observation image: attached.")
                attached_line_added = True
            continue
        lines.append(raw_line)
    return "\n".join(lines)


def _text_content(text: str) -> Any:
    if _anthropic_image_blocks_mode():
        return {"type": "text", "text": text}
    try:
        from mcp.types import TextContent

        return TextContent(type="text", text=text)
    except Exception:
        return {"type": "text", "text": text}


def _image_mime_type(image_path: Path, image_bytes: bytes | None = None) -> str:
    if image_bytes is not None:
        header = image_bytes[:32]
    else:
        try:
            with image_path.open("rb") as handle:
                header = handle.read(32)
        except OSError:
            header = b""

    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"

    guessed = mimetypes.guess_type(str(image_path))[0]
    return guessed if guessed and guessed.startswith("image/") else "image/jpeg"


def _image_format_from_mime(mime_type: str) -> str:
    subtype = mime_type.split("/", 1)[-1].lower()
    return "jpeg" if subtype in {"jpg", "jpeg"} else subtype


def _base64_char_count(image_bytes: bytes) -> int:
    return ((len(image_bytes) + 2) // 3) * 4


def _image_bytes_for_model(image_path: Path) -> tuple[bytes, str]:
    image_bytes = image_path.read_bytes()
    mime_type = _image_mime_type(image_path, image_bytes)
    if not _compress_images_mode():
        return image_bytes, mime_type

    max_base64_chars = _env_int("TONGSIM_MCP_IMAGE_MAX_BASE64_CHARS", 70000, minimum=4096)
    if _base64_char_count(image_bytes) <= max_base64_chars:
        return image_bytes, mime_type

    try:
        from PIL import Image as PILImage
    except Exception:
        return image_bytes, mime_type

    max_side = _env_int("TONGSIM_MCP_IMAGE_MAX_SIDE", 768, minimum=64, maximum=4096)
    quality = _env_int("TONGSIM_MCP_IMAGE_JPEG_QUALITY", 70, minimum=10, maximum=95)
    side_candidates: list[int] = []
    for side in (max_side, 1024, 896, 768, 640, 512, 384):
        if side <= max_side and side not in side_candidates:
            side_candidates.append(side)
    if not side_candidates:
        side_candidates.append(max_side)
    quality_candidates: list[int] = []
    for candidate_quality in (quality, 80, 75, 70, 65, 60, 55, 50):
        if candidate_quality <= quality and candidate_quality not in quality_candidates:
            quality_candidates.append(candidate_quality)
    if not quality_candidates:
        quality_candidates.append(quality)

    try:
        with PILImage.open(io.BytesIO(image_bytes)) as opened:
            image = opened.convert("RGB")
            resampling = getattr(PILImage, "Resampling", None)
            resample_filter = getattr(resampling, "LANCZOS", getattr(PILImage, "LANCZOS", PILImage.BICUBIC))
            best_bytes: bytes | None = None
            for side in side_candidates:
                resized = image.copy()
                width, height = resized.size
                longest_side = max(width, height)
                if longest_side > side:
                    scale = side / float(longest_side)
                    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                    resized = resized.resize(new_size, resample_filter)
                for candidate_quality in quality_candidates:
                    buffer = io.BytesIO()
                    resized.save(buffer, format="JPEG", quality=candidate_quality, optimize=True)
                    candidate_bytes = buffer.getvalue()
                    if best_bytes is None or len(candidate_bytes) < len(best_bytes):
                        best_bytes = candidate_bytes
                    if _base64_char_count(candidate_bytes) <= max_base64_chars:
                        return candidate_bytes, "image/jpeg"
            if best_bytes is not None and len(best_bytes) < len(image_bytes):
                return best_bytes, "image/jpeg"
    except Exception:
        return image_bytes, mime_type

    return image_bytes, mime_type


def _image_content(image_path: Path) -> Any | None:
    if not image_path.exists() or not image_path.is_file():
        return None
    image_bytes = (
        _image_bytes_for_model(image_path)[0]
        if (_image_content_only_mode() or _anthropic_image_blocks_mode())
        else None
    )
    mime_type = _image_mime_type(image_path, image_bytes)
    if _anthropic_image_blocks_mode():
        encoded = base64.b64encode(image_bytes or b"").decode("ascii")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encoded,
            },
        }
    if _image_content_only_mode():
        encoded = base64.b64encode(image_bytes or b"").decode("ascii")
        try:
            from mcp.types import ImageContent

            return ImageContent(type="image", data=encoded, mimeType=mime_type)
        except Exception:
            return {"type": "image", "data": encoded, "mimeType": mime_type}

    try:
        from mcp.server.fastmcp import Image

        image_format = _image_format_from_mime(mime_type)
        try:
            return Image(path=str(image_path), format=image_format)
        except TypeError:
            return Image(path=str(image_path))
    except Exception:
        pass

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    try:
        from mcp.types import ImageContent

        return ImageContent(type="image", data=encoded, mimeType=mime_type)
    except Exception:
        return {"type": "image", "data": encoded, "mimeType": mime_type}


def _redact_observation_action_types(payload: dict[str, Any]) -> dict[str, Any]:
    action_candidates = payload.get("action_candidates")
    if isinstance(action_candidates, list):
        redacted_candidates: list[Any] = []
        for candidate in action_candidates:
            if isinstance(candidate, dict):
                redacted = dict(candidate)
                redacted.pop("action_type", None)
                redacted_candidates.append(redacted)
            else:
                redacted_candidates.append(candidate)
        payload["action_candidates"] = redacted_candidates
    return payload


def _build_decision_payload(
    *,
    template_id: str,
    role_bindings: dict[str, str],
    action_type: str = "",
    brief_reason: str = "",
    perceived_state_predicates: list[str] | None = None,
    uncertain_predicates: list[str] | None = None,
    visual_evidence: Any = None,
) -> dict[str, Any]:
    normalized_action = _normalize_selected_action_fields(
        template_id=template_id,
        action_type=action_type,
        role_bindings=role_bindings,
    )
    return {
        "perceived_state_predicates": list(perceived_state_predicates or []),
        "uncertain_predicates": list(uncertain_predicates or []),
        "visual_evidence": _normalize_visual_evidence(visual_evidence),
        "selected_action": {
            "action_level": "atomic",
            "template_id": normalized_action["template_id"],
            "action_type": normalized_action["action_type"],
            "role_bindings": normalized_action["role_bindings"],
        },
        "brief_reason": brief_reason,
    }


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _normalize_role_bindings_value(value: Any) -> dict[str, str]:
    def normalize_key(raw_key: Any) -> str:
        key = str(raw_key).strip()
        while key.startswith("{") and key.endswith("}") and len(key) > 2:
            inner = key[1:-1].strip()
            if not inner or inner == key:
                break
            key = inner
        return key

    def add_role(roles: dict[str, str], raw_key: Any, raw_value: Any) -> None:
        key = normalize_key(raw_key)
        value_text = str(raw_value).strip()
        if not key or not value_text:
            return
        raw_key_text = str(raw_key).strip()
        if key in roles and raw_key_text != key:
            return
        roles[key] = value_text

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            parsed_roles: dict[str, str] = {}
            chunks = [chunk.strip() for chunk in stripped.replace(";", ",").split(",")]
            for chunk in chunks:
                if "=" not in chunk:
                    continue
                key, raw_value = chunk.split("=", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if key and raw_value:
                    add_role(parsed_roles, key, raw_value)
            return parsed_roles
    if not isinstance(value, dict):
        return {}
    normalized_roles: dict[str, str] = {}
    for key, raw_value in value.items():
        add_role(normalized_roles, key, raw_value)
    return normalized_roles


def _normalize_selected_action_fields(
    *,
    template_id: str,
    action_type: str,
    role_bindings: Any,
) -> dict[str, Any]:
    normalized_template = str(template_id or "").strip()
    normalized_action_type = str(action_type or "").strip()
    normalized_roles = _normalize_role_bindings_value(role_bindings)

    if "{" in normalized_template and "}" in normalized_template:
        return {
            "template_id": normalized_template,
            "action_type": _normalize_action_type_for_template(normalized_template, normalized_action_type),
            "role_bindings": normalized_roles,
        }

    place_match = _normalize_instantiated_place_template(normalized_template, normalized_roles)
    if place_match is not None:
        return {
            "template_id": place_match["template_id"],
            "action_type": _normalize_action_type_for_template(place_match["template_id"], normalized_action_type),
            "role_bindings": place_match["role_bindings"],
        }

    single_match = _normalize_instantiated_single_object_template(
        normalized_template,
        normalized_action_type,
        normalized_roles,
    )
    if single_match is not None:
        return {
            "template_id": single_match["template_id"],
            "action_type": _normalize_action_type_for_template(
                single_match["template_id"],
                normalized_action_type or single_match["action_type"],
            ),
            "role_bindings": single_match["role_bindings"],
        }

    return {
        "template_id": normalized_template,
        "action_type": normalized_action_type,
        "role_bindings": normalized_roles,
    }


def _canonical_action_type_for_template(template_id: str) -> str | None:
    return {
        "look_at_{object}": "look_at",
        "point_at_{object}": "point_at",
        "walk_to_{object}": "walk_to",
        "pick_up_{object}": "pick_up",
        "place_{object}": "place",
        "place_{object1}_on_{object2}": "place",
        "place_{object1}_in_{object2}": "place",
        "drop_{object}": "drop",
        "open_{object}": "open",
        "close_{object}": "close",
        "switch_on_{object}": "switch_on",
        "switch_off_{object}": "switch_off",
        "pour_{object}": "pour",
        "hang_{object}": "hang_object",
        "hand_over_{object}": "hand_over",
        "cut_{object}": "cut",
        "plug_in_{object}": "plug_in",
        "unplug_{object}": "unplug",
        "wash_{object}": "wash_object",
    }.get(template_id)


def _normalize_action_type_for_template(template_id: str, action_type: str) -> str:
    normalized_action_type = str(action_type or "").strip()
    canonical_action_type = _canonical_action_type_for_template(template_id)
    if canonical_action_type is None:
        return normalized_action_type
    return canonical_action_type


def _normalize_instantiated_place_template(
    template_id: str,
    role_bindings: dict[str, str],
) -> dict[str, Any] | None:
    if not template_id.startswith("place_"):
        return None
    remainder = template_id[len("place_") :]
    for relation in ("_on_", "_in_"):
        if relation not in remainder:
            continue
        object1, object2 = remainder.split(relation, 1)
        if not object1 or not object2:
            continue
        roles = dict(role_bindings)
        roles.setdefault("object1", object1)
        roles.setdefault("object2", object2)
        return {
            "template_id": f"place_{{object1}}{relation}{{object2}}",
            "role_bindings": roles,
        }
    return None


def _normalize_instantiated_single_object_template(
    template_id: str,
    action_type: str,
    role_bindings: dict[str, str],
) -> dict[str, Any] | None:
    patterns = [
        ("look_at_", "look_at_{object}", "look_at"),
        ("point_at_", "point_at_{object}", "point_at"),
        ("walk_to_", "walk_to_{object}", "walk_to"),
        ("pick_up_", "pick_up_{object}", "pick_up"),
        ("drop_", "drop_{object}", "drop"),
        ("open_", "open_{object}", "open"),
        ("close_", "close_{object}", "close"),
        ("switch_on_", "switch_on_{object}", "switch_on"),
        ("switch_off_", "switch_off_{object}", "switch_off"),
        ("pour_", "pour_{object}", "pour"),
        ("hang_", "hang_{object}", "hang_object"),
        ("hand_over_", "hand_over_{object}", "hand_over"),
        ("cut_", "cut_{object}", "cut"),
        ("plug_in_", "plug_in_{object}", "plug_in"),
        ("unplug_", "unplug_{object}", "unplug"),
        ("wash_", "wash_{object}", "wash_object"),
    ]
    for prefix, canonical_template, canonical_action_type in patterns:
        if not template_id.startswith(prefix):
            continue
        object_id = template_id[len(prefix) :].strip()
        roles = dict(role_bindings)
        if object_id:
            roles.setdefault("object", object_id)
        return {
            "template_id": canonical_template,
            "action_type": action_type or canonical_action_type,
            "role_bindings": roles,
        }
    return None


def _normalize_visual_evidence(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"evidence": value}]
    if isinstance(value, dict):
        return [{str(key): str(item) for key, item in value.items()}]
    if isinstance(value, list):
        normalized: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                normalized.append({str(key): str(raw_value) for key, raw_value in item.items()})
            else:
                normalized.append({"evidence": str(item)})
        return normalized
    return [{"evidence": str(value)}]


def _sequence_steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    steps = config.get("sequence_steps", [])
    return [item for item in steps if isinstance(item, dict)]


def _sequence_state_path(config: dict[str, Any]) -> Path:
    raw_path = str(config.get("sequence_state_path", "") or "").strip()
    if raw_path:
        return Path(raw_path)
    return _session_config_path().parent / ".choice_sequence_state.json"


def _read_sequence_state(config: dict[str, Any]) -> dict[str, Any]:
    path = _sequence_state_path(config)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return {"current_index": 0}


def _write_sequence_state(config: dict[str, Any], state: dict[str, Any]) -> None:
    path = _sequence_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _current_sequence_step(config: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
    steps = _sequence_steps(config)
    state = _read_sequence_state(config)
    index = max(0, min(int(state.get("current_index", 0)), len(steps)))
    if index >= len(steps):
        return index, None
    return index, steps[index]


def _advance_sequence(config: dict[str, Any]) -> dict[str, Any]:
    steps = _sequence_steps(config)
    state = _read_sequence_state(config)
    current = max(0, min(int(state.get("current_index", 0)), len(steps)))
    next_index = min(current + 1, len(steps))
    updated = {"current_index": next_index}
    _write_sequence_state(config, updated)
    return updated


def _sequence_finished_payload(config: dict[str, Any]) -> dict[str, Any]:
    _ = config
    return {
        "all_tasks_complete": True,
        "message": "All tasks are complete. Provide a short final summary and stop using task tools.",
    }


def _advance_sequence_if_done(config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    next_observation = result.get("next_observation")
    is_done = bool(result.get("done", False)) or (
        isinstance(next_observation, dict) and bool(next_observation.get("done", False))
    )
    if not is_done:
        return result
    state = _advance_sequence(config)
    all_complete = int(state.get("current_index", 0)) >= len(_sequence_steps(config))
    return {
        **result,
        "task_complete": True,
        "all_tasks_complete": all_complete,
        "message": (
            "All tasks are complete. Stop using task tools."
            if all_complete
            else "The current task is complete. Call observe to begin the next task."
        ),
    }


def _build_sequence_server(FastMCP: Any, server_config: dict[str, Any]) -> Any:
    mcp = FastMCP("household_task")

    @mcp.tool(
        name="observe",
        description=(
            "Observe the current household task. The result includes the current goal, an attached image, "
            "and the available lettered choices. Call one task tool at a time and wait for the result."
        ),
    )
    def observe() -> Any:
        _, step = _current_sequence_step(server_config)
        if step is None:
            return _sequence_finished_payload(server_config)
        result = _advance_sequence_if_done(
            server_config,
            _send_sequence_request(step, "observe"),
        )
        return _content_response_with_images(_mirror_observation_images(result))

    @mcp.tool(
        name="select_option",
        description=(
            "Select one listed option for the current household task. Use only the option letter shown by observe."
        ),
    )
    def select_option(option_id: str, brief_reason: str = "") -> Any:
        _, step = _current_sequence_step(server_config)
        if step is None:
            return _sequence_finished_payload(server_config)
        result = _send_sequence_request(
            step,
            "act",
            {
                "option_id": option_id,
                "brief_reason": brief_reason,
            },
        )
        result = _advance_sequence_if_done(server_config, result)
        if isinstance(result.get("next_observation"), dict):
            result["next_observation"] = _mirror_observation_images(result["next_observation"])
        (_resolve_local_workspace_root() / "last_action_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if _response_image_paths(result):
            return _content_response_with_images(result)
        return _visible_payload(result)

    @mcp.tool(
        name="status",
        description="Check whether the current household task is complete. Call one task tool at a time.",
    )
    def status() -> dict[str, Any]:
        _, step = _current_sequence_step(server_config)
        if step is None:
            return _sequence_finished_payload(server_config)
        result = _send_sequence_request(step, "status")
        return _visible_payload(_advance_sequence_if_done(server_config, {
            "done": bool(result.get("done", False)),
            "step_count": int(result.get("step_count", 0)),
        }))

    return mcp


def _build_single_choice_server(FastMCP: Any, server_config: dict[str, Any]) -> Any:
    _ = server_config
    mcp = FastMCP("household_task")

    @mcp.tool(
        name="observe",
        description=(
            "Observe the current household task. The result includes the current goal, an attached image, "
            "and the available lettered choices. Call one task tool at a time and wait for the result."
        ),
    )
    def observe() -> Any:
        return _content_response_with_images(_mirror_observation_images(_send_request("observe")))

    @mcp.tool(
        name="select_option",
        description=(
            "Select one listed option for the current household task. Use only the option letter shown by observe."
        ),
    )
    def select_option(option_id: str, brief_reason: str = "") -> Any:
        result = _send_request(
            "act",
            {
                "option_id": option_id,
                "brief_reason": brief_reason,
            },
        )
        if isinstance(result.get("next_observation"), dict):
            result["next_observation"] = _mirror_observation_images(result["next_observation"])
        (_resolve_local_workspace_root() / "last_action_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if _response_image_paths(result):
            return _content_response_with_images(result)
        return _visible_payload(result)

    @mcp.tool(
        name="status",
        description="Check whether the current household task is complete. Call one task tool at a time.",
    )
    def status() -> dict[str, Any]:
        result = _send_request("status")
        return {
            "done": bool(result.get("done", False)),
            "step_count": int(result.get("step_count", 0)),
        }

    return mcp


def _build_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The Python package 'mcp' is required to run TongSIM MCP server. "
            "Run this script with the Hermes virtualenv Python."
        ) from exc

    server_config = _load_session_config()
    if _sequence_steps(server_config):
        return _build_sequence_server(FastMCP, server_config)
    if str(server_config.get("protocol_surface") or "legacy") == "safe_choice":
        return _build_single_choice_server(FastMCP, server_config)

    mcp = FastMCP("tongsim")
    decision_only = bool(server_config.get("decision_only", False))

    @mcp.tool(
        name="observe_current_state",
        description=(
            "Get the current TongSIM observation. The result includes the task, "
            "available action templates, available objects, feedback, done, and the current image "
            "attached in the same tool result. Call at most one TongSIM tool per assistant response, "
            "then wait for the tool result."
        ),
    )
    def observe_current_state() -> Any:
        return _content_response_with_images(_mirror_observation_images(_send_request("observe")))

    if decision_only:
        @mcp.tool(
            name="choose_action",
            description=(
                "Submit one TongSIM action decision. Use an exposed template_id and exact "
                "role_bindings from the latest observation. The action type is inferred automatically. "
                "Call at most one TongSIM tool per assistant response, then wait for the tool result."
            ),
        )
        def choose_action(
            template_id: str,
            role_bindings: dict[str, str],
        ) -> Any:
            normalized_action = _normalize_selected_action_fields(
                template_id=template_id,
                action_type="",
                role_bindings=role_bindings,
            )
            payload = {
                "selected_action": {
                    "action_level": "atomic",
                    "template_id": normalized_action["template_id"],
                    "action_type": normalized_action["action_type"],
                    "role_bindings": normalized_action["role_bindings"],
                }
            }
            result = _send_request("act", payload)
            if isinstance(result.get("next_observation"), dict):
                result["next_observation"] = _mirror_observation_images(result["next_observation"])
            (_resolve_local_workspace_root() / "last_action_result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if _response_image_paths(result):
                return _content_response_with_images(result)
            return _visible_payload(result)
    else:
        @mcp.tool(
            name="choose_action",
            description=(
                "Submit one TongSIM action decision. Use an exposed template_id and exact "
                "role_bindings from the latest observation. The action type is inferred automatically. "
                "Call at most one TongSIM tool per assistant response, then wait for the tool result."
            ),
        )
        def choose_action(
            template_id: str,
            role_bindings: dict[str, str],
            action_type: str = "",
            brief_reason: str = "",
            perceived_state_predicates: Any = None,
            uncertain_predicates: Any = None,
            visual_evidence: Any = None,
        ) -> Any:
            payload = _build_decision_payload(
                template_id=template_id,
                action_type=action_type,
                role_bindings=role_bindings,
                brief_reason=brief_reason,
                perceived_state_predicates=_normalize_string_list(perceived_state_predicates),
                uncertain_predicates=_normalize_string_list(uncertain_predicates),
                visual_evidence=visual_evidence,
            )
            result = _send_request("act", payload)
            if isinstance(result.get("next_observation"), dict):
                result["next_observation"] = _mirror_observation_images(result["next_observation"])
            (_resolve_local_workspace_root() / "last_action_result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if _response_image_paths(result):
                return _content_response_with_images(result)
            return _visible_payload(result)

    @mcp.tool(
        name="check_task_status",
        description="Check whether the current TongSIM episode is done and read the step count. Call at most one TongSIM tool per assistant response, then wait for the tool result.",
    )
    def check_task_status() -> dict[str, Any]:
        result = _send_request("status")
        return {
            "done": bool(result.get("done", False)),
            "step_count": int(result.get("step_count", 0)),
        }

    return mcp


def main() -> int:
    parser = argparse.ArgumentParser(description="TongSIM MCP server for Hermes semantic tool interaction.")
    parser.add_argument("--session-config", help="Path to .tongsim_session.json inside the agent container.")
    args = parser.parse_args()

    global SESSION_CONFIG_PATH
    if args.session_config:
        SESSION_CONFIG_PATH = Path(args.session_config)

    try:
        global _CONTEXT_COMPACTOR
        context_mode = os.environ.get("TONGSIM_MCP_CONTEXT_MODE", "full").strip()
        if context_mode not in {"full", "task_compact"}:
            raise ValueError(f"Unsupported MCP context mode: {context_mode!r}")
        if context_mode == "task_compact":
            _CONTEXT_COMPACTOR = SharedMcpCompactor(
                _session_config_path().parent / "context_compaction_audit.json"
            )
        server = _build_server()
        server.run(transport="stdio")
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
