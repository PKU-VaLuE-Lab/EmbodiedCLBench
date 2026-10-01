from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

from ..base import AgentDecision, VLMBackend
from ..parser import parse_model_decision


def _encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _extract_text_from_choice(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return json.dumps(content)


class OpenAICompatibleBackend(VLMBackend):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for openai-compatible backend")
        if not self.model:
            raise RuntimeError("Missing model name for openai-compatible backend")

    def choose_action(
        self,
        observation: dict[str, Any],
        prompt: str,
        image_paths: list[Path],
    ) -> AgentDecision:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "requests is required for the openai-compatible backend. Install requirements.txt or requirements-api-vlm.txt."
            ) from exc
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(image_path)}})
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        raw_response = _extract_text_from_choice(payload["choices"][0]["message"]["content"])
        return parse_model_decision(
            raw_response=raw_response,
            action_interface=observation.get("action_interface", "full_action"),
            candidate_action_ids=[item["action_id"] for item in observation.get("candidate_actions", [])],
            allowed_action_types=[item["action_type"] for item in observation.get("action_types", [])],
            object_choice_ids=[item["object_id"] for item in observation.get("object_choices", [])],
            action_type_requirements={
                item["action_type"]: list(item.get("required_args", []))
                for item in observation.get("action_types", [])
            },
            action_candidates=observation.get("model_facing_action_candidates", []),
        )
