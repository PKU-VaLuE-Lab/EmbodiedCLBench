from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import AgentDecision, VLMBackend


class MockBackend(VLMBackend):
    name = "mock"

    def __init__(self, scripted_actions: list[object] | None = None) -> None:
        self.scripted_actions = scripted_actions or []
        self._index = 0

    def choose_action(
        self,
        observation: dict[str, Any],
        prompt: str,
        image_paths: list[Path],
    ) -> AgentDecision:
        candidate_actions = observation.get("candidate_actions", [])
        selected_action_id: str | None = None
        action_level: str | None = None
        template_id: str | None = None
        action_type: str | None = None
        object_id: str | None = None
        target_id: str | None = None
        role_bindings: dict[str, str] = {}
        rejected_actions: list[str] = []
        rejected_options: list[str] = []
        if self._index < len(self.scripted_actions):
            scripted = self.scripted_actions[self._index]
            if isinstance(scripted, dict):
                action_level = str(scripted.get("action_level")) if scripted.get("action_level") is not None else None
                template_id = str(scripted.get("template_id")) if scripted.get("template_id") is not None else None
                action_type = str(scripted.get("action_type")) if scripted.get("action_type") is not None else None
                object_id = str(scripted.get("object_id")) if scripted.get("object_id") is not None else None
                target_id = str(scripted.get("target_id")) if scripted.get("target_id") is not None else None
                if isinstance(scripted.get("role_bindings"), dict):
                    role_bindings = {
                        str(key): str(value)
                        for key, value in scripted.get("role_bindings", {}).items()
                        if value is not None
                    }
            elif scripted is not None:
                selected_action_id = str(scripted)
        elif observation.get("action_interface") == "library_factorized":
            candidates = observation.get("model_facing_action_candidates", [])
            object_candidates = observation.get("model_facing_object_candidates", [])
            if candidates:
                first = candidates[0]
                action_level = str(first.get("action_level")) if first.get("action_level") is not None else None
                template_id = str(first.get("template_id")) if first.get("template_id") is not None else None
                action_type = str(first.get("action_type")) if first.get("action_type") is not None else None
                first_object_id = str(object_candidates[0]["object_id"]) if object_candidates else None
                role_bindings = {}
                for role in first.get("required_roles", []):
                    if first_object_id is not None:
                        role_bindings[str(role)] = first_object_id
                rejected_options = [json.dumps(candidate, ensure_ascii=False) for candidate in candidates[1:]]
            else:
                action_level = None
                template_id = None
                role_bindings = {}
        elif observation.get("action_interface") in {"factorized", "global_factorized"}:
            bindings = observation.get("valid_bindings_internal", [])
            if bindings:
                first = bindings[0]
                action_type = str(first.get("action_type")) if first.get("action_type") is not None else None
                object_id = str(first.get("object_id")) if first.get("object_id") is not None else None
                target_id = str(first.get("target_id")) if first.get("target_id") is not None else None
                rejected_options = [
                    json.dumps(binding, ensure_ascii=False)
                    for binding in bindings[1:]
                ]
            else:
                action_types = observation.get("action_types", [])
                object_choices = observation.get("object_choices", [])
                if action_types:
                    first_type = action_types[0]
                    action_type = str(first_type.get("action_type")) if first_type.get("action_type") is not None else None
                    required_args = list(first_type.get("required_args", []))
                    first_object_id = str(object_choices[0]["object_id"]) if object_choices else None
                    if "object_id" in required_args:
                        object_id = first_object_id
                    if "target_id" in required_args:
                        target_id = first_object_id
        elif candidate_actions:
            selected_action_id = str(candidate_actions[0]["action_id"])
            rejected_actions = [
                str(item["action_id"]) for item in candidate_actions if str(item["action_id"]) != selected_action_id
            ]
        self._index += 1
        payload = {
            "selected_action_id": selected_action_id,
            "action_level": action_level,
            "template_id": template_id,
            "action_type": action_type,
            "object_id": object_id,
            "target_id": target_id,
            "role_bindings": role_bindings,
            "confidence": 1.0 if (selected_action_id or action_type) else 0.0,
            "brief_reason": "Mock backend selected a deterministic action.",
            "perceived_state": "Using provided observation without model inference.",
            "rejected_actions": rejected_actions,
            "rejected_options": rejected_options,
        }
        return AgentDecision(
            selected_action_id=selected_action_id,
            action_level=payload.get("action_level"),
            template_id=payload.get("template_id"),
            action_type=action_type,
            object_id=object_id,
            target_id=target_id,
            role_bindings=payload.get("role_bindings", {}),
            confidence=payload["confidence"],
            brief_reason=payload["brief_reason"],
            perceived_state=payload["perceived_state"],
            rejected_actions=rejected_actions,
            rejected_options=rejected_options,
            raw_response=json.dumps(payload),
            parsed_json=payload,
            parse_error=None,
        )
