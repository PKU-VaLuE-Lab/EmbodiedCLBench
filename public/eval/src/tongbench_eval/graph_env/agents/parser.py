from __future__ import annotations

import json
import re
from typing import Any

from .base import AgentDecision
from ..object_normalizer import infer_object_type_from_id, normalize_object_type


def _extract_json_string(raw_response: str) -> str:
    fenced_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL)
    if fenced_matches:
        return fenced_matches[0]

    decoder = json.JSONDecoder()
    for index, char in enumerate(raw_response):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(raw_response[index:])
            return raw_response[index : index + end]
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object found in model response")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _safe_visual_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        payload: dict[str, Any] = {}
        if item.get("predicate") is not None:
            payload["predicate"] = str(item.get("predicate"))
        if item.get("evidence") is not None:
            payload["evidence"] = str(item.get("evidence"))
        if item.get("confidence") is not None:
            payload["confidence"] = _safe_float(item.get("confidence"))
        if payload:
            evidence.append(payload)
    return evidence


def parse_model_decision(
    raw_response: str,
    action_interface: str = "full_action",
    candidate_action_ids: list[str] | None = None,
    allowed_action_types: list[str] | None = None,
    object_choice_ids: list[str] | None = None,
    action_type_requirements: dict[str, list[str]] | None = None,
    action_candidates: list[dict[str, Any]] | None = None,
) -> AgentDecision:
    candidate_action_ids = candidate_action_ids or []
    allowed_action_types = allowed_action_types or []
    object_choice_ids = object_choice_ids or []
    action_type_requirements = action_type_requirements or {}
    try:
        json_string = _extract_json_string(raw_response)
        parsed_json = json.loads(json_string)
    except Exception as exc:
        return AgentDecision(
            selected_action_id=None,
            rejected_actions=[],
            rejected_options=[],
            raw_response=raw_response,
            parsed_json=None,
            parse_error=str(exc),
            parse_error_details=None,
        )

    parse_error: str | None = None
    parse_error_details: dict[str, Any] | None = None
    selected_action_id = None
    action_level = None
    template_id = None
    action_type = None
    object_id = None
    target_id = None
    role_bindings: dict[str, str] = {}
    rejected_actions = parsed_json.get("rejected_actions", [])
    if not isinstance(rejected_actions, list):
        rejected_actions = []
    rejected_options = parsed_json.get("rejected_options", [])
    if not isinstance(rejected_options, list):
        rejected_options = []
    perceived_state_predicates = _safe_string_list(parsed_json.get("perceived_state_predicates"))
    uncertain_predicates = _safe_string_list(parsed_json.get("uncertain_predicates"))
    visual_evidence = _safe_visual_evidence(parsed_json.get("visual_evidence"))
    selected_action_payload = parsed_json.get("selected_action")
    if not isinstance(selected_action_payload, dict):
        selected_action_payload = parsed_json

    if action_interface == "library_factorized":
        if selected_action_payload.get("action_level") is not None:
            action_level = str(selected_action_payload.get("action_level"))
        if selected_action_payload.get("template_id") is not None:
            template_id = str(selected_action_payload.get("template_id"))
        if selected_action_payload.get("action_type") is not None:
            action_type = str(selected_action_payload.get("action_type"))
        raw_role_bindings = selected_action_payload.get("role_bindings", {})
        if isinstance(raw_role_bindings, dict):
            role_bindings = {}
            for raw_key, value in raw_role_bindings.items():
                if value is None:
                    continue
                key = str(raw_key).strip()
                while key.startswith("{") and key.endswith("}") and len(key) > 2:
                    inner = key[1:-1].strip()
                    if not inner or inner == key:
                        break
                    key = inner
                if key:
                    role_bindings[key] = str(value)
        chosen_candidate = None
        for candidate in action_candidates or []:
            if (
                str(candidate.get("action_level", "")) == str(action_level or "")
                and str(candidate.get("template_id", "")) == str(template_id or "")
            ):
                chosen_candidate = candidate
                break
        if chosen_candidate is None:
            parse_error = "The selected action template is not part of the exposed action candidates."
            parse_error_details = {
                "action_level": action_level,
                "template_id": template_id,
                "action_type": action_type,
            }
        elif str(chosen_candidate.get("action_type", "")) != str(action_type or ""):
            parse_error = "action_type does not match selected template candidate"
        else:
            required_roles = [str(item) for item in chosen_candidate.get("required_roles", [])]
            role_constraints = {
                str(key): [str(value) for value in values]
                for key, values in chosen_candidate.get("role_constraints", {}).items()
                if isinstance(values, list)
            }
            for role in required_roles:
                if not role_bindings.get(role):
                    parse_error = f"missing required role binding: {role}"
                    break
                if role_bindings[role] not in object_choice_ids:
                    parse_error = f"role binding object not in object choices: {role_bindings[role]}"
                    parse_error_details = {
                        "failed_role": role,
                        "selected_object_id": role_bindings[role],
                        "allowed_object_ids": list(object_choice_ids),
                        "message": "Role bindings must exactly match one provided object_id from model_facing_object_candidates.",
                    }
                    break
                allowed_types = role_constraints.get(role, [])
                canonical_allowed_types = sorted({normalize_object_type(value) for value in allowed_types})
                if allowed_types:
                    selected_object_id = role_bindings[role]
                    canonical_selected_type = infer_object_type_from_id(selected_object_id)
                    if canonical_selected_type not in canonical_allowed_types:
                        parse_error = "Role binding violates template object constraints."
                        parse_error_details = {
                            "failed_role": role,
                            "selected_object_id": selected_object_id,
                            "selected_object_type": canonical_selected_type,
                            "allowed_object_types": list(allowed_types),
                            "canonical_selected_type": canonical_selected_type,
                            "canonical_allowed_types": canonical_allowed_types,
                        }
                        break
    elif action_interface in {"factorized", "global_factorized"}:
        if selected_action_payload.get("action_type") is not None:
            action_type = str(selected_action_payload.get("action_type"))
        if selected_action_payload.get("object_id") is not None:
            object_id = str(selected_action_payload.get("object_id"))
        if selected_action_payload.get("target_id") is not None:
            target_id = str(selected_action_payload.get("target_id"))
        if not action_type:
            parse_error = "action_type missing from model output"
        elif action_type not in allowed_action_types:
            parse_error = f"action_type not in allowed action types: {action_type}"
            action_type = None
        else:
            required_args = action_type_requirements.get(action_type, [])
            if "object_id" in required_args and not object_id:
                parse_error = f"object_id required for action_type {action_type}"
            elif object_id and object_id not in object_choice_ids:
                parse_error = f"object_id not in object choices: {object_id}"
                parse_error_details = {
                    "selected_object_id": object_id,
                    "allowed_object_ids": list(object_choice_ids),
                    "message": "object_id must exactly match one provided object_id from object_choices.",
                }
                object_id = None
            elif "target_id" in required_args and not target_id:
                parse_error = f"target_id required for action_type {action_type}"
            elif target_id and target_id not in object_choice_ids:
                parse_error = f"target_id not in object choices: {target_id}"
                parse_error_details = {
                    "selected_target_id": target_id,
                    "allowed_object_ids": list(object_choice_ids),
                    "message": "target_id must exactly match one provided object_id from object_choices.",
                }
                target_id = None
    else:
        selected_action_id = selected_action_payload.get("selected_action_id")
        if selected_action_id is not None:
            selected_action_id = str(selected_action_id)
        if not selected_action_id:
            parse_error = "selected_action_id missing from model output"
            selected_action_id = None
        elif selected_action_id not in candidate_action_ids:
            parse_error = f"selected_action_id not in candidate actions: {selected_action_id}"
            selected_action_id = None

    return AgentDecision(
        selected_action_id=selected_action_id,
        action_level=action_level,
        template_id=template_id,
        action_type=action_type,
        object_id=object_id,
        target_id=target_id,
        role_bindings=role_bindings,
        confidence=_safe_float(parsed_json.get("confidence")),
        brief_reason=str(parsed_json.get("brief_reason")) if parsed_json.get("brief_reason") is not None else None,
        perceived_state=str(parsed_json.get("perceived_state")) if parsed_json.get("perceived_state") is not None else None,
        perceived_state_predicates=perceived_state_predicates,
        uncertain_predicates=uncertain_predicates,
        visual_evidence=visual_evidence,
        rejected_actions=[str(item) for item in rejected_actions],
        rejected_options=[str(item) for item in rejected_options],
        raw_response=raw_response,
        parsed_json=parsed_json,
        parse_error=parse_error,
        parse_error_details=parse_error_details,
    )
