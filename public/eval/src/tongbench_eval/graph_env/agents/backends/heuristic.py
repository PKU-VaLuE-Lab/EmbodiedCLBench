from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..base import AgentDecision, VLMBackend
from ...action_factorizer import parse_edge_action


class HeuristicBackend(VLMBackend):
    name = "heuristic"

    def choose_action(
        self,
        observation: dict[str, Any],
        prompt: str,
        image_paths: list[Path],
    ) -> AgentDecision:
        if observation.get("action_interface") == "library_factorized":
            candidates = observation.get("model_facing_action_candidates", [])
            objects = observation.get("model_facing_object_candidates", [])
            chosen = candidates[0] if candidates else None
            first_object = str(objects[0]["object_id"]) if objects else None
            role_bindings = {}
            if chosen is not None:
                for role in chosen.get("required_roles", []):
                    if first_object is not None:
                        role_bindings[str(role)] = first_object
            payload = {
                "selected_action_id": None,
                "action_level": chosen.get("action_level") if chosen else None,
                "template_id": chosen.get("template_id") if chosen else None,
                "action_type": chosen.get("action_type") if chosen else None,
                "role_bindings": role_bindings,
                "confidence": 0.5 if chosen else 0.0,
                "brief_reason": "Heuristic backend chose the first exposed library action candidate.",
                "perceived_state": "Using task predicates and template metadata as a non-model baseline.",
                "rejected_options": [json.dumps(candidate, ensure_ascii=False) for candidate in candidates[1:]],
            }
            return AgentDecision(
                selected_action_id=None,
                action_level=payload["action_level"],
                template_id=payload["template_id"],
                action_type=payload["action_type"],
                role_bindings=role_bindings,
                confidence=payload["confidence"],
                brief_reason=payload["brief_reason"],
                perceived_state=payload["perceived_state"],
                rejected_options=payload["rejected_options"],
                raw_response=json.dumps(payload),
                parsed_json=payload,
                parse_error=None,
            )
        if observation.get("action_interface") in {"factorized", "global_factorized"}:
            bindings = observation.get("valid_bindings_internal", [])
            current_depth = int(observation.get("current_depth", 0))
            success_traces = observation.get("success_traces", [])
            chosen_binding: dict[str, Any] | None = None
            for trace in success_traces:
                if not isinstance(trace, list) or current_depth >= len(trace):
                    continue
                preferred = parse_edge_action(str(trace[current_depth]))
                for binding in bindings:
                    if (
                        binding.get("action_type") == preferred.get("action_type")
                        and binding.get("object_id") == preferred.get("object_id")
                        and binding.get("target_id") == preferred.get("target_id")
                    ):
                        chosen_binding = binding
                        break
                if chosen_binding is not None:
                    break
            if chosen_binding is None and not bindings:
                for trace in success_traces:
                    if not isinstance(trace, list) or current_depth >= len(trace):
                        continue
                    preferred = parse_edge_action(str(trace[current_depth]))
                    chosen_binding = {
                        "action_type": preferred.get("action_type"),
                        "object_id": preferred.get("object_id"),
                        "target_id": preferred.get("target_id"),
                    }
                    break
            if chosen_binding is None and bindings:
                chosen_binding = bindings[0]
            payload = {
                "selected_action_id": None,
                "action_type": chosen_binding.get("action_type") if chosen_binding else None,
                "object_id": chosen_binding.get("object_id") if chosen_binding else None,
                "target_id": chosen_binding.get("target_id") if chosen_binding else None,
                "confidence": 0.6 if chosen_binding else 0.0,
                "brief_reason": "Heuristic backend matched a factorized success-trace prefix or fell back to the first valid binding.",
                "perceived_state": "Using state predicates and action prefixes as a non-model baseline.",
                "rejected_options": [
                    json.dumps(binding, ensure_ascii=False)
                    for binding in bindings
                    if binding is not chosen_binding
                ],
            }
            return AgentDecision(
                selected_action_id=None,
                action_type=payload["action_type"],
                object_id=payload["object_id"],
                target_id=payload["target_id"],
                confidence=payload["confidence"],
                brief_reason=payload["brief_reason"],
                perceived_state=payload["perceived_state"],
                rejected_options=payload["rejected_options"],
                raw_response=json.dumps(payload),
                parsed_json=payload,
                parse_error=None,
            )

        candidate_actions = observation.get("candidate_actions", [])
        selected_action_id: str | None = None
        current_depth = int(observation.get("current_depth", 0))
        success_traces = observation.get("success_traces", [])
        for trace in success_traces:
            if not isinstance(trace, list) or current_depth >= len(trace):
                continue
            preferred = str(trace[current_depth])
            for candidate in candidate_actions:
                if str(candidate["action_id"]) == preferred:
                    selected_action_id = preferred
                    break
            if selected_action_id is not None:
                break
        if selected_action_id is None and candidate_actions:
            selected_action_id = str(candidate_actions[0]["action_id"])

        payload = {
            "selected_action_id": selected_action_id,
            "confidence": 0.6 if selected_action_id else 0.0,
            "brief_reason": "Heuristic backend matched a success-trace prefix or fell back to the first candidate.",
            "perceived_state": "Using state predicates and action prefixes as a non-model baseline.",
            "rejected_actions": [
                str(item["action_id"]) for item in candidate_actions if str(item["action_id"]) != selected_action_id
            ],
        }
        return AgentDecision(
            selected_action_id=selected_action_id,
            confidence=payload["confidence"],
            brief_reason=payload["brief_reason"],
            perceived_state=payload["perceived_state"],
            rejected_actions=payload["rejected_actions"],
            raw_response=json.dumps(payload),
            parsed_json=payload,
            parse_error=None,
        )
