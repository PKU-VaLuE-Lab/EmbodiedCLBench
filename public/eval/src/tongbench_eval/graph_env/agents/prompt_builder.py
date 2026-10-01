from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _sanitize_prompt_objects(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for item in items:
        object_id = item.get("object_id")
        if object_id is None:
            continue
        sanitized.append({"object_id": str(object_id)})
    return sanitized


def build_action_selection_prompt(
    observation: dict[str, Any],
    image_paths: list[Path],
    action_interface: str = "full_action",
    include_state_predicates: bool = True,
    include_goal_state: bool = True,
) -> str:
    prompt_payload = {
        "task_instruction": observation.get("task_instruction", ""),
        "current_state_id": observation.get("current_state_id", ""),
        "current_depth": observation.get("current_depth", 0),
        "is_virtual_state": observation.get("is_virtual_state", False),
        "image_paths": [str(path) for path in image_paths],
        "observation_status": observation.get("observation_status"),
        "observation_warning": observation.get("observation_warning"),
    }
    if include_state_predicates:
        prompt_payload["state_predicates"] = observation.get("state_predicates", [])
    if include_goal_state:
        prompt_payload["goal_state"] = observation.get("goal_state", [])
    if action_interface == "library_factorized":
        prompt_payload["action_interface"] = "library_factorized"
        prompt_payload["action_library_mode"] = observation.get("action_library_mode", "atomic_only")
        prompt_payload["model_facing_action_candidates"] = observation.get("model_facing_action_candidates", [])
        prompt_payload["model_facing_object_candidates"] = _sanitize_prompt_objects(
            observation.get("model_facing_object_candidates", [])
        )
        prompt_payload["authoritative_object_ids"] = [
            str(item.get("object_id"))
            for item in observation.get("model_facing_object_candidates", [])
            if item.get("object_id") is not None
        ]
        available_levels = sorted(
            {
                str(candidate.get("action_level", "")).strip()
                for candidate in prompt_payload["model_facing_action_candidates"]
                if str(candidate.get("action_level", "")).strip()
            }
        )
        action_level_hint = " or ".join(available_levels) if available_levels else "atomic or subtask"
        previous_report = observation.get("previous_action_report")
        if previous_report is not None:
            prompt_payload["previous_action_report"] = previous_report
        return "\n".join(
            [
                "You are an embodied decision agent for an offline environment.",
                "Use the current image or images as the primary source of truth for the current state.",
                "First infer the current visual state, then choose one exposed action template and bind its roles to task objects.",
                (
                    "In this run, the exposed action candidates are atomic action templates."
                    if observation.get("action_library_mode") == "atomic_only"
                    else ""
                ),
                "Infer only visually supported predicates.",
                "Put ambiguous, weakly supported, or uncertain predicates into uncertain_predicates instead of perceived_state_predicates.",
                "If the image does not support a confident manipulation action, prefer an action that helps reduce uncertainty.",
                "Do not invent template ids, action levels, action types, or objects.",
                "Do not output raw DAG action ids.",
                "Every role_bindings value must exactly copy one provided object_id from model_facing_object_candidates.",
                "Do not use short_name, type_guess, aliases, or natural-language names such as fruit, plate, or cup.",
                "If the object list provides BP_Fruit_Apple_02_FB_C_2, you must output BP_Fruit_Apple_02_FB_C_2, not fruit.",
                "The object list is an identity reference only. It does not tell you which objects are currently visible, reachable, or held.",
                "Do not assume hidden predicates just because an action template would require them.",
                "Use template pre_state and the image evidence to decide whether an action is plausible.",
                "Not every exposed template is executable in the current state. The environment will check executability.",
                "If a previous action report is provided, use it as environment feedback to revise your next decision.",
                "Do not output hidden chain-of-thought.",
                "",
                "Return JSON with exactly this schema:",
                "{",
                '  "perceived_state_predicates": ["..."],',
                '  "uncertain_predicates": ["..."],',
                '  "visual_evidence": [{"predicate": "...", "evidence": "..."}],',
                '  "selected_action": {',
                f'    "action_level": "{action_level_hint}",',
                '    "template_id": "...",',
                '    "action_type": "...",',
                '    "role_bindings": {"object": "...", "object1": "...", "object2": "...", "r1": "...", "r2": "..."}',
                "  },",
                '  "brief_reason": "directly explain why this action template and object binding were selected",',
                '  "perceived_state": "short visual/state observation",',
                '  "rejected_options": ["..."]',
                "}",
                "",
                "Previous action report:" if previous_report is not None else "",
                json.dumps(previous_report, ensure_ascii=False, indent=2) if previous_report is not None else "",
                "",
                "Context:",
                json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            ]
        ).replace("\n\n\n", "\n\n")
    if action_interface in {"factorized", "global_factorized"}:
        prompt_payload["action_interface"] = action_interface
        prompt_payload["action_types"] = observation.get("action_types", [])
        prompt_payload["object_choices"] = _sanitize_prompt_objects(observation.get("object_choices", []))
        prompt_payload["authoritative_object_ids"] = [
            str(item.get("object_id"))
            for item in observation.get("object_choices", [])
            if item.get("object_id") is not None
        ]
        previous_report = (
            observation.get("previous_action_report")
            if action_interface == "global_factorized"
            else observation.get("previous_invalid_action_report")
        )
        if previous_report is not None:
            prompt_payload["previous_action_report" if action_interface == "global_factorized" else "previous_invalid_action_report"] = previous_report
        return "\n".join(
            [
                (
                    "You are selecting an action for an offline embodied environment."
                    if action_interface == "global_factorized"
                    else "You are a multimodal action-selection agent for a local offline embodied DAG environment."
                ),
                "Choose action_type only from action_types.",
                "Choose object_id and target_id only from object_choices when required.",
                "Every object_id and target_id must exactly copy one provided object_id from object_choices.",
                "Do not invent object ids or use aliases such as fruit, plate, cup, or table.",
                "Do not output any raw full action_id.",
                "If action_type requires object_id, object_id must not be null.",
                "If action_type requires target_id, target_id must not be null.",
                "For place, choose both the object being placed and the target surface.",
                (
                    "Not every global action is executable in the current state."
                    if action_interface == "global_factorized"
                    else "Use images when available."
                ),
                (
                    "The environment will validate whether your selected action is executable and whether it has an offline transition."
                    if action_interface == "global_factorized"
                    else "Use images when available."
                ),
                "Keep the reason brief and observable. Do not output hidden chain-of-thought.",
                (
                    "Your previous action was rejected by the environment. Use the diagnostic report to revise your next decision. "
                    "The report explains why the action was invalid, but it does not directly provide the correct answer."
                    if previous_report is not None and action_interface == "factorized"
                    else (
                        "If a previous action report is provided, use it as environment feedback to revise your next decision."
                        if previous_report is not None and action_interface == "global_factorized"
                        else ""
                    )
                ),
                "Use images when available.",
                (
                    "Do not assume current valid outgoing actions are shown to you."
                    if action_interface == "global_factorized"
                    else ""
                ),
                "",
                "Return JSON with exactly this schema:",
                "{",
                '  "action_type": "...",',
                '  "object_id": "... or null",',
                '  "target_id": "... or null",',
                '  "brief_reason": "one or two sentences",',
                '  "perceived_state": "short description of relevant visual/state facts",',
                '  "rejected_options": ["..."]',
                "}",
                "",
                ("Previous invalid action feedback:" if action_interface == "factorized" else "Previous action report:")
                if previous_report is not None
                else "",
                json.dumps(previous_report, ensure_ascii=False, indent=2) if previous_report is not None else "",
                "",
                "Context:",
                json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            ]
        ).replace("\n\n\n", "\n\n")

    prompt_payload["action_interface"] = "full_action"
    prompt_payload["candidate_actions"] = observation.get("candidate_actions", [])
    return "\n".join(
        [
            "You are a multimodal action-selection agent for a local offline embodied DAG environment.",
            "Choose exactly one action_id from candidate_actions.",
            "Do not invent any new action.",
            "If uncertain, choose the best candidate action that most likely advances toward the goal.",
            "Keep the reason brief and based only on observable or provided state information.",
            "Do not output hidden chain-of-thought.",
            "",
            "Return JSON with exactly this schema:",
            "{",
            '  "selected_action_id": "...",',
            '  "brief_reason": "one or two sentences, no hidden chain-of-thought",',
            '  "perceived_state": "short description of what matters visually",',
            '  "rejected_actions": ["..."]',
            "}",
            "",
            "Context:",
            json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        ]
    )
