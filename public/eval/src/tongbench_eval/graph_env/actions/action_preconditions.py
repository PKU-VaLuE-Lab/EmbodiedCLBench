from __future__ import annotations

import re
from typing import Any

from tongbench_eval.graph_env.io.object_normalizer import infer_object_type_from_id, normalize_object_type
from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate, normalize_predicate_set


def _extract_ids_from_predicate(predicate: str) -> list[str]:
    if "(" not in predicate or ")" not in predicate:
        return []
    inner = predicate.split("(", 1)[1].rsplit(")", 1)[0]
    return [item.strip() for item in inner.split(",") if item.strip()]


def _normalize_object_id_token(value: Any) -> str:
    return str(value).strip()


def _task_object_set(task_objects: list[str], current_predicates: list[str]) -> set[str]:
    object_ids = {
        _normalize_object_id_token(item)
        for item in task_objects
        if _normalize_object_id_token(item)
    }
    for predicate in normalize_predicate_set(current_predicates):
        for object_id in _extract_ids_from_predicate(predicate):
            normalized = _normalize_object_id_token(object_id)
            if normalized:
                object_ids.add(normalized)
    return object_ids


def _predicate_set(current_predicates: list[str]) -> set[str]:
    return set(normalize_predicate_set(current_predicates))


def _has_predicate(prefix: str, object_id: str | None, current_predicates: set[str]) -> bool:
    if not object_id:
        return False
    return f"{prefix}({object_id})" in current_predicates


def _normalize_action(action_type: str, object_id: str | None, target_id: str | None) -> tuple[str, str | None, str | None]:
    normalized_object_id = object_id
    normalized_target_id = target_id
    if action_type == "reach" and normalized_target_id is None and normalized_object_id is not None:
        normalized_target_id = normalized_object_id
        normalized_object_id = None
    return action_type, normalized_object_id, normalized_target_id


def check_action_executable(
    action_type: str,
    object_id: str | None,
    target_id: str | None,
    current_predicates: list[str],
    task_objects: list[str],
) -> dict[str, Any]:
    action_type, object_id, target_id = _normalize_action(action_type, object_id, target_id)
    predicate_set = _predicate_set(current_predicates)
    task_object_set = _task_object_set(task_objects, current_predicates)
    required_args_map = {
        "acquire": ["object_id"],
        "acquire_and_clean": ["object_id"],
        "reach": ["target_id"],
        "place": ["object_id", "target_id"],
    }
    required_args = required_args_map.get(action_type, [])
    violations: list[dict[str, str]] = []

    if "object_id" in required_args and not object_id:
        violations.append({"type": "missing_required_argument", "message": f"{action_type} requires object_id."})
    if "target_id" in required_args and not target_id:
        violations.append({"type": "missing_required_argument", "message": f"{action_type} requires target_id."})
    if violations:
        return {
            "executable_by_state": False,
            "precondition_violations": violations,
            "required_args": required_args,
            "normalized_action": {
                "action_type": action_type,
                "object_id": object_id,
                "target_id": target_id,
            },
        }

    if action_type == "acquire":
        assert object_id is not None
        object_id = _normalize_object_id_token(object_id)
        if object_id not in task_object_set:
            violations.append({"type": "object_not_in_task_scope", "message": "The object is not in task scope."})
        if not _has_predicate("exists", object_id, predicate_set):
            violations.append({"type": "object_does_not_exist", "message": "The object does not exist in the current state."})
        if "empty" not in predicate_set:
            violations.append({"type": "hand_not_empty", "message": "The hand is not empty."})
        if _has_predicate("holding", object_id, predicate_set):
            violations.append({"type": "object_already_held", "message": "The object is already being held."})
    elif action_type == "acquire_and_clean":
        assert object_id is not None
        object_id = _normalize_object_id_token(object_id)
        if object_id not in task_object_set:
            violations.append({"type": "object_not_in_task_scope", "message": "The object is not in task scope."})
        if not _has_predicate("exists", object_id, predicate_set):
            violations.append({"type": "object_does_not_exist", "message": "The object does not exist in the current state."})
        if "empty" not in predicate_set:
            violations.append({"type": "hand_not_empty", "message": "The hand is not empty."})
        if not _has_predicate("dirty", object_id, predicate_set):
            violations.append(
                {
                    "type": "object_not_dirty_or_not_cleanable",
                    "message": "The object is not dirty or cleaning is not meaningful right now.",
                }
            )
    elif action_type == "reach":
        assert target_id is not None
        target_id = _normalize_object_id_token(target_id)
        if target_id not in task_object_set:
            violations.append({"type": "target_not_in_task_scope", "message": "The target is not in task scope."})
        if not _has_predicate("exists", target_id, predicate_set):
            violations.append({"type": "target_does_not_exist", "message": "The target does not exist in the current state."})
        if _has_predicate("with_reach", target_id, predicate_set):
            violations.append({"type": "target_already_reached", "message": "The target is already within reach."})
    elif action_type == "place":
        assert object_id is not None and target_id is not None
        object_id = _normalize_object_id_token(object_id)
        target_id = _normalize_object_id_token(target_id)
        if object_id not in task_object_set:
            violations.append({"type": "object_not_in_task_scope", "message": "The object is not in task scope."})
        if target_id not in task_object_set:
            violations.append({"type": "target_not_in_task_scope", "message": "The target is not in task scope."})
        if not _has_predicate("holding", object_id, predicate_set):
            violations.append({"type": "not_holding_object", "message": "The object is not currently being held."})
        if not _has_predicate("exists", target_id, predicate_set):
            violations.append({"type": "target_does_not_exist", "message": "The target does not exist in the current state."})
        if not (_has_predicate("with_reach", target_id, predicate_set) or _has_predicate("in_view", target_id, predicate_set)):
            violations.append(
                {
                    "type": "target_not_reached_or_visible",
                    "message": "The target is neither reached nor visible in the current state.",
                }
            )
    else:
        violations.append({"type": "unknown_action_type", "message": "The action_type is not supported by the benchmark."})

    return {
        "executable_by_state": not violations,
        "precondition_violations": violations,
        "required_args": required_args,
        "normalized_action": {
            "action_type": action_type,
            "object_id": object_id,
            "target_id": target_id,
        },
    }


def _instantiate_value(value: str, role_bindings: dict[str, str]) -> str:
    instantiated = str(value)
    for role, object_id in role_bindings.items():
        instantiated = instantiated.replace(f"{{{role}}}", str(object_id))
        instantiated = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(role)}(?![A-Za-z0-9_])", str(object_id), instantiated)
    return instantiated


def check_template_executable(
    selected_action_candidate: dict[str, Any],
    role_bindings: dict[str, str],
    current_state_predicates: list[str],
    task_objects: list[str],
) -> dict[str, Any]:
    required_roles = [str(item) for item in selected_action_candidate.get("required_roles", [])]
    role_constraints = {
        str(key): [str(value) for value in values]
        for key, values in selected_action_candidate.get("role_constraints", {}).items()
        if isinstance(values, list)
    }
    current_state = {str(item) for item in current_state_predicates}
    current_state = set(normalize_predicate_set(list(current_state)))
    task_object_set = _task_object_set(task_objects, list(current_state))
    violations: list[dict[str, str]] = []
    instantiated_pre_state: list[str] = []
    satisfied_preconditions: list[str] = []
    missing_preconditions: list[str] = []

    for role in required_roles:
        if role not in role_bindings or not role_bindings[role]:
            violations.append(
                {
                    "type": "missing_role_binding",
                    "predicate": role,
                    "message": f"Missing required role binding for {role}.",
                }
            )
            continue
        object_id = _normalize_object_id_token(role_bindings[role])
        if object_id not in task_object_set:
            violations.append(
                {
                    "type": "object_not_in_task_scope",
                    "predicate": object_id,
                    "message": f"{object_id} is not in task scope.",
                }
            )
        allowed_types = role_constraints.get(role, [])
        canonical_selected_type = infer_object_type_from_id(object_id)
        canonical_allowed_types = {normalize_object_type(item) for item in allowed_types}
        if allowed_types and canonical_selected_type not in canonical_allowed_types:
            violations.append(
                {
                    "type": "object_type_not_allowed_by_template",
                    "predicate": object_id,
                    "message": f"{object_id} does not satisfy the template type constraint for role {role}.",
                }
            )

    for predicate in selected_action_candidate.get("pre_state", []):
        instantiated = normalize_predicate(_instantiate_value(str(predicate), role_bindings))
        instantiated_pre_state.append(instantiated)
        if instantiated in current_state:
            satisfied_preconditions.append(instantiated)
        else:
            missing_preconditions.append(instantiated)
            violations.append(
                {
                    "type": "missing_precondition",
                    "predicate": instantiated,
                    "message": f"Missing required precondition: {instantiated}.",
                }
            )

    return {
        "executable_by_state": not violations,
        "instantiated_pre_state": normalize_predicate_set(instantiated_pre_state),
        "satisfied_preconditions": normalize_predicate_set(satisfied_preconditions),
        "missing_preconditions": normalize_predicate_set(missing_preconditions),
        "precondition_violations": violations,
    }
