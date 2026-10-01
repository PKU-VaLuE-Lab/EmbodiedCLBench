from __future__ import annotations

import re
from typing import Any

from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate, normalize_predicate_set
from tongbench_eval.graph_env.runtime.state_consistency import apply_state_consistency_rules


def _instantiate_value(value: str, role_bindings: dict[str, str]) -> str:
    instantiated = str(value)
    for role, object_id in role_bindings.items():
        instantiated = instantiated.replace(f"{{{role}}}", str(object_id))
        instantiated = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(role)}(?![A-Za-z0-9_])", str(object_id), instantiated)
    return instantiated


def _extract_args(predicate: str) -> list[str]:
    if "(" not in predicate or ")" not in predicate:
        return []
    return [item.strip() for item in predicate.split("(", 1)[1].rsplit(")", 1)[0].split(",") if item.strip()]


def _derived_delete_rules(add_effects: list[str]) -> tuple[set[str], set[str]]:
    deletes: set[str] = set()
    adds: set[str] = set()
    for predicate in add_effects:
        predicate = normalize_predicate(predicate)
        args = _extract_args(predicate)
        if predicate.startswith("holding(") and args:
            deletes.add("empty")
        if predicate == "empty":
            deletes.add("holding(*)")
        if predicate.startswith("clean(") and args:
            deletes.add(f"dirty({args[0]})")
        if predicate.startswith("dirty(") and args:
            deletes.add(f"clean({args[0]})")
        if predicate.startswith("open(") and args:
            deletes.add(f"closed({args[0]})")
        if predicate.startswith("closed(") and args:
            deletes.add(f"open({args[0]})")
        if predicate.startswith("powered_on(") and args:
            deletes.add(f"powered_off({args[0]})")
        if predicate.startswith("powered_off(") and args:
            deletes.add(f"powered_on({args[0]})")
        if predicate.startswith("plugged_in(") and args:
            deletes.add(f"unplugged({args[0]})")
        if predicate.startswith("unplugged(") and args:
            deletes.add(f"plugged_in({args[0]})")
        if (predicate.startswith("on(") or predicate.startswith("in(")) and len(args) >= 1:
            deletes.add(f"holding({args[0]})")
            adds.add("empty")
    return deletes, adds


def _apply_deletes(state: set[str], delete_effects: list[str]) -> set[str]:
    next_state = set(state)
    for predicate in delete_effects:
        predicate = normalize_predicate(predicate)
        if predicate == "holding(*)":
            next_state = {item for item in next_state if not item.startswith("holding(")}
        else:
            next_state.discard(predicate)
    return next_state


def apply_template_effects(
    selected_action_candidate: dict[str, Any],
    role_bindings: dict[str, str],
    current_state_predicates: list[str],
) -> dict[str, Any]:
    previous_state = normalize_predicate_set([str(value) for value in current_state_predicates])
    current_state = set(previous_state)
    action_level = str(selected_action_candidate.get("action_level", "subtask"))
    raw_add_effects: list[str]
    raw_delete_effects: list[str]
    if action_level == "subtask":
        raw_add = selected_action_candidate.get("net_add") or selected_action_candidate.get("post_state") or []
        raw_delete = selected_action_candidate.get("net_delete") or []
        raw_add_effects = normalize_predicate_set([_instantiate_value(str(value), role_bindings) for value in raw_add])
        raw_delete_effects = normalize_predicate_set([_instantiate_value(str(value), role_bindings) for value in raw_delete])
        effect_source = "net_add_net_delete" if selected_action_candidate.get("net_add") or selected_action_candidate.get("net_delete") else "post_state_with_derived_deletes"
    else:
        raw_add = selected_action_candidate.get("post_state") or []
        raw_add_effects = normalize_predicate_set([_instantiate_value(str(value), role_bindings) for value in raw_add])
        derived_deletes, derived_adds = _derived_delete_rules(raw_add_effects)
        raw_delete_effects = normalize_predicate_set(sorted(derived_deletes))
        raw_add_effects = normalize_predicate_set(sorted(set(raw_add_effects).union({normalize_predicate(item) for item in derived_adds})))
        effect_source = "post_state_with_derived_deletes"
    consistency = apply_state_consistency_rules(
        previous_state_predicates=previous_state,
        add_effects=raw_add_effects,
        delete_effects=raw_delete_effects,
        selected_action_candidate=selected_action_candidate,
        role_bindings=role_bindings,
    )
    add_effects = normalize_predicate_set(consistency["final_add_effects"])
    delete_effects = normalize_predicate_set(consistency["final_delete_effects"])
    next_state = _apply_deletes(current_state, delete_effects)
    next_state.update(add_effects)
    new_state_predicates = normalize_predicate_set(sorted(next_state))
    return {
        "previous_state_predicates": previous_state,
        "raw_add_effects": list(raw_add_effects),
        "raw_delete_effects": list(raw_delete_effects),
        "add_effects": add_effects,
        "delete_effects": delete_effects,
        "consistency_adds": list(consistency["consistency_adds"]),
        "consistency_deletes": list(consistency["consistency_deletes"]),
        "consistency_rules_applied": list(consistency["consistency_rules_applied"]),
        "new_state_predicates": new_state_predicates,
        "normalized_new_state_predicates": list(new_state_predicates),
        "effect_source": effect_source,
    }
