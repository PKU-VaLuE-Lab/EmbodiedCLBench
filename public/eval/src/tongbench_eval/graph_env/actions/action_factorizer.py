from __future__ import annotations

import re
from typing import Any


GLOBAL_ACTION_SCHEMA: dict[str, dict[str, Any]] = {
    "acquire": {
        "action_type": "acquire",
        "required_args": ["object_id"],
        "description": "Acquire or pick up an object.",
    },
    "acquire_and_clean": {
        "action_type": "acquire_and_clean",
        "required_args": ["object_id"],
        "description": "Acquire and clean an object.",
    },
    "reach": {
        "action_type": "reach",
        "required_args": ["target_id"],
        "description": "Move or reach toward a target object or surface.",
    },
    "place": {
        "action_type": "place",
        "required_args": ["object_id", "target_id"],
        "description": "Place an object on a target.",
    },
}


def normalize_short_name(object_id: str) -> str:
    lowered = object_id.lower()
    if "cup" in lowered:
        return "cup"
    if "fruit" in lowered or "apple" in lowered:
        return "fruit"
    if "plate" in lowered:
        return "plate"
    if "remotecontroller" in lowered or "remote_controller" in lowered or "remote" in lowered:
        return "remote_control"
    if "coffeetable" in lowered or "coffee_table" in lowered or "table" in lowered:
        return "coffee_table"
    return "unknown"


def parse_edge_action(action_id: str) -> dict[str, Any]:
    action_id = str(action_id)
    parsed = {
        "raw_action_id": action_id,
        "action_type": "unknown",
        "object_id": None,
        "target_id": None,
        "required_args": [],
        "display_name": action_id,
    }

    patterns = [
        (r"^acquire_and_clean_\{object\}__(.+)$", "acquire_and_clean", "object_id", ["object_id"]),
        (r"^acquire_\{object\}__(.+)$", "acquire", "object_id", ["object_id"]),
        (r"^reach_\{object\}__(.+)$", "reach", "target_id", ["target_id"]),
    ]
    for pattern, action_type, key, required_args in patterns:
        match = re.match(pattern, action_id)
        if not match:
            continue
        parsed["action_type"] = action_type
        parsed[key] = match.group(1)
        parsed["required_args"] = required_args
        parsed["display_name"] = f"{action_type} {match.group(1)}"
        return parsed

    match = re.match(r"^place_(.+?)_on_(.+)$", action_id)
    if match:
        parsed["action_type"] = "place"
        parsed["object_id"] = match.group(1)
        parsed["target_id"] = match.group(2)
        parsed["required_args"] = ["object_id", "target_id"]
        parsed["display_name"] = f"place {match.group(1)} on {match.group(2)}"
        return parsed

    return parsed


def factorize_outgoing_edges(edges: list[dict[str, Any]]) -> dict[str, Any]:
    descriptions = {name: spec["description"] for name, spec in GLOBAL_ACTION_SCHEMA.items()}
    descriptions["unknown"] = "Unparsed action type"
    action_type_index: dict[str, dict[str, Any]] = {}
    valid_bindings_internal: list[dict[str, Any]] = []
    for edge in edges:
        parsed = parse_edge_action(str(edge.get("action", "")))
        action_type = parsed["action_type"]
        if action_type not in action_type_index:
            action_type_index[action_type] = {
                "action_type": action_type,
                "required_args": list(parsed["required_args"]),
                "description": descriptions.get(action_type, descriptions["unknown"]),
            }
        valid_bindings_internal.append(
            {
                "action_type": action_type,
                "object_id": parsed["object_id"],
                "target_id": parsed["target_id"],
                "raw_action_id": parsed["raw_action_id"],
                "to_state": str(edge.get("to", "")),
            }
        )
    return {
        "action_types": list(action_type_index.values()),
        "valid_bindings_internal": valid_bindings_internal,
    }


def build_global_action_types(task: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for edge in task.get("dag_edges", []):
        parsed = parse_edge_action(str(edge.get("action", "")))
        action_type = parsed["action_type"]
        if action_type in GLOBAL_ACTION_SCHEMA and action_type not in seen:
            ordered.append(dict(GLOBAL_ACTION_SCHEMA[action_type]))
            seen.add(action_type)
    for action_type in ("acquire", "acquire_and_clean", "reach", "place"):
        if action_type not in seen:
            ordered.append(dict(GLOBAL_ACTION_SCHEMA[action_type]))
    return ordered


def _extract_ids_from_predicate(predicate: str) -> list[str]:
    match = re.search(r"\((.*)\)", predicate)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(",") if item.strip()]


def build_object_choices(task: dict[str, Any], current_node_state: list[str]) -> list[dict[str, Any]]:
    object_ids: set[str] = set()
    task_scope = task.get("task_scope")
    if isinstance(task_scope, list):
        object_ids.update(str(item) for item in task_scope)
    for predicate in current_node_state:
        object_ids.update(_extract_ids_from_predicate(str(predicate)))
    for edge in task.get("dag_edges", []):
        parsed = parse_edge_action(str(edge.get("action", "")))
        if parsed["object_id"]:
            object_ids.add(str(parsed["object_id"]))
        if parsed["target_id"]:
            object_ids.add(str(parsed["target_id"]))

    properties_by_object: dict[str, set[str]] = {object_id: set() for object_id in object_ids}
    for predicate in current_node_state:
        predicate = str(predicate)
        if predicate.startswith("holding("):
            for object_id in _extract_ids_from_predicate(predicate):
                properties_by_object.setdefault(object_id, set()).add("holding")
        elif predicate.startswith("dirty("):
            for object_id in _extract_ids_from_predicate(predicate):
                properties_by_object.setdefault(object_id, set()).add("dirty")
        elif predicate.startswith("clean("):
            for object_id in _extract_ids_from_predicate(predicate):
                properties_by_object.setdefault(object_id, set()).add("clean")
        elif predicate.startswith("exists("):
            for object_id in _extract_ids_from_predicate(predicate):
                props = properties_by_object.setdefault(object_id, set())
                props.add("exists")
                props.add("in_view")
        elif predicate.startswith("on("):
            args = _extract_ids_from_predicate(predicate)
            if len(args) == 2:
                properties_by_object.setdefault(args[0], set()).add(f"on({args[1]})")
        elif predicate.startswith("with_reach("):
            for object_id in _extract_ids_from_predicate(predicate):
                properties_by_object.setdefault(object_id, set()).add("with_reach")

    output: list[dict[str, Any]] = []
    for object_id in sorted(object_ids):
        output.append(
            {
                "object_id": object_id,
                "short_name": normalize_short_name(object_id),
                "current_properties": sorted(properties_by_object.get(object_id, set())),
            }
        )
    return output
