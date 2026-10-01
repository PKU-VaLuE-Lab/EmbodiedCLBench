from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from .loader import load_tongsim_task


def _task_json_exists(task_dir: Path) -> bool:
    return any((task_dir / name).exists() for name in ("task.json", "graph.json"))


def _edge_key(edge: Any) -> tuple[str, str, str]:
    if isinstance(edge, dict):
        return (
            str(edge.get("from", "")),
            str(edge.get("to", "")),
            str(edge.get("action", "")),
        )
    return ("", "", "")


def _path_to_actions(path: Any) -> list[str]:
    if not isinstance(path, list):
        return []
    actions: list[str] = []
    for item in path:
        if isinstance(item, str):
            actions.append(item)
        elif isinstance(item, dict) and "action" in item:
            actions.append(str(item["action"]))
    return actions


def _reachable_node_ids(task: Any) -> set[str]:
    start = task.initial_node_id()
    seen = {start}
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        for edge in task.outgoing_edges(node_id):
            if edge.to_node not in seen:
                seen.add(edge.to_node)
                queue.append(edge.to_node)
    return seen


def _trace_maps_to_valid_path(task: Any, actions: list[str]) -> bool:
    if not actions:
        return False
    current = task.initial_node_id()
    for action in actions:
        matches = [edge for edge in task.outgoing_edges(current) if edge.action == action]
        if not matches:
            return False
        current = matches[0].to_node
    return True


def validate_task(task_dir: Path) -> list[str]:
    task_dir = Path(task_dir)
    messages: list[str] = []

    if not _task_json_exists(task_dir):
        return [f"ERROR: No task.json or graph.json found in {task_dir}"]

    json_path = next((task_dir / name for name in ("task.json", "graph.json") if (task_dir / name).exists()), None)
    try:
        json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise
    except Exception as exc:
        return [f"ERROR: Failed to read task JSON: {exc}"]

    try:
        task = load_tongsim_task(task_dir)
    except Exception as exc:
        return [f"ERROR: Failed to load task: {exc}"]

    node_ids = [node.id for node in task.nodes]
    node_id_counts = Counter(node_ids)
    duplicate_node_ids = sorted(node_id for node_id, count in node_id_counts.items() if count > 1)
    if duplicate_node_ids:
        messages.append(f"ERROR: Duplicate node ids found: {duplicate_node_ids}")

    node_id_set = set(node_ids)
    edge_keys = {(edge.from_node, edge.to_node, edge.action) for edge in task.dag_edges}

    for edge in task.dag_edges:
        if edge.from_node not in node_id_set:
            messages.append(f"ERROR: dag_edge.from references missing node: {edge.from_node}")
        if edge.to_node not in node_id_set:
            messages.append(f"ERROR: dag_edge.to references missing node: {edge.to_node}")

    for goal_node in task.goal_nodes:
        if goal_node not in node_id_set:
            messages.append(f"ERROR: goal_node references missing node: {goal_node}")

    try:
        initial_node_id = task.initial_node_id()
        if initial_node_id not in node_id_set:
            messages.append("ERROR: No valid initial node found")
    except Exception:
        messages.append("ERROR: No valid initial node found")

    if not task.goal_nodes and not any(node.is_goal for node in task.nodes) and not task.goal_state:
        messages.append("ERROR: No goal node or goal_state found")

    for path in task.goal_paths:
        if isinstance(path, list) and path and isinstance(path[0], dict):
            for edge in path:
                if _edge_key(edge) not in edge_keys:
                    messages.append(f"ERROR: goal_path contains invalid edge: {edge}")

    try:
        reachable = _reachable_node_ids(task)
    except Exception:
        reachable = set()
    for node_id in sorted(reachable):
        if not task.is_goal_node(node_id) and not task.outgoing_edges(node_id):
            messages.append(f"ERROR: Reachable non-goal node has no outgoing edge: {node_id}")

    for trace in task.success_traces:
        actions = _path_to_actions(trace)
        if actions and not _trace_maps_to_valid_path(task, actions):
            messages.append(
                f"WARNING: success_trace could not be exactly mapped in DAG: {actions}"
            )

    observations_dir = task_dir / "observations"
    if observations_dir.exists():
        for node in task.nodes:
            if not task.observations.get(node.id) or not task.observations[node.id].image_paths:
                messages.append(f"WARNING: Missing observation image for node {node.id}")

    for node in task.nodes:
        outgoing_actions = [edge.action for edge in task.outgoing_edges(node.id)]
        duplicates = sorted(action for action, count in Counter(outgoing_actions).items() if count > 1)
        if duplicates:
            messages.append(
                f"ERROR: Duplicate outgoing action_id from node {node.id}: {duplicates}"
            )

    try:
        task.shortest_success_trace_length()
    except Exception as exc:
        messages.append(f"ERROR: shortest success trace length cannot be computed: {exc}")

    try:
        task.max_goal_depth()
    except Exception as exc:
        messages.append(f"ERROR: max goal depth cannot be computed: {exc}")

    if not messages:
        messages.append("OK: task is valid")
    return messages
