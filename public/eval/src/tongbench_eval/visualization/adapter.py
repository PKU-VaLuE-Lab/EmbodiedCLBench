from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VisualizationInputs:
    graph_path: Path
    image_manifest_path: Path | None = None
    trace_path: Path | None = None
    success_paths_path: Path | None = None
    atomic_transition_manifest_path: Path | None = None
    unique_states_path: Path | None = None
    full_branch_graph_path: Path | None = None


def load_viz_graph(inputs: VisualizationInputs) -> dict[str, Any]:
    graph = _read_json(inputs.graph_path)
    image_manifest = _read_json(inputs.image_manifest_path) if inputs.image_manifest_path else {}
    trace_steps = _read_trace(inputs.trace_path) if inputs.trace_path else []
    success_paths = _read_json(inputs.success_paths_path) if inputs.success_paths_path else {}
    atomic_manifest = _read_json(inputs.atomic_transition_manifest_path) if inputs.atomic_transition_manifest_path else {}
    unique_states = _read_json(inputs.unique_states_path) if inputs.unique_states_path else {}
    full_branch_graph = _read_json(inputs.full_branch_graph_path) if inputs.full_branch_graph_path else {}
    full_nodes, full_edges = _extract_full_graph_parts(full_branch_graph)

    image_by_state_id = _image_manifest_by_state_id(image_manifest)
    image_by_node_id = _trace_images_by_node_id(trace_steps)
    trace_node_steps = _trace_steps_by_node_id(trace_steps)
    goal_predicates = set(graph.get("goal_state", []) or [])

    nodes: list[dict[str, Any]] = []
    for node in graph.get("nodes", []) or []:
        node_id = str(node.get("id"))
        state = list(node.get("state", []) or [])
        unique_state_id = str(node.get("unique_state_id", "") or "")
        image_path = image_by_state_id.get(unique_state_id) or image_by_node_id.get(node_id)
        is_goal = bool(node.get("is_goal_state")) or bool(goal_predicates and goal_predicates.issubset(set(state)))
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "unique_state_id": unique_state_id,
                "state": state,
                "state_key": node.get("state_key"),
                "is_initial": bool(node.get("is_initial_state")),
                "is_goal": is_goal,
                "image_path": image_path,
                "trace_steps": trace_node_steps.get(node_id, []),
                "source": "atomic_expanded_graph",
            }
        )

    raw_edges = list(graph.get("edges", []) or [])
    selected_trace_steps_by_edge_id = _select_trace_edges(raw_edges, trace_steps)
    trace_edge_ids = set(selected_trace_steps_by_edge_id)
    transition_by_key = _atomic_transition_by_key(atomic_manifest)
    edges: list[dict[str, Any]] = []
    for edge in raw_edges:
        edge_id = str(edge.get("id"))
        trace_steps_for_edge = selected_trace_steps_by_edge_id.get(edge_id, [])
        transition = transition_by_key.get(_transition_key(edge))
        edges.append(
            {
                "id": edge_id,
                "source": str(edge.get("from")),
                "target": str(edge.get("to")),
                "label": str(edge.get("atomic_action") or edge.get("action_type") or edge_id),
                "atomic_action": edge.get("atomic_action"),
                "template_id": edge.get("template_id"),
                "action_type": edge.get("action_type"),
                "parent_edge_id": edge.get("parent_edge_id"),
                "atomic_step_id": edge.get("atomic_step_id"),
                "parent_action": edge.get("parent_action"),
                "parent_source": edge.get("parent_source"),
                "branch_status": edge.get("branch_status"),
                "is_success_edge": bool(edge.get("is_success_edge")),
                "is_shortest_success_edge": bool((transition or {}).get("appears_in_shortest_success_paths")),
                "appears_in_success_paths": list((transition or {}).get("appears_in_success_paths", []) or []),
                "appears_in_shortest_success_paths": list((transition or {}).get("appears_in_shortest_success_paths", []) or []),
                "appears_in_longer_success_paths": list((transition or {}).get("appears_in_longer_success_paths", []) or []),
                "appears_in_full_edges": list((transition or {}).get("appears_in_full_edges", []) or []),
                "atomic_transition_id": (transition or {}).get("transition_id"),
                "is_trace_edge": bool(trace_steps_for_edge),
                "trace_steps": trace_steps_for_edge,
                "role_bindings": edge.get("role_bindings", {}),
                "before_state": edge.get("before_state", []),
                "after_state": edge.get("after_state", []),
            }
        )
    edge_groups = _aggregate_parallel_edges(edges)

    return {
        "schema_version": "tongbench_viz_graph_v1",
        "task": {
            "task_id": graph.get("task_id"),
            "description": graph.get("description"),
            "description_original": graph.get("description_original"),
            "description_highlevel": graph.get("description_highlevel"),
            "source_graph": graph.get("source_graph"),
            "expansion_mode": graph.get("expansion_mode"),
            "initial_state": graph.get("initial_state", []),
            "goal_state": graph.get("goal_state", []),
            "component_sources": graph.get("component_sources", []),
            "composition": graph.get("composition", {}),
        },
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edge_group_count": len(edge_groups),
            "parallel_edge_reduction": len(edges) - len(edge_groups),
            "goal_node_count": sum(1 for node in nodes if node["is_goal"]),
            "trace_step_count": len(trace_steps),
            "trace_edge_count": len(trace_edge_ids),
            "success_edge_count": sum(1 for edge in edges if edge["is_success_edge"]),
            "shortest_success_edge_count": sum(1 for edge in edge_groups if edge.get("is_shortest_success_edge")),
            "success_path_count": success_paths.get("success_path_count"),
            "total_success_path_count_in_full_graph": success_paths.get("total_success_path_count_in_full_graph"),
            "unique_atomic_transition_count": atomic_manifest.get("unique_atomic_transition_count"),
            "unique_state_count": unique_states.get("unique_state_count"),
            "full_branch_graph_available": bool(full_nodes),
            "full_branch_node_count": len(full_nodes),
            "full_branch_edge_count": len(full_edges),
            "full_branch_status_counts": _full_branch_status_counts(full_edges),
        },
        "nodes": nodes,
        "edges": edges,
        "edge_groups": edge_groups,
        "layers": _build_layers(
            graph=graph,
            nodes=nodes,
            edges=edges,
            edge_groups=edge_groups,
            trace_steps=trace_steps,
            success_paths=success_paths,
            atomic_manifest=atomic_manifest,
            unique_states=unique_states,
            full_branch_graph=full_branch_graph,
        ),
        "trace": trace_steps,
        "inputs": {
            "graph_path": str(inputs.graph_path),
            "image_manifest_path": str(inputs.image_manifest_path) if inputs.image_manifest_path else None,
            "trace_path": str(inputs.trace_path) if inputs.trace_path else None,
            "success_paths_path": str(inputs.success_paths_path) if inputs.success_paths_path else None,
            "atomic_transition_manifest_path": str(inputs.atomic_transition_manifest_path) if inputs.atomic_transition_manifest_path else None,
            "unique_states_path": str(inputs.unique_states_path) if inputs.unique_states_path else None,
            "full_branch_graph_path": str(inputs.full_branch_graph_path) if inputs.full_branch_graph_path else None,
        },
    }


def write_viz_graph(viz_graph: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(viz_graph, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path | None) -> Any:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_trace(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    steps: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        steps.append(json.loads(line))
    return steps


def _image_manifest_by_state_id(manifest: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for state in manifest.get("states", []) or []:
        state_id = str(state.get("target_state_id", "") or "")
        image_path = state.get("copied_image_path") or state.get("reused_image_path")
        if state_id and image_path:
            result[state_id] = str(image_path)
    return result


def _trace_images_by_node_id(trace_steps: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for step in trace_steps:
        node_id = str(step.get("render_node_id") or step.get("current_state_id") or "")
        images = list(step.get("observation_image_paths", []) or [])
        if node_id and images and node_id not in result:
            result[node_id] = str(images[0])
    return result


def _aggregate_parallel_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, str]] = []
    for edge in edges:
        key = (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("atomic_action") or edge.get("action_type") or edge.get("label") or ""),
        )
        if key not in grouped:
            order.append(key)
        grouped.setdefault(key, []).append(edge)

    result: list[dict[str, Any]] = []
    for index, key in enumerate(order):
        members = grouped[key]
        representative = members[0]
        raw_edge_count = len(members)
        trace_steps = sorted({step for edge in members for step in edge.get("trace_steps", [])})
        raw_edge_ids = [str(edge.get("id")) for edge in members]
        parent_actions = sorted({str(edge.get("parent_action")) for edge in members if edge.get("parent_action")})
        parent_sources = sorted({str(edge.get("parent_source")) for edge in members if edge.get("parent_source")})
        branch_statuses = sorted({str(edge.get("branch_status")) for edge in members if edge.get("branch_status")})
        appears_in_success_paths = sorted({str(path) for edge in members for path in edge.get("appears_in_success_paths", [])})
        appears_in_shortest_success_paths = sorted({str(path) for edge in members for path in edge.get("appears_in_shortest_success_paths", [])})
        appears_in_longer_success_paths = sorted({str(path) for edge in members for path in edge.get("appears_in_longer_success_paths", [])})
        appears_in_full_edges = sorted({str(edge_id) for edge in members for edge_id in edge.get("appears_in_full_edges", [])})
        label = str(representative.get("atomic_action") or representative.get("action_type") or representative.get("label"))
        if raw_edge_count > 1:
            label = f"{label} x{raw_edge_count}"
        result.append(
            {
                **representative,
                "id": f"eg{index}",
                "raw_edge_count": raw_edge_count,
                "raw_edge_ids": raw_edge_ids,
                "parent_actions": parent_actions,
                "parent_sources": parent_sources,
                "branch_statuses": branch_statuses,
                "is_success_edge": any(edge.get("is_success_edge") for edge in members),
                "is_shortest_success_edge": bool(appears_in_shortest_success_paths) or any(edge.get("is_shortest_success_edge") for edge in members),
                "is_trace_edge": any(edge.get("is_trace_edge") for edge in members),
                "trace_steps": trace_steps,
                "appears_in_success_paths": appears_in_success_paths,
                "appears_in_shortest_success_paths": appears_in_shortest_success_paths,
                "appears_in_longer_success_paths": appears_in_longer_success_paths,
                "appears_in_full_edges": appears_in_full_edges,
                "label": label,
                "members": [
                    {
                        "id": edge.get("id"),
                        "parent_edge_id": edge.get("parent_edge_id"),
                        "atomic_step_id": edge.get("atomic_step_id"),
                        "parent_action": edge.get("parent_action"),
                        "parent_source": edge.get("parent_source"),
                        "branch_status": edge.get("branch_status"),
                        "is_success_edge": edge.get("is_success_edge"),
                        "is_shortest_success_edge": edge.get("is_shortest_success_edge"),
                        "is_trace_edge": edge.get("is_trace_edge"),
                        "trace_steps": edge.get("trace_steps", []),
                    }
                    for edge in members
                ],
            }
        )
    return result


def _atomic_transition_by_key(manifest: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for transition in manifest.get("transitions", []) or []:
        key = (
            str(transition.get("before_state_key") or _state_key(transition.get("before_state", []))),
            str(transition.get("after_state_key") or _state_key(transition.get("after_state", []))),
            str(transition.get("atomic_action") or ""),
        )
        result[key] = transition
    return result


def _transition_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("before_state_key") or _state_key(edge.get("before_state", []))),
        str(edge.get("after_state_key") or _state_key(edge.get("after_state", []))),
        str(edge.get("atomic_action") or ""),
    )


def _state_key(predicates: list[str] | tuple[str, ...] | Any) -> str:
    return " | ".join(sorted(str(item) for item in (predicates or [])))


def _extract_full_graph_parts(full_branch_graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not full_branch_graph:
        return [], []
    nodes = full_branch_graph.get("nodes")
    edges = full_branch_graph.get("edges")
    if nodes is None or edges is None:
        graph = full_branch_graph.get("graph", {}) or {}
        nodes = graph.get("nodes", []) if nodes is None else nodes
        edges = graph.get("edges", []) if edges is None else edges
    return list(nodes or []), list(edges or [])


def _full_branch_status_counts(full_edges: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in full_edges:
        status = str(edge.get("branch_status") or edge.get("status") or "explored")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _build_layers(
    *,
    graph: dict[str, Any],
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    edge_groups: list[dict[str, Any]],
    trace_steps: list[dict[str, Any]],
    success_paths: dict[str, Any],
    atomic_manifest: dict[str, Any],
    unique_states: dict[str, Any],
    full_branch_graph: dict[str, Any],
) -> dict[str, Any]:
    return {
        "full": _build_full_graph_layer(graph, full_branch_graph, success_paths, atomic_manifest, unique_states),
        "search": _build_atomic_overlay_layer(
            "search",
            "Search Result",
            "Atomic state graph annotated with search outcomes: shortest success, success, pruned/loop branches, and explored edges.",
            nodes,
            edges,
            edge_groups,
        ),
        "expand": _build_expand_layer(success_paths, edges),
        "image": _build_atomic_overlay_layer(
            "image",
            "Image State",
            "Atomic state graph with captured/reused observation images attached to state nodes.",
            nodes,
            edges,
            edge_groups,
        ),
        "atomic": _build_atomic_overlay_layer(
            "atomic",
            "Atomic Graph",
            "Final offline graph used by Eval. Parallel raw edges are collapsed by source/target/atomic action by default.",
            nodes,
            edges,
            edge_groups,
        ),
        "trace": _build_trace_layer(nodes, edge_groups, trace_steps),
        "pipeline": _build_pipeline_layer(graph, len(nodes), len(edges), len(edge_groups), len(trace_steps)),
    }


def _build_atomic_overlay_layer(
    layer_id: str,
    title: str,
    description: str,
    nodes: list[dict[str, Any]],
    raw_edges: list[dict[str, Any]],
    edge_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    layer_nodes = [{**node, "type": "state", "view": layer_id} for node in nodes]
    return {
        "id": layer_id,
        "title": title,
        "description": description,
        "available": True,
        "layout": "breadthfirst",
        "nodes": layer_nodes,
        "edges": [_with_search_status(edge, layer_id) for edge in edge_groups],
        "raw_edges": [_with_search_status(edge, layer_id) for edge in raw_edges],
        "default_edge_source": "grouped",
        "stats": {
            "node_count": len(layer_nodes),
            "raw_edge_count": len(raw_edges),
            "edge_count": len(edge_groups),
        },
    }


def _with_search_status(edge: dict[str, Any], layer_id: str) -> dict[str, Any]:
    statuses = set(edge.get("branch_statuses") or [])
    if edge.get("branch_status"):
        statuses.add(str(edge.get("branch_status")))
    if edge.get("status"):
        statuses.add(str(edge.get("status")))
    if edge.get("search_status"):
        statuses.add(str(edge.get("search_status")))
    if edge.get("is_trace_edge"):
        search_status = "trace"
    elif edge.get("is_shortest_success_edge"):
        search_status = "shortest"
    elif (
        (edge.get("is_success_edge") or edge.get("appears_in_success_paths") or statuses & {"success", "reaches_goal"})
        and (
            any("pruned" in status for status in statuses)
            or statuses & {"loop", "loop_skipped", "loop_detected", "dead_end", "depth_limit", "budget_not_expanded", "shadow_state_not_expanded"}
        )
    ):
        search_status = "mixed"
    elif edge.get("is_success_edge") or edge.get("appears_in_success_paths") or statuses & {"success", "reaches_goal"}:
        search_status = "success"
    elif any("pruned" in status for status in statuses):
        search_status = "pruned"
    elif statuses & {"loop", "loop_skipped", "loop_detected"}:
        search_status = "loop"
    elif statuses & {"dead_end", "depth_limit", "budget_not_expanded", "shadow_state_not_expanded"}:
        search_status = "stopped"
    else:
        search_status = "explored"
    display_label = _short_action(str(edge.get("atomic_action") or edge.get("label") or edge.get("id")))
    if edge.get("raw_edge_count", 1) > 1:
        display_label = f"{display_label} x{edge.get('raw_edge_count')}"
    return {
        **edge,
        "view": layer_id,
        "search_status": search_status,
        "display_label": display_label,
    }


def _build_trace_layer(
    nodes: list[dict[str, Any]],
    edge_groups: list[dict[str, Any]],
    trace_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_node_ids = {
        str(value)
        for step in trace_steps
        for key in ("current_state_id", "render_node_id", "to_state", "next_state_id")
        for value in [step.get(key)]
        if value
    }
    trace_edges = [edge for edge in edge_groups if edge.get("is_trace_edge")]
    success_backdrop = [edge for edge in edge_groups if edge.get("is_success_edge") and not edge.get("is_trace_edge")]
    for edge in trace_edges + success_backdrop:
        trace_node_ids.add(str(edge.get("source")))
        trace_node_ids.add(str(edge.get("target")))
    layer_nodes = [
        {**node, "type": "state", "view": "trace", "dimmed": node["id"] not in trace_node_ids}
        for node in nodes
    ]
    layer_edges = [
        _with_search_status({**edge, "display_label": _trace_edge_label(edge)}, "trace")
        for edge in success_backdrop + trace_edges
    ]
    return {
        "id": "trace",
        "title": "Agent Trace Overlay",
        "description": "Model decisions overlaid on the Eval atomic graph. Orange edges are the actual trace; green edges are success-path context.",
        "available": True,
        "layout": "breadthfirst",
        "nodes": layer_nodes,
        "edges": layer_edges,
        "raw_edges": layer_edges,
        "default_edge_source": "grouped",
        "stats": {
            "node_count": len(layer_nodes),
            "edge_count": len(layer_edges),
            "trace_step_count": len(trace_steps),
        },
    }


def _trace_edge_label(edge: dict[str, Any]) -> str:
    steps = edge.get("trace_steps") or []
    prefix = f"step {','.join(str(step) for step in steps)}: " if steps else ""
    return prefix + _short_action(str(edge.get("atomic_action") or edge.get("label") or edge.get("id")))


def _build_expand_layer(success_paths: dict[str, Any], raw_edges: list[dict[str, Any]]) -> dict[str, Any]:
    sequences = _parent_action_sequences(success_paths, raw_edges)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for parent_index, (parent_action, steps) in enumerate(sequences.items()):
        parent_id = f"parent_{parent_index}"
        nodes.append(
            {
                "id": parent_id,
                "label": _short_action(parent_action),
                "display_label": _short_action(parent_action),
                "type": "parent_action",
                "view": "expand",
                "parent_action": parent_action,
            }
        )
        previous_id = parent_id
        for step_index, step in enumerate(steps):
            step_id = f"{parent_id}_step_{step_index}"
            nodes.append(
                {
                    "id": step_id,
                    "label": f"{step_index + 1}. {_short_action(str(step.get('atomic_action') or step.get('label') or 'atomic'))}",
                    "display_label": f"{step_index + 1}. {_short_action(str(step.get('atomic_action') or step.get('label') or 'atomic'))}",
                    "type": "atomic_step",
                    "view": "expand",
                    "parent_action": parent_action,
                    "atomic_action": step.get("atomic_action"),
                    "template_id": step.get("template_id"),
                    "action_type": step.get("action_type"),
                    "role_bindings": step.get("role_bindings", {}),
                    "before_state": step.get("before_state", []),
                    "after_state": step.get("after_state", []),
                    "appears_in_success_paths": step.get("appears_in_success_paths", []),
                    "appears_in_full_edges": step.get("appears_in_full_edges", []),
                }
            )
            edges.append(
                {
                    "id": f"expand_{parent_index}_{step_index}",
                    "source": previous_id,
                    "target": step_id,
                    "label": "expands" if step_index == 0 else "then",
                    "display_label": "expands" if step_index == 0 else "then",
                    "view": "expand",
                    "search_status": "expand",
                }
            )
            previous_id = step_id
    return {
        "id": "expand",
        "title": "Expand Mapping",
        "description": "High-level parent actions expanded into atomic actions. This is the parent-edge to atomic-chain view.",
        "available": bool(nodes),
        "layout": "breadthfirst",
        "nodes": nodes or [{"id": "expand_missing", "label": "No expansion data", "display_label": "No expansion data", "type": "missing", "view": "expand"}],
        "edges": edges,
        "raw_edges": edges,
        "default_edge_source": "grouped",
        "stats": {
            "parent_action_count": len(sequences),
            "atomic_step_count": sum(len(steps) for steps in sequences.values()),
        },
    }


def _parent_action_sequences(success_paths: dict[str, Any], raw_edges: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = {}
    seen_actions: dict[str, set[str]] = {}
    for path in success_paths.get("success_paths", []) or []:
        for step in path.get("atomic_steps", []) or []:
            parent_action = str(step.get("parent_action") or "unknown_parent_action")
            atomic_action = str(step.get("atomic_action") or "")
            if not atomic_action:
                continue
            seen = seen_actions.setdefault(parent_action, set())
            if atomic_action in seen:
                continue
            seen.add(atomic_action)
            sequences.setdefault(parent_action, []).append(
                {
                    **step,
                    "appears_in_success_paths": [path.get("path_id")] if path.get("path_id") else [],
                }
            )
    if sequences:
        return sequences
    for edge in raw_edges:
        parent_action = str(edge.get("parent_action") or "unknown_parent_action")
        atomic_action = str(edge.get("atomic_action") or "")
        if not atomic_action:
            continue
        seen = seen_actions.setdefault(parent_action, set())
        if atomic_action in seen:
            continue
        seen.add(atomic_action)
        sequences.setdefault(parent_action, []).append(edge)
    for steps in sequences.values():
        steps.sort(key=lambda step: _atomic_action_sort_key(str(step.get("atomic_action") or "")))
    return sequences


def _atomic_action_sort_key(action: str) -> tuple[int, str]:
    order = {"look_at": 0, "walk_to": 1, "switch_off": 2, "switch_on": 2, "open": 2, "close": 2, "pick": 2, "place": 3}
    prefix = action.split("_", 1)[0] if "_" in action else action
    if action.startswith("look_at"):
        prefix = "look_at"
    elif action.startswith("walk_to"):
        prefix = "walk_to"
    elif action.startswith("switch_off"):
        prefix = "switch_off"
    elif action.startswith("switch_on"):
        prefix = "switch_on"
    return (order.get(prefix, 10), action)


def _build_full_graph_layer(
    graph: dict[str, Any],
    full_branch_graph: dict[str, Any],
    success_paths: dict[str, Any],
    atomic_manifest: dict[str, Any],
    unique_states: dict[str, Any],
) -> dict[str, Any]:
    full_nodes, full_edges = _extract_full_graph_parts(full_branch_graph)
    if full_nodes:
        goal_predicates = set(full_branch_graph.get("goal_state", []) or graph.get("goal_state", []) or [])
        nodes = []
        for index, node in enumerate(full_nodes):
            node_id = str(node.get("id", index))
            state = list(node.get("state") or node.get("predicates") or [])
            is_goal = bool(node.get("is_goal") or node.get("is_goal_state")) or bool(goal_predicates and goal_predicates.issubset(set(state)))
            nodes.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "display_label": _full_node_label(node_id, node, is_goal),
                    "type": "full_state",
                    "view": "full",
                    "state": state,
                    "depth": node.get("depth"),
                    "visit_count": node.get("visit_count"),
                    "expanded_action_count": node.get("expanded_action_count"),
                    "pruned_action_count": node.get("pruned_action_count"),
                    "terminal_branch_statuses": node.get("terminal_branch_statuses", []),
                    "incoming_edge_ids": node.get("incoming_edge_ids", []),
                    "outgoing_edge_ids": node.get("outgoing_edge_ids", []),
                    "pruned_actions": node.get("pruned_actions", []),
                    "is_initial": node_id == "s0" or node.get("depth") == 0,
                    "is_goal": is_goal,
                    "is_dead_end": bool(node.get("is_dead_end")),
                    "source": "full_branch_graph",
                }
            )
        raw_edges = []
        for index, edge in enumerate(full_edges):
            source = edge.get("from") or edge.get("source")
            target = edge.get("to") or edge.get("target")
            if source is None or target is None:
                continue
            status = str(edge.get("branch_status") or edge.get("status") or "explored")
            is_success = bool(edge.get("is_success_edge")) or status in {"success", "reaches_goal"}
            action = str(edge.get("action") or edge.get("parent_action") or edge.get("id", index))
            raw_edges.append(
                {
                    "id": str(edge.get("id", index)),
                    "source": str(source),
                    "target": str(target),
                    "label": action,
                    "display_label": _full_edge_label(action, status),
                    "action": action,
                    "action_type": edge.get("action_type"),
                    "template_id": edge.get("template_id"),
                    "parent_action": edge.get("parent_action"),
                    "parent_source": edge.get("source"),
                    "branch_status": status,
                    "is_success_edge": is_success,
                    "is_shortest_success_edge": bool(edge.get("is_shortest_success_edge")),
                    "type": "full_edge",
                    "view": "full",
                    "rank_score": edge.get("rank_score"),
                    "expansion_status": edge.get("expansion_status"),
                    "coverage_bucket": edge.get("coverage_bucket"),
                    "pruning_reason": edge.get("pruning_reason"),
                    "pre_state": edge.get("pre_state"),
                    "add_effects": edge.get("add_effects"),
                    "delete_effects": edge.get("delete_effects"),
                    "before_state": edge.get("before_state", []),
                    "after_state": edge.get("after_state", []),
                    "atomic_steps": edge.get("atomic_steps", []),
                }
            )
        edges = [_with_search_status(edge, "full") for edge in _aggregate_parallel_edges(raw_edges)]
        raw_edges = [_with_search_status(edge, "full") for edge in raw_edges]
        available = True
        description = "Original full-branch BFS search graph emitted by task generation, before the final atomic graph view."
    else:
        nodes = [
            {
                "id": "full_missing",
                "label": "Full branch graph\nnot available",
                "display_label": "Full branch graph\nnot available",
                "type": "missing",
                "view": "full",
                "detail": "The current Eval input does not include *_full_branch_graph.json.",
            },
            {
                "id": "search_summary",
                "label": f"Search summary\n{success_paths.get('total_success_path_count_in_full_graph', 'unknown')} full success paths",
                "display_label": f"Search summary\n{success_paths.get('total_success_path_count_in_full_graph', 'unknown')} full success paths",
                "type": "summary",
                "view": "full",
                "success_path_summary": success_paths.get("success_path_summary", {}),
            },
            {
                "id": "atomic_summary",
                "label": f"Atomic result\n{unique_states.get('unique_state_count', graph.get('node_count'))} states / {atomic_manifest.get('unique_atomic_transition_count', 'unknown')} transitions",
                "display_label": f"Atomic result\n{unique_states.get('unique_state_count', graph.get('node_count'))} states / {atomic_manifest.get('unique_atomic_transition_count', 'unknown')} transitions",
                "type": "summary",
                "view": "full",
            },
        ]
        edges = [
            {"id": "full_missing_e0", "source": "full_missing", "target": "search_summary", "label": "best-effort", "display_label": "best-effort", "view": "full", "search_status": "stopped"},
            {"id": "full_missing_e1", "source": "search_summary", "target": "atomic_summary", "label": "available outputs", "display_label": "available outputs", "view": "full", "search_status": "success"},
        ]
        available = False
        description = "Full graph file is missing. This layer shows the available search/atomic summaries instead."
        raw_edges = edges
    return {
        "id": "full",
        "title": "Full Graph",
        "description": description,
        "available": available,
        "layout": "breadthfirst",
        "nodes": nodes,
        "edges": edges,
        "raw_edges": raw_edges,
        "default_edge_source": "grouped",
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "raw_edge_count": len(raw_edges),
            "status_counts": _full_branch_status_counts(full_edges),
        },
    }


def _full_node_label(node_id: str, node: dict[str, Any], is_goal: bool) -> str:
    parts = [node_id]
    if node.get("depth") is not None:
        parts.append(f"d={node.get('depth')}")
    if is_goal:
        parts.append("goal")
    elif node.get("is_dead_end"):
        parts.append("dead_end")
    pruned = int(node.get("pruned_action_count") or 0)
    if pruned:
        parts.append(f"pruned={pruned}")
    return "\n".join(parts)


def _full_edge_label(action: str, status: str) -> str:
    label = _short_action(action)
    if status and status not in {"continues", "explored"}:
        return f"{label}\n[{status}]"
    return label


def _build_pipeline_layer(graph: dict[str, Any], node_count: int, edge_count: int, edge_group_count: int, trace_step_count: int) -> dict[str, Any]:
    nodes = [
        {"id": "scene", "label": "Scene\nobjects/templates", "level": "private", "type": "pipeline", "view": "pipeline", "detail": "TongSim scene object set and action templates."},
        {"id": "taskgen", "label": "Task generation\nL1/L2/L3", "level": "private", "type": "pipeline", "view": "pipeline", "detail": "Generate compositional task descriptions and predicates."},
        {"id": "fullbranch", "label": "Full-branch\nsearch graph", "level": "private", "type": "pipeline", "view": "pipeline", "detail": f"Source graph: {graph.get('source_graph')}."},
        {"id": "atomic", "label": f"Atomic graph\n{node_count} nodes / {edge_count} raw edges", "level": "private", "type": "pipeline", "view": "pipeline", "detail": f"Expanded into atomic actions; {edge_group_count} visual edge groups after collapsing parallel edges."},
        {"id": "capture", "label": "Image capture\nstate manifest", "level": "private", "type": "pipeline", "view": "pipeline", "detail": "Map unique states to captured/reused observation images."},
        {"id": "eval", "label": "Eval graph_env\ntext + image + actions", "level": "public", "type": "pipeline", "view": "pipeline", "detail": "Offline environment exposes one observation and candidate actions each step."},
        {"id": "agent", "label": "Agent\nHermes/Codex/etc.", "level": "public", "type": "pipeline", "view": "pipeline", "detail": "Agent chooses one action template and bound objects."},
        {"id": "trace", "label": f"Trace + score\n{trace_step_count} steps", "level": "output", "type": "pipeline", "view": "pipeline", "detail": "Recorded trace.jsonl, decision logs, grade/score files."},
    ]
    edges = [
        {"id": "p0", "source": "scene", "target": "taskgen", "label": "enumerate", "display_label": "enumerate", "view": "pipeline"},
        {"id": "p1", "source": "taskgen", "target": "fullbranch", "label": "search", "display_label": "search", "view": "pipeline"},
        {"id": "p2", "source": "fullbranch", "target": "atomic", "label": "expand", "display_label": "expand", "view": "pipeline"},
        {"id": "p3", "source": "atomic", "target": "capture", "label": "state images", "display_label": "state images", "view": "pipeline"},
        {"id": "p4", "source": "capture", "target": "eval", "label": "package", "display_label": "package", "view": "pipeline"},
        {"id": "p5", "source": "atomic", "target": "eval", "label": "graph", "display_label": "graph", "view": "pipeline"},
        {"id": "p6", "source": "eval", "target": "agent", "label": "observe", "display_label": "observe", "view": "pipeline"},
        {"id": "p7", "source": "agent", "target": "eval", "label": "choose_action", "display_label": "choose_action", "view": "pipeline"},
        {"id": "p8", "source": "eval", "target": "trace", "label": "record", "display_label": "record", "view": "pipeline"},
    ]
    return {
        "id": "pipeline",
        "title": "Pipeline Overview",
        "description": "Auxiliary process overview. The real graph layers above are the primary visualization.",
        "available": True,
        "layout": "breadthfirst",
        "nodes": nodes,
        "edges": edges,
        "raw_edges": edges,
        "default_edge_source": "grouped",
        "stats": {"node_count": len(nodes), "edge_count": len(edges)},
    }


def _short_action(action: str) -> str:
    if "__" in action:
        left, right = action.split("__", 1)
        left = left.replace("_{object}", "").replace("{object}", "object")
        return f"{left}\n{right}"
    prefixes = ["switch_off_", "switch_on_", "look_at_", "walk_to_", "open_", "close_", "pick_up_", "place_"]
    for prefix in prefixes:
        if action.startswith(prefix):
            return f"{prefix.rstrip('_')}\n{action[len(prefix):]}"
    return action


def _trace_steps_by_node_id(trace_steps: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for step in trace_steps:
        step_index = int(step.get("step_index", len(result)))
        for key in ("current_state_id", "render_node_id", "to_state"):
            node_id = step.get(key)
            if node_id:
                result.setdefault(str(node_id), []).append(step_index)
    return {key: sorted(set(values)) for key, values in result.items()}


def _select_trace_edges(edges: list[dict[str, Any]], trace_steps: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for step in trace_steps:
        matches = [edge for edge in edges if _edge_matches_step(edge, step)]
        if not matches:
            continue
        matches.sort(key=_trace_edge_sort_key)
        edge_id = str(matches[0].get("id"))
        result.setdefault(edge_id, []).append(int(step.get("step_index", len(result))))
    return result


def _trace_edge_sort_key(edge: dict[str, Any]) -> tuple[int, int, int]:
    edge_id = str(edge.get("id", ""))
    numeric = int(edge_id[2:]) if edge_id.startswith("ae") and edge_id[2:].isdigit() else 10**9
    return (
        0 if edge.get("is_success_edge") else 1,
        0 if edge.get("branch_status") == "continues" else 1,
        numeric,
    )


def _edge_matches_step(edge: dict[str, Any], step: dict[str, Any]) -> bool:
    source = step.get("current_state_id")
    target = step.get("to_state")
    if source and str(edge.get("from")) != str(source):
        return False
    if target and str(edge.get("to")) != str(target):
        return False
    canonical = step.get("canonical_action") or step.get("selected_action_id")
    if canonical and canonical == edge.get("atomic_action"):
        return True
    if step.get("selected_template_id") and step.get("selected_template_id") != edge.get("template_id"):
        return False
    if step.get("selected_action_type") and step.get("selected_action_type") != edge.get("action_type"):
        return False
    step_objects = {str(value) for value in (step.get("selected_role_bindings", {}) or {}).values()}
    edge_objects = {str(value) for value in (edge.get("role_bindings", {}) or {}).values()}
    return not step_objects or bool(step_objects & edge_objects)
