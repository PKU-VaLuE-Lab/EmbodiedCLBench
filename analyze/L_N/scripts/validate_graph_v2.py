#!/usr/bin/env python3
"""Validate compact V2 graph artifacts without invoking the planner or API."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def state_key(values) -> tuple[str, ...]:
    return tuple(sorted(str(value) for value in values or []))


def path_steps(path: dict):
    return [item for item in path.get("atomic_steps", []) or [] if isinstance(item, dict)]


def validate_task(graph: dict, success: dict, manifest: dict, image_manifest_root: Path | None = None) -> dict:
    issues = []
    nodes = {str(item.get("id")): item for item in graph.get("nodes", []) or []}
    edges = [item for item in graph.get("edges", []) or [] if isinstance(item, dict)]
    outgoing = defaultdict(list)
    outgoing_actions = defaultdict(set)
    incoming = defaultdict(list)
    for edge in edges:
        source = str(edge.get("from") or "")
        target = str(edge.get("to") or "")
        if source not in nodes or target not in nodes:
            issues.append(f"edge endpoint missing: {edge.get('id')}")
        outgoing[source].append(target)
        outgoing_actions[source].add(str(edge.get("atomic_action") or edge.get("action") or edge.get("id") or ""))
        incoming[target].append(source)
    initial_ids = [node_id for node_id, node in nodes.items() if node.get("is_initial_state")]
    if len(initial_ids) != 1:
        issues.append(f"expected one initial node, got {len(initial_ids)}")
    goal_predicates = set(str(value) for value in graph.get("goal_state", []) or [])
    goal_ids = {
        node_id for node_id, node in nodes.items()
        if goal_predicates.issubset(set(str(value) for value in node.get("state", []) or []))
    }
    if not goal_ids:
        issues.append("no node satisfies goal predicates")
    reachable = set(initial_ids)
    queue = deque(initial_ids)
    while queue:
        source = queue.popleft()
        for target in outgoing.get(source, []):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    can_reach_goal = set(goal_ids)
    queue = deque(goal_ids)
    while queue:
        target = queue.popleft()
        for source in incoming.get(target, []):
            if source not in can_reach_goal:
                can_reach_goal.add(source)
                queue.append(source)
    if reachable != set(nodes):
        issues.append(f"disconnected nodes: {len(set(nodes) - reachable)}")
    if can_reach_goal != set(nodes):
        issues.append(f"non-recoverable nodes: {len(set(nodes) - can_reach_goal)}")

    shortest_distance = None
    if initial_ids and goal_ids:
        distances = {initial_ids[0]: 0}
        queue = deque([initial_ids[0]])
        while queue:
            source = queue.popleft()
            for target in outgoing.get(source, []):
                if target in distances:
                    continue
                distances[target] = distances[source] + 1
                queue.append(target)
        goal_distances = [distances[node_id] for node_id in goal_ids if node_id in distances]
        shortest_distance = min(goal_distances) if goal_distances else None
        if shortest_distance is None:
            issues.append("goal is not reachable from initial node")

    paths = [item for item in success.get("success_paths", []) or [] if isinstance(item, dict)]
    shortest_paths = [item for item in paths if item.get("is_shortest_success_path")]
    if len(shortest_paths) != 1:
        issues.append(f"expected one selected shortest path, got {len(shortest_paths)}")
    path_lengths = [len(path_steps(item)) for item in paths]
    if shortest_distance is not None and shortest_paths and len(path_steps(shortest_paths[0])) != shortest_distance:
        issues.append("recorded shortest path differs from graph shortest distance")
    shortest_path_length = min(path_lengths) if path_lengths else None
    if any(length <= shortest_path_length for length in path_lengths if shortest_path_length is not None and length != shortest_path_length):
        issues.append("a selected detour path is not longer than the shortest path")
    max_outgoing = max((len(actions) for actions in outgoing_actions.values()), default=0)
    if max_outgoing > 2:
        issues.append(f"more than one alternate valid action at a state: max_outgoing={max_outgoing}")

    validation = manifest.get("validation", {})
    state_budget = int(manifest.get("state_budget") or 0)
    if state_budget and len(nodes) > state_budget:
        issues.append(f"state budget exceeded: {len(nodes)} > {state_budget}")
    if validation.get("ok") is not True:
        issues.append("selection manifest validation is not ok")
    image_total = None
    image_missing = None
    if image_manifest_root is not None:
        image_path = image_manifest_root / str(graph.get("task_id")) / "task_image_manifest.json"
        if not image_path.is_file():
            issues.append(f"image manifest missing: {image_path}")
            image_missing = len(nodes)
            image_total = len(nodes)
        else:
            image_payload = read_json(image_path)
            image_by_state = {
                state_key(item.get("target_state", [])): item
                for item in image_payload.get("states", []) or []
                if isinstance(item, dict)
            }
            missing_state_ids = [
                node_id
                for node_id, node in nodes.items()
                if str(image_by_state.get(state_key(node.get("state", [])), {}).get("match_status") or "") != "reused"
            ]
            if missing_state_ids:
                issues.append(f"missing images for {len(missing_state_ids)} selected states")
            image_missing = len(missing_state_ids)
            image_total = len(nodes)
    return {
        "task_id": str(graph.get("task_id") or manifest.get("task_id") or ""),
        "ok": not issues,
        "issues": issues,
        "state_count": len(nodes),
        "edge_count": len(edges),
        "state_budget": state_budget,
        "shortest_atomic_length": len(path_steps(shortest_paths[0])) if shortest_paths else None,
        "graph_shortest_distance": shortest_distance,
        "success_path_count": len(paths),
        "detour_path_count": len(paths) - len(shortest_paths),
        "max_outgoing_actions": max_outgoing,
        "image_checked": image_manifest_root is not None,
        "image_state_count": image_total,
        "image_missing_state_count": image_missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--image-manifest-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    graph_dir = args.graph_dir.resolve()
    rows = []
    for manifest_path in sorted(graph_dir.glob("*_v2_selection_manifest.json")):
        manifest = read_json(manifest_path)
        task_id = str(manifest.get("task_id") or manifest_path.name.split("_v2_selection_manifest", 1)[0])
        graph = read_json(graph_dir / f"{task_id}_atomic_expanded_graph.json")
        success = read_json(graph_dir / f"{task_id}_success_paths.json")
        rows.append(validate_task(graph, success, manifest, args.image_manifest_root.resolve() if args.image_manifest_root else None))
    summary = {
        "ok": bool(rows) and all(row["ok"] for row in rows),
        "task_count": len(rows),
        "failure_count": sum(not row["ok"] for row in rows),
        "state_min": min((row["state_count"] for row in rows), default=0),
        "state_max": max((row["state_count"] for row in rows), default=0),
        "state_avg": (sum(row["state_count"] for row in rows) / len(rows)) if rows else 0,
        "detour_path_min": min((row["detour_path_count"] for row in rows), default=0),
        "detour_path_max": max((row["detour_path_count"] for row in rows), default=0),
        "image_checked": args.image_manifest_root is not None,
        "image_missing_state_count": sum(int(row.get("image_missing_state_count") or 0) for row in rows),
        "results": rows,
    }
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
