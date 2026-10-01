#!/usr/bin/env python3
"""Build a compact, recoverable V2 graph from the shared V1 search.

V1 remains the reference implementation. V2 first runs the same guided
planner, then keeps the shortest successful path plus a deterministic subset
of longer successful paths that fit the per-task atomic-state budget.  A
longer successful path is the source of a real valid-detour transition; no
synthetic transition or state is introduced.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any


def load_search_module(path: Path):
    spec = importlib.util.spec_from_file_location("tongbench_guided_dag_search_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load graph search module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def state_key(state: list[str] | tuple[str, ...]) -> str:
    return " | ".join(sorted(str(value) for value in state or []))


def path_steps(path: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in path.get("atomic_steps", []) or [] if isinstance(item, dict)]


def path_state_keys(path: dict[str, Any]) -> list[str]:
    steps = path_steps(path)
    if not steps:
        return []
    keys = [state_key(list(steps[0].get("before_state", []) or []))]
    keys.extend(state_key(list(step.get("after_state", []) or [])) for step in steps)
    return keys


def path_actions_by_state(path: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for step in path_steps(path):
        before = state_key(list(step.get("before_state", []) or []))
        action = str(step.get("atomic_action") or step.get("action") or "")
        if before and action:
            result.setdefault(before, set()).add(action)
    return result


def find_v1_unique_states(v1_graph_dir: Path, task_id: str) -> tuple[Path, int]:
    candidates = sorted(v1_graph_dir.glob(f"{task_id}*unique_states.json"))
    if not candidates:
        raise FileNotFoundError(f"V1 unique-state manifest not found for {task_id} in {v1_graph_dir}")
    path = candidates[0]
    payload = read_json(path)
    count = int(payload.get("unique_state_count") or len(payload.get("states", []) or []))
    if count <= 0:
        raise ValueError(f"V1 unique-state manifest is empty: {path}")
    return path, count


def image_safe_paths(
    paths: list[dict[str, Any]],
    *,
    image_manifest_root: Path,
    task_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_path = image_manifest_root / task_id / "task_image_manifest.json"
    if not image_path.is_file():
        raise FileNotFoundError(f"Image manifest not found for {task_id}: {image_path}")
    payload = read_json(image_path)
    image_ok_keys = {
        state_key(item.get("target_state", []))
        for item in payload.get("states", []) or []
        if isinstance(item, dict) and str(item.get("match_status") or "") == "reused"
    }
    safe: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in paths:
        missing = sorted(set(path_state_keys(path)) - image_ok_keys)
        if missing:
            rejected.append({"path_id": path.get("path_id"), "missing_state_count": len(missing), "missing_state_keys": missing[:10]})
        else:
            safe.append(path)
    return safe, {
        "enabled": True,
        "manifest_path": str(image_path),
        "image_ok_state_count": len(image_ok_keys),
        "input_success_path_count": len(paths),
        "image_safe_success_path_count": len(safe),
        "image_rejected_success_path_count": len(rejected),
        "rejected_paths": rejected[:50],
    }


def prepare_guided_search(args: argparse.Namespace) -> dict[str, Any]:
    sdk = args.sdk_root.resolve()
    guided = load_search_module(sdk / "dag_task_search_success_only_autopath_supported_detailed.py")
    planner = guided.Planner(
        args.base_dir.resolve(),
        sdk / "outputs",
        args.rules.resolve(),
        args.subtasks.resolve(),
        tasks_path=args.tasks_file.resolve(),
        atomic_path=args.atomic_file.resolve(),
    )
    planner.rules.setdefault("search", {})["max_success_traces"] = args.max_success_traces

    seed = planner.search_task(
        args.task_id,
        search_mode="success_only",
        max_depth_override=args.max_depth,
        max_nodes=max(args.max_nodes, 5000),
        max_outgoing_per_node=args.max_outgoing_per_node,
        action_level="subtask",
    )
    traces = seed.get("success_traces") or []
    if not seed.get("has_solution") or not traces:
        raise RuntimeError(f"No success-only path exists for {args.task_id}")
    preferred_actions = set(traces[0])
    original_score = planner.score_action_for_branch

    def guided_score(
        self,
        action,
        before,
        after,
        goal,
        goal_relevance_predicates,
        task_scope,
        intended_skill_signature,
        path_action_types,
    ):
        score = original_score(
            action=action,
            before=before,
            after=after,
            goal=goal,
            goal_relevance_predicates=goal_relevance_predicates,
            task_scope=task_scope,
            intended_skill_signature=intended_skill_signature,
            path_action_types=path_action_types,
        )
        if action.name in preferred_actions:
            score += 1000.0
        return score

    import types

    planner.score_action_for_branch = types.MethodType(guided_score, planner)
    result = planner.search_task(
        args.task_id,
        search_mode="full_branch",
        max_depth_override=args.max_depth,
        max_nodes=args.max_nodes,
        max_outgoing_per_node=args.max_outgoing_per_node,
        prune_strategy=args.prune_strategy,
        success_path_output="all",
        action_level="subtask",
    )
    if not result.get("has_solution"):
        raise RuntimeError(f"Guided full-branch search did not retain a success path for {args.task_id}")
    return {"planner": planner, "result": result, "preferred_action_count": len(preferred_actions)}


def atomic_edge_index(graph: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []) or []:
        before = state_key(list(edge.get("before_state", []) or []))
        after = state_key(list(edge.get("after_state", []) or []))
        action = str(edge.get("atomic_action") or edge.get("action") or "")
        if not action:
            continue
        index.setdefault((before, action, after), []).append(edge)
    return index


def select_paths(
    all_paths: list[dict[str, Any]],
    *,
    state_budget: int,
    max_detour_paths: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usable = [item for item in all_paths if path_steps(item)]
    if not usable:
        raise RuntimeError("Full graph returned no usable successful paths")
    min_len = min(int(item.get("atomic_len") or len(path_steps(item))) for item in usable)
    shortest = [item for item in usable if int(item.get("atomic_len") or len(path_steps(item))) == min_len]
    shortest = sorted(shortest, key=lambda item: str(item.get("path_id") or ""))
    chosen_shortest = copy.deepcopy(shortest[0])
    chosen_shortest["is_shortest_success_path"] = True
    chosen_keys = set(path_state_keys(chosen_shortest))
    selected_actions = path_actions_by_state(chosen_shortest)
    original_shortest_count = len(chosen_keys)

    candidates = []
    for item in usable:
        length = int(item.get("atomic_len") or len(path_steps(item)))
        if length <= min_len:
            continue
        keys = set(path_state_keys(item))
        new_keys = keys - chosen_keys
        if not new_keys:
            continue
        candidates.append((len(new_keys), length, str(item.get("path_id") or ""), item, keys))
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))

    selected = [chosen_shortest]
    selected_ids = {str(chosen_shortest.get("path_id") or "")}
    selected_detour_ids: list[str] = []
    skipped_for_outgoing_limit: list[str] = []
    for _marginal, _length, _path_id, item, keys in candidates:
        if len(selected_detour_ids) >= max(0, max_detour_paths):
            break
        if len(chosen_keys | keys) > state_budget:
            continue
        candidate_actions = path_actions_by_state(item)
        if any(len(selected_actions.get(key, set()) | actions) > 2 for key, actions in candidate_actions.items()):
            skipped_for_outgoing_limit.append(str(item.get("path_id") or ""))
            continue
        clone = copy.deepcopy(item)
        clone["is_shortest_success_path"] = False
        selected.append(clone)
        selected_ids.add(str(clone.get("path_id") or ""))
        selected_detour_ids.append(str(clone.get("path_id") or ""))
        chosen_keys.update(keys)
        for key, actions in candidate_actions.items():
            selected_actions.setdefault(key, set()).update(actions)

    return selected, {
        "shortest_atomic_length": min_len,
        "shortest_state_count": original_shortest_count,
        "candidate_success_path_count": len(usable),
        "candidate_longer_path_count": len(candidates),
        "selected_detour_path_ids": selected_detour_ids,
        "skipped_for_max_two_valid_actions_per_state": skipped_for_outgoing_limit,
        "states_with_one_graph_supported_detour": sum(1 for actions in selected_actions.values() if len(actions) == 2),
        "max_graph_supported_valid_actions_per_state": max((len(actions) for actions in selected_actions.values()), default=0),
        "selected_success_path_count": len(selected),
        "selected_state_count_before_node_filter": len(chosen_keys),
        "state_budget": state_budget,
        "unused_state_budget": max(0, state_budget - len(chosen_keys)),
    }


def build_reduced_graph(
    full_graph: dict[str, Any],
    selected_paths: list[dict[str, Any]],
    *,
    full_success_path_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    edge_index = atomic_edge_index(full_graph)
    selected_edges: list[dict[str, Any]] = []
    selected_edge_ids: set[str] = set()
    selected_keys: set[str] = set()
    missing_transitions: list[dict[str, Any]] = []
    for path in selected_paths:
        for step in path_steps(path):
            before = state_key(list(step.get("before_state", []) or []))
            after = state_key(list(step.get("after_state", []) or []))
            action = str(step.get("atomic_action") or step.get("action") or "")
            matches = edge_index.get((before, action, after), [])
            if not matches:
                missing_transitions.append({"action": action, "before_state_key": before, "after_state_key": after})
                continue
            edge = matches[0]
            edge_id = str(edge.get("id") or "")
            if edge_id not in selected_edge_ids:
                selected_edges.append(copy.deepcopy(edge))
                selected_edge_ids.add(edge_id)
            selected_keys.add(before)
            selected_keys.add(after)
    if missing_transitions:
        raise RuntimeError(f"Selected success transition missing from atomic graph: {missing_transitions[:3]}")

    node_by_key = {
        state_key(list(node.get("state", []) or [])): node
        for node in full_graph.get("nodes", []) or []
        if isinstance(node, dict)
    }
    missing_nodes = sorted(selected_keys - set(node_by_key))
    if missing_nodes:
        raise RuntimeError(f"Selected path state missing from atomic graph: {missing_nodes[:3]}")
    nodes = [copy.deepcopy(node_by_key[key]) for key in node_by_key if key in selected_keys]
    old_to_new_node_id: dict[str, str] = {}
    for index, node in enumerate(nodes):
        old_id = str(node.get("id") or "")
        new_id = f"as{index}"
        old_to_new_node_id[old_id] = new_id
        node["id"] = new_id
        node["unique_state_id"] = f"state_{index:04d}"
    for node in nodes:
        node["incoming_edge_ids"] = []
        node["outgoing_edge_ids"] = []
    edges = []
    for edge_index, edge in enumerate(selected_edges):
        old_from = str(edge.get("from") or "")
        old_to = str(edge.get("to") or "")
        if old_from not in old_to_new_node_id or old_to not in old_to_new_node_id:
            raise RuntimeError(f"Selected edge endpoint missing: {edge.get('id')}")
        edge["id"] = f"ae{edge_index}"
        edge["from"] = old_to_new_node_id[old_from]
        edge["to"] = old_to_new_node_id[old_to]
        edge["branch_status"] = "reaches_goal" if edge.get("is_success_edge") else edge.get("branch_status")
        edges.append(edge)
        source = next(node for node in nodes if str(node.get("id")) == str(edge.get("from")))
        target = next(node for node in nodes if str(node.get("id")) == str(edge.get("to")))
        source["outgoing_edge_ids"].append(edge.get("id"))
        target["incoming_edge_ids"].append(edge.get("id"))

    reduced = {
        "task_id": full_graph.get("task_id"),
        "description": full_graph.get("description", ""),
        "source_graph": "v2_shortest_plus_recoverable_detours",
        "expansion_mode": full_graph.get("expansion_mode", "atomic_state_nodes"),
        "initial_state": copy.deepcopy(full_graph.get("initial_state", [])),
        "goal_state": copy.deepcopy(full_graph.get("goal_state", [])),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }
    selected_path_payload = {
        "task_id": full_graph.get("task_id"),
        "success_path_output": "v2_selected_shortest_and_recoverable_detours",
        "total_success_path_count_in_full_graph": full_success_path_count,
        "unique_atomic_success_path_count": len(selected_paths),
        "shortest_success_path_count": 1,
        "min_success_atomic_len": min(int(item.get("atomic_len") or len(path_steps(item))) for item in selected_paths),
        "success_path_count": len(selected_paths),
        "success_paths": selected_paths,
        "success_path_summary": {
            "source": "candidate_full_branch_graph",
            "selected_longer_success_paths_are_recoverable_detours": True,
        },
    }
    return reduced, selected_path_payload


def validate_v2(
    reduced_graph: dict[str, Any],
    success_paths: dict[str, Any],
    *,
    state_budget: int,
) -> dict[str, Any]:
    nodes = {str(node.get("id")): node for node in reduced_graph.get("nodes", []) or []}
    edge_keys = {
        (
            state_key(list(edge.get("before_state", []) or [])),
            str(edge.get("atomic_action") or ""),
            state_key(list(edge.get("after_state", []) or [])),
        )
        for edge in reduced_graph.get("edges", []) or []
    }
    paths = [item for item in success_paths.get("success_paths", []) or [] if isinstance(item, dict)]
    issues: list[str] = []
    path_reports: list[dict[str, Any]] = []
    for path in paths:
        steps = path_steps(path)
        transitions_ok = True
        for step in steps:
            key = (
                state_key(list(step.get("before_state", []) or [])),
                str(step.get("atomic_action") or ""),
                state_key(list(step.get("after_state", []) or [])),
            )
            if key not in edge_keys:
                transitions_ok = False
        path_reports.append({"path_id": path.get("path_id"), "atomic_len": len(steps), "transitions_ok": transitions_ok})
        if not transitions_ok:
            issues.append(f"path {path.get('path_id')} has missing transition")
    initial_key = state_key(list(reduced_graph.get("initial_state", []) or []))
    goal_predicates = set(str(value) for value in reduced_graph.get("goal_state", []) or [])
    node_keys = {state_key(list(node.get("state", []) or [])) for node in nodes.values()}
    if initial_key not in node_keys:
        issues.append("initial state missing")
    goal_node_ids = {
        node_id
        for node_id, node in nodes.items()
        if goal_predicates.issubset(set(str(value) for value in node.get("state", []) or []))
    }
    if not goal_node_ids:
        issues.append("goal state missing")
    if len(nodes) > state_budget:
        issues.append(f"state budget exceeded: {len(nodes)} > {state_budget}")
    shortest = [item for item in paths if item.get("is_shortest_success_path")]
    detours = [item for item in paths if not item.get("is_shortest_success_path")]
    valid = not issues and bool(shortest) and all(item["transitions_ok"] for item in path_reports)
    return {
        "ok": valid,
        "issues": issues,
        "node_count": len(nodes),
        "edge_count": len(reduced_graph.get("edges", []) or []),
        "state_budget": state_budget,
        "shortest_path_count": len(shortest),
        "selected_recoverable_detour_path_count": len(detours),
        "path_reports": path_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--atomic-file", type=Path, required=True)
    parser.add_argument("--subtasks", type=Path, required=True)
    parser.add_argument("--tasks-file", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v1-graph-dir", type=Path, required=True)
    parser.add_argument("--image-manifest-root", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--max-outgoing-per-node", type=int, required=True)
    parser.add_argument("--max-success-traces", type=int, default=500)
    parser.add_argument("--max-detour-paths", type=int, default=24)
    parser.add_argument("--state-budget-divisor", type=int, default=3)
    parser.add_argument(
        "--max-state-cap",
        type=int,
        default=100,
        help="Maximum retained atomic states per task; the shortest path may exceed it if necessary.",
    )
    parser.add_argument("--prune-strategy", default="goal_relevance", choices=["goal_relevance", "state_coverage"])
    args = parser.parse_args()

    v1_path, v1_count = find_v1_unique_states(args.v1_graph_dir.resolve(), args.task_id)
    divisor = max(1, int(args.state_budget_divisor))
    nominal_budget = int(math.ceil(float(v1_count) / divisor))
    search = prepare_guided_search(args)
    planner = search["planner"]
    candidate = search["result"]
    all_paths = candidate.get("success_paths", {}).get("success_paths", []) or []
    image_summary = {"enabled": False}
    if args.image_manifest_root is not None:
        all_paths, image_summary = image_safe_paths(
            all_paths,
            image_manifest_root=args.image_manifest_root.resolve(),
            task_id=args.task_id,
        )
        if not all_paths:
            raise RuntimeError(f"No image-safe successful path remains for {args.task_id}")
    min_len = min(int(item.get("atomic_len") or len(path_steps(item))) for item in all_paths if path_steps(item))
    shortest_state_count = len(path_state_keys(next(item for item in all_paths if int(item.get("atomic_len") or len(path_steps(item))) == min_len)))
    max_state_cap = max(0, int(args.max_state_cap))
    if max_state_cap:
        # Keep every shortest-path state, then use the cap to limit optional branches.
        state_budget = max(shortest_state_count, min(nominal_budget, max_state_cap))
    else:
        state_budget = max(nominal_budget, shortest_state_count)
    selected_paths, selection = select_paths(
        all_paths,
        state_budget=state_budget,
        max_detour_paths=args.max_detour_paths,
    )
    candidate_graph = planner.build_atomic_expanded_graph(args.task_id, candidate)
    reduced_graph, selected_success = build_reduced_graph(
        candidate_graph,
        selected_paths,
        full_success_path_count=len(all_paths),
    )
    validation = validate_v2(reduced_graph, selected_success, state_budget=state_budget)
    if not validation["ok"]:
        raise RuntimeError(f"V2 validation failed for {args.task_id}: {validation['issues']}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{args.task_id}_atomic_expanded_graph.json", reduced_graph)
    write_json(output_dir / f"{args.task_id}_success_paths.json", selected_success)
    write_json(output_dir / f"{args.task_id}_unique_states.json", planner.build_unique_state_manifest(args.task_id, reduced_graph))
    write_json(
        output_dir / f"{args.task_id}_atomic_transition_manifest.json",
        planner.build_atomic_transition_manifest(
            args.task_id,
            selected_success,
            selected_success.get("success_paths", []),
            reduced_graph.get("edges", []),
        ),
    )
    write_json(
        output_dir / f"{args.task_id}_v2_selection_manifest.json",
        {
            "task_id": args.task_id,
            "version": "v2",
            "v1_unique_states_path": str(v1_path),
            "v1_unique_state_count": v1_count,
            "state_budget_divisor": divisor,
            "nominal_state_budget": nominal_budget,
            "state_budget": state_budget,
            "max_state_cap": max_state_cap or None,
            "state_budget_cap_exception": bool(max_state_cap and shortest_state_count > max_state_cap),
            "state_budget_exception": shortest_state_count > nominal_budget,
            "selection_policy": {
                "mandatory": "one shortest successful path",
                "optional": "real longer successful paths only; each is recoverable to the goal",
                "max_detour_paths": args.max_detour_paths,
                "max_valid_actions_per_state": 2,
                "synthetic_states_or_transitions": False,
                "image_safe_success_paths_only": bool(args.image_manifest_root),
            },
            "selection": selection,
            "image_safety": image_summary,
            "validation": validation,
            "candidate_search": {
                "max_depth": args.max_depth,
                "max_nodes": args.max_nodes,
                "max_outgoing_per_node": args.max_outgoing_per_node,
                "prune_strategy": args.prune_strategy,
            },
        },
    )
    print(json.dumps({
        "ok": True,
        "task_id": args.task_id,
        "v1_unique_state_count": v1_count,
        "nominal_state_budget": nominal_budget,
        "state_budget": state_budget,
        "v2_unique_state_count": validation["node_count"],
        "v2_edge_count": validation["edge_count"],
        "shortest_atomic_length": selection["shortest_atomic_length"],
        "selected_detour_path_count": validation["selected_recoverable_detour_path_count"],
        "selected_detour_path_ids": selection["selected_detour_path_ids"],
        "v1": str(v1_path),
        "output_dir": str(output_dir),
        "selection_manifest": str(output_dir / f"{args.task_id}_v2_selection_manifest.json"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
