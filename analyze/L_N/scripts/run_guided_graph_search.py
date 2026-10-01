#!/usr/bin/env python3
"""Run the shared graph search with a task-local success-path guide.

The shared full-branch search ranks independent goal actions ahead of
precondition-enabling actions.  For composite tasks this can prune the only
way to reach a later goal before the branch budget is exhausted.  We first
obtain a real success-only path, then use it only as a ranking hint while
retaining the shared full-branch implementation and its other candidates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import types
from pathlib import Path


def load_search_module(path: Path):
    spec = importlib.util.spec_from_file_location("tongbench_guided_dag_search", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load graph search module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, required=True)
    parser.add_argument("--max-outgoing-per-node", type=int, required=True)
    parser.add_argument("--max-success-traces", type=int, default=100)
    parser.add_argument("--prune-strategy", default="goal_relevance", choices=["goal_relevance", "state_coverage"])
    parser.add_argument("--skip-full-branch-graph", action="store_true")
    args = parser.parse_args()

    sdk = args.sdk_root.resolve()
    search_module = load_search_module(sdk / "dag_task_search_success_only_autopath_supported_detailed.py")
    planner = search_module.Planner(
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

    planner.score_action_for_branch = types.MethodType(guided_score, planner)
    result = planner.search_task(
        args.task_id,
        search_mode="full_branch",
        max_depth_override=args.max_depth,
        max_nodes=args.max_nodes,
        max_outgoing_per_node=args.max_outgoing_per_node,
        prune_strategy=args.prune_strategy,
        success_path_output="shortest_only",
        action_level="subtask",
    )
    if not result.get("has_solution"):
        raise RuntimeError(
            f"Guided full-branch search did not retain a success path for {args.task_id}; "
            f"nodes={result.get('node_count')}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = result["task_id"]
    full_graph = None
    if not args.skip_full_branch_graph:
        full_graph = output_dir / f"{task_id}_full_branch_graph.json"
        write_json(full_graph, result)
    success_paths = output_dir / f"{task_id}_success_paths.json"
    atomic_manifest = output_dir / f"{task_id}_atomic_transition_manifest.json"
    unique_states = output_dir / f"{task_id}_unique_states.json"
    atomic_expanded = output_dir / f"{task_id}_atomic_expanded_graph.json"
    write_json(success_paths, planner.public_success_paths_payload(result["success_paths"]))
    write_json(atomic_manifest, result["atomic_transition_manifest"])
    write_json(unique_states, result["unique_state_manifest"])
    write_json(atomic_expanded, planner.build_atomic_expanded_graph(task_id, result))

    coverage_path = None
    if result.get("state_coverage_manifest") is not None:
        coverage_path = output_dir / f"{task_id}_state_coverage_manifest.json"
        write_json(coverage_path, result["state_coverage_manifest"])

    print(json.dumps({
        "task_id": task_id,
        "has_solution": bool(result.get("has_solution")),
        "node_count": result.get("node_count"),
        "transition_edge_count": result.get("transition_edge_count"),
        "success_trace_count": result.get("success_trace_count", result.get("success_paths", {}).get("success_path_count", 0)),
        "goal_path_count": len(result.get("goal_paths", [])),
        "max_path_nodes": result.get("max_path_nodes"),
        "search_mode": "full_branch",
        "action_level": result.get("action_level", "subtask"),
        "guided_by_success_only_path": True,
        "guided_action_count": len(preferred_actions),
        "output": str(full_graph) if full_graph is not None else None,
        "success_paths_output": str(success_paths),
        "atomic_manifest_output": str(atomic_manifest),
        "unique_states_output": str(unique_states),
        "atomic_expanded_output": str(atomic_expanded),
        "state_coverage_manifest_output": str(coverage_path) if coverage_path else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
