from __future__ import annotations

import math
from typing import Any

from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate_set
from tongbench_eval.graph_env.io.schema import TongSimTask


def _normalize_action_paths(values: list[Any]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for value in values:
        if not isinstance(value, list):
            continue
        actions: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                actions.append(item)
            elif isinstance(item, dict) and "action" in item:
                action = str(item["action"]).strip()
                if action:
                    actions.append(action)
        if actions:
            normalized.append(actions)
    return normalized


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    dp = [0] * (len(right) + 1)
    for item in left:
        prev = 0
        for index, other in enumerate(right, start=1):
            current = dp[index]
            if item == other:
                dp[index] = prev + 1
            else:
                dp[index] = max(dp[index], dp[index - 1])
            prev = current
    return dp[-1]


def _longest_prefix_match(actual: list[str], reference: list[str]) -> int:
    count = 0
    for actual_item, reference_item in zip(actual, reference):
        if actual_item != reference_item:
            break
        count += 1
    return count


def _action_tokens(action: str) -> set[str]:
    tokens = action.replace("(", " ").replace(")", " ").replace(",", " ").split()
    return {token.lower() for token in tokens if token}


def _leading_verb(action: str) -> str:
    return action.split("(")[0].split()[0].lower() if action else ""


def _state_sequence(task: TongSimTask, trace: list[dict[str, Any]]) -> list[str]:
    states = [task.initial_node_id()]
    for step in trace:
        if step.get("valid"):
            states.append(str(step.get("to_state", states[-1])))
    return states


def _safe_node(task: TongSimTask, node_id: str) -> Any | None:
    try:
        return task.node_by_id(node_id)
    except KeyError:
        return None


def grade_trace(task: TongSimTask, trace: list[dict[str, Any]]) -> dict[str, Any]:
    trace = trace or []
    valid_steps = [step for step in trace if step.get("valid")]
    invalid_steps = [step for step in trace if not step.get("valid")]
    valid_action_ids = [str(step.get("selected_action_id", "")) for step in valid_steps]
    invalid_action_count = len(invalid_steps)
    valid_action_count = len(valid_steps)
    reference_paths = _normalize_action_paths(task.success_traces) or _normalize_action_paths(task.goal_paths)
    shortest_success_trace_length = min((len(path) for path in reference_paths), default=0)
    if shortest_success_trace_length <= 0:
        try:
            shortest_success_trace_length = task.shortest_success_trace_length()
        except ValueError:
            shortest_success_trace_length = 0

    final_state_id = (
        str(trace[-1].get("to_state"))
        if trace
        else task.initial_node_id()
    )
    final_node = _safe_node(task, final_state_id)
    final_predicates = normalize_predicate_set(
        list(
            trace[-1].get(
                "current_predicates",
                trace[-1].get("to_state_predicates", final_node.state if final_node else []),
            )
        )
    ) if trace else normalize_predicate_set(list(final_node.state if final_node else task.initial_state))
    normalized_goal_state = set(normalize_predicate_set(task.goal_state))
    reached_goal = (
        final_state_id in task.goal_nodes
        or (bool(final_node.is_goal) if final_node is not None else False)
        or normalized_goal_state.issubset(set(final_predicates))
    )

    best_reference = min(reference_paths, key=len) if reference_paths else []
    lcs = _lcs_length(valid_action_ids, best_reference)
    prefix_lengths = [_longest_prefix_match(valid_action_ids, path) for path in reference_paths] or [0]
    max_prefix_len = max(prefix_lengths)
    best_prefix_index = prefix_lengths.index(max_prefix_len) if reference_paths else 0
    prefix_reference = reference_paths[best_prefix_index] if reference_paths else []

    shortest_goal_depth = max(1, min(task.max_goal_depth(), shortest_success_trace_length or task.max_goal_depth()))
    deepest_reached_depth = max([task.node_by_id(task.initial_node_id()).depth] + [int(step.get("to_depth", 0)) for step in valid_steps])
    node_progress = min(1.0, deepest_reached_depth / max(1, shortest_goal_depth))
    prefix_progress = min(1.0, max_prefix_len / max(1, shortest_success_trace_length))
    achieved_goal_predicates = len(normalized_goal_state.intersection(set(final_predicates)))
    predicate_progress = (
        achieved_goal_predicates / max(1, len(normalized_goal_state))
        if normalized_goal_state
        else float(reached_goal)
    )
    node_progress_for_score = min(1.0, node_progress)
    progress_score = 1.0 if reached_goal else max(node_progress_for_score, prefix_progress, predicate_progress)

    repeated_state_count = max(0, len(_state_sequence(task, trace)) - len(set(_state_sequence(task, trace))))
    redundant_action_count = max(0, valid_action_count - lcs, repeated_state_count)
    actual_valid_step_count = max(1, valid_action_count)
    reference_step_count = max(1, shortest_success_trace_length)
    total_step_count = len(trace)
    extra_steps = max(0, total_step_count - max(1, shortest_success_trace_length))
    if reached_goal:
        efficiency_score = min(1.0, shortest_success_trace_length / actual_valid_step_count) if valid_action_count else 0.0
    else:
        efficiency_score = min(1.0, valid_action_count / reference_step_count) if valid_action_count else 0.0
    path_quality_score = (
        min(1.0, max(0.0, (lcs / actual_valid_step_count) - (repeated_state_count / max(1, actual_valid_step_count * 2))))
        if reached_goal
        else min(1.0, max(prefix_progress, lcs / max(1, len(best_reference))))
    )
    success_score = 1.0 if reached_goal else 0.0

    breakpoint_index = max_prefix_len
    valid_state_sequence = _state_sequence(task, trace)
    breakpoint_state_id = valid_state_sequence[min(breakpoint_index, len(valid_state_sequence) - 1)]
    breakpoint_node = _safe_node(task, breakpoint_state_id)
    expected_next_actions = [item["action_id"] for item in task.candidate_actions(breakpoint_state_id)] if breakpoint_node is not None else []

    actual_failed_action = ""
    if invalid_steps:
        actual_failed_action = str(invalid_steps[0].get("selected_action_id", ""))
    elif valid_action_count > max_prefix_len:
        actual_failed_action = valid_action_ids[max_prefix_len]

    completed_milestones = prefix_reference[:max_prefix_len]
    remaining_goal_predicates = [
        predicate for predicate in normalize_predicate_set(task.goal_state) if predicate not in set(final_predicates)
    ]

    all_task_actions = {edge.action for edge in task.dag_edges}
    error_type = "none"
    if not reached_goal:
        if actual_failed_action and actual_failed_action not in expected_next_actions and actual_failed_action not in all_task_actions:
            error_type = "invalid_action_error"
        elif actual_failed_action and actual_failed_action in all_task_actions and actual_failed_action not in expected_next_actions:
            error_type = "sequencing_error"
        elif actual_failed_action and expected_next_actions:
            actual_tokens = _action_tokens(actual_failed_action)
            expected_tokens = set().union(*[_action_tokens(action) for action in expected_next_actions])
            same_object = bool(actual_tokens.intersection(expected_tokens))
            if same_object and _leading_verb(actual_failed_action) not in {_leading_verb(action) for action in expected_next_actions}:
                error_type = "manipulation_error"
            elif invalid_action_count > 0:
                error_type = "perception_error"
            else:
                error_type = "exploration_or_planning_error"
        elif invalid_action_count > 0:
            error_type = "invalid_action_error"
        else:
            error_type = "exploration_or_planning_error"

    invalid_penalty = 0.2 * (1.0 - math.exp(-invalid_action_count / 10.0))

    if reached_goal:
        overall = (
            0.55 * success_score
            + 0.25 * efficiency_score
            + 0.20 * path_quality_score
            - invalid_penalty
        )
    else:
        overall = (
            0.25 * node_progress_for_score
            + 0.50 * predicate_progress
            + 0.25 * path_quality_score
            - invalid_penalty
        )

    return {
        "overall": max(0.0, min(1.0, overall)),
        "success_score": success_score,
        "progress_score": progress_score,
        "efficiency_score": efficiency_score,
        "path_quality_score": path_quality_score,
        "node_progress": node_progress,
        "node_progress_for_score": node_progress_for_score,
        "prefix_progress": prefix_progress,
        "predicate_progress": predicate_progress,
        "invalid_penalty": invalid_penalty,
        "final_state_id": final_state_id,
        "reached_goal": reached_goal,
        "valid_action_count": valid_action_count,
        "invalid_action_count": invalid_action_count,
        "total_step_count": total_step_count,
        "shortest_success_trace_length": shortest_success_trace_length,
        "extra_steps": extra_steps,
        "redundant_action_count": redundant_action_count,
        "breakpoint_state_id": breakpoint_state_id,
        "breakpoint_depth": breakpoint_node.depth if breakpoint_node is not None else int(trace[-1].get("to_depth", 0)) if trace else 0,
        "expected_next_actions": expected_next_actions,
        "actual_failed_action": actual_failed_action,
        "completed_milestones": completed_milestones,
        "remaining_goal_predicates": remaining_goal_predicates,
        "error_type": error_type,
        "virtual_state_count": sum(1 for step in trace if step.get("is_virtual_state")),
        "rendered_observation_coverage": (
            sum(1 for step in trace if step.get("observation_status") == "rendered_node_observation") / max(1, len(trace))
            if trace else 1.0
        ),
        "dag_alignment_match_count": sum(
            1 for step in trace
            if step.get("dag_alignment", {}).get("status") == "matched_reference_edge"
        ),
        "dag_alignment_miss_count": sum(
            1 for step in trace
            if step.get("dag_alignment", {}).get("status") == "no_matching_reference_edge"
        ),
    }
