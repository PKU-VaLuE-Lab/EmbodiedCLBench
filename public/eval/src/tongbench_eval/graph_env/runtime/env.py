from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate_set
from tongbench_eval.graph_env.io.schema import TongSimNode, TongSimStepResult, TongSimTask


class TongSimGraphEnv:
    def __init__(self, task: TongSimTask, task_dir: Path, max_steps: int = 30) -> None:
        self.task = task
        self.task_dir = Path(task_dir)
        self.max_steps = max_steps
        self._current_node_id = ""
        self._steps_taken = 0
        self._done = False
        self.trace: list[dict[str, Any]] = []
        self.reset()

    def _resolve_initial_node_id(self) -> str:
        return self.task.initial_node_id()

    def _image_paths_for_node(self, node_id: str) -> list[str]:
        observation = self.task.observations.get(node_id)
        return observation.image_paths if observation else []

    def _goal_reached(self, node: TongSimNode) -> bool:
        return self.task.is_goal_node(node.id)

    def reset(self) -> dict[str, Any]:
        self._current_node_id = self._resolve_initial_node_id()
        self._steps_taken = 0
        self._done = self._goal_reached(self.current_node())
        self.trace = []
        return self.observe()

    def observe(self) -> dict[str, Any]:
        node = self.current_node()
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.description,
            "current_state_id": node.id,
            "current_depth": node.depth,
            "state_predicates": list(node.state),
            "image_paths": list(self._image_paths_for_node(node.id)),
            "candidate_actions": self.task.candidate_actions(node.id),
            "goal_state": list(self.task.goal_state),
            "done": self.is_done(),
        }

    def step(self, action_id: str) -> TongSimStepResult:
        from_node = self.current_node()
        outgoing_edges = self.task.outgoing_edges(from_node.id)
        matching_edges = [edge for edge in outgoing_edges if edge.action == action_id]
        valid = bool(matching_edges)
        to_node = self.task.node_by_id(matching_edges[0].to_node) if valid else from_node

        self._current_node_id = to_node.id
        self._steps_taken += 1
        reached_goal = self._goal_reached(to_node)
        self._done = reached_goal or self._steps_taken >= self.max_steps

        trace_entry = {
            "step_index": self._steps_taken - 1,
            "from_state": from_node.id,
            "from_depth": from_node.depth,
            "observation_image_paths": list(self._image_paths_for_node(from_node.id)),
            "candidate_actions": self.task.candidate_actions(from_node.id),
            "selected_action_id": action_id,
            "valid": valid,
            "to_state": to_node.id,
            "to_depth": to_node.depth,
            "reached_goal": reached_goal,
            "current_predicates": normalize_predicate_set(list(to_node.state)),
        }
        if not valid:
            trace_entry["event"] = "invalid_action"
        self.trace.append(trace_entry)

        return TongSimStepResult(
            step_index=trace_entry["step_index"],
            selected_action_id=action_id,
            valid=valid,
            from_state=from_node.id,
            to_state=to_node.id,
            from_depth=from_node.depth,
            to_depth=to_node.depth,
            reached_goal=reached_goal,
            done=self._done,
            observation_image_paths=trace_entry["observation_image_paths"],
            candidate_actions=trace_entry["candidate_actions"],
            current_predicates=list(to_node.state),
        )

    def is_done(self) -> bool:
        return self._done

    def current_node(self) -> TongSimNode:
        return self.task.node_by_id(self._current_node_id)

    def save_trace(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = output_dir / "trace.json"
        trace_path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
