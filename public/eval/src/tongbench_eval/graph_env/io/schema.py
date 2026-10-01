from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .predicate_normalizer import normalize_predicate_set


def _normalize_predicates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return normalize_predicate_set([str(item) for item in value])
    return normalize_predicate_set([str(value)])


@dataclass(slots=True)
class TongSimEdge:
    from_node: str
    to_node: str
    action: str
    source: str = ""
    type: str = "transition"
    action_type: str = ""
    template_id: str = ""
    role_bindings: dict[str, str] = field(default_factory=dict)
    is_success_edge: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "TongSimEdge":
        action = str(
            raw.get("action")
            or raw.get("atomic_action")
            or raw.get("parent_action")
            or raw.get("action_id")
            or ""
        )
        return cls(
            from_node=str(raw.get("from", "")),
            to_node=str(raw.get("to", "")),
            action=action,
            source=str(raw.get("source", raw.get("parent_source", ""))),
            type=str(raw.get("type", "transition")),
            action_type=str(raw.get("action_type", "")),
            template_id=str(raw.get("template_id", "")),
            role_bindings={str(key): str(value) for key, value in dict(raw.get("role_bindings", {}) or {}).items()},
            is_success_edge=bool(raw.get("is_success_edge", False)),
        )

    def candidate_action(self) -> dict[str, str]:
        return {
            "action_id": self.action,
            "source": self.source,
            "to_state": self.to_node,
            "text": self.action,
            "type": self.type,
            "action_type": self.action_type,
            "template_id": self.template_id,
            "is_success_edge": str(bool(self.is_success_edge)).lower(),
        }


@dataclass(slots=True)
class TongSimNode:
    id: str
    depth: int
    state: list[str] = field(default_factory=list)
    is_goal: bool = False
    prefix_count: int = 0

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "TongSimNode":
        return cls(
            id=str(raw.get("id", "")),
            depth=int(raw.get("depth", 0)),
            state=_normalize_predicates(raw.get("state")),
            is_goal=bool(raw.get("is_goal", raw.get("is_goal_state", False))),
            prefix_count=int(raw.get("prefix_count", 0)),
        )


@dataclass(slots=True)
class TongSimObservation:
    state_id: str
    image_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TongSimStepResult:
    step_index: int
    selected_action_id: str
    valid: bool
    from_state: str
    to_state: str
    from_depth: int
    to_depth: int
    reached_goal: bool
    done: bool
    observation_image_paths: list[str]
    candidate_actions: list[dict[str, str]]
    current_predicates: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TongSimTask:
    task_id: str
    description: str
    initial_state: list[str]
    goal_state: list[str]
    nodes: list[TongSimNode]
    dag_edges: list[TongSimEdge]
    goal_nodes: list[str] = field(default_factory=list)
    goal_paths: list[Any] = field(default_factory=list)
    success_traces: list[Any] = field(default_factory=list)
    detailed_success_traces: list[Any] = field(default_factory=list)
    observations: dict[str, TongSimObservation] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def node_by_id(self, node_id: str) -> TongSimNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"Unknown node_id: {node_id}")

    def outgoing_edges(self, node_id: str) -> list[TongSimEdge]:
        return [edge for edge in self.dag_edges if edge.from_node == node_id]

    def candidate_actions(self, node_id: str) -> list[dict[str, str]]:
        return [edge.candidate_action() for edge in self.outgoing_edges(node_id)]

    def initial_node_id(self) -> str:
        for node in self.nodes:
            if node.state == self.initial_state:
                return node.id
        depth_zero_nodes = [node for node in self.nodes if node.depth == 0]
        if depth_zero_nodes:
            depth_zero_nodes.sort(key=lambda node: node.id)
            return depth_zero_nodes[0].id
        if self.nodes:
            return self.nodes[0].id
        raise ValueError("Task has no initial node")

    def is_goal_node(self, node_id: str) -> bool:
        node = self.node_by_id(node_id)
        return (
            node.id in self.goal_nodes
            or node.is_goal
            or (bool(self.goal_state) and set(self.goal_state).issubset(set(node.state)))
        )

    def shortest_success_trace_length(self) -> int:
        lengths: list[int] = []
        for trace in self.success_traces:
            if isinstance(trace, list) and trace:
                lengths.append(len(trace))
        for path in self.goal_paths:
            if isinstance(path, list) and path:
                lengths.append(len(path))
        if not lengths:
            raise ValueError("No success traces or goal paths available")
        return min(lengths)

    def max_goal_depth(self) -> int:
        goal_depths = [self.node_by_id(node_id).depth for node_id in self.goal_nodes]
        goal_depths.extend(node.depth for node in self.nodes if node.is_goal)
        goal_depths.extend(
            node.depth
            for node in self.nodes
            if self.goal_state and set(self.goal_state).issubset(set(node.state))
        )
        if not goal_depths:
            raise ValueError("No goal depth could be computed")
        return max(goal_depths)
