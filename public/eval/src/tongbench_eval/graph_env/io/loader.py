from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from .predicate_normalizer import extract_predicate_list, normalize_predicate_set
from .schema import TongSimEdge, TongSimNode, TongSimObservation, TongSimTask


DESCRIPTION_FALLBACK_FIELDS = ("description_original", "description")


def _raw_description_value(raw: dict, field_name: str) -> str:
    value = raw.get(field_name)
    if value is None and isinstance(raw.get("task"), dict):
        value = raw["task"].get(field_name)
    if value is None:
        return ""
    return str(value).strip()


def select_task_description(raw: dict, description_field: str = "description") -> str:
    field_name = str(description_field or "description").strip() or "description"
    fields = [field_name]
    fields.extend(field for field in DESCRIPTION_FALLBACK_FIELDS if field not in fields)
    if "description_highlevel" not in fields:
        fields.append("description_highlevel")

    for field in fields:
        value = _raw_description_value(raw, field)
        if value:
            return value
    return ""


def _task_json_path(task_dir: Path) -> Path:
    task_dir = Path(task_dir)
    if task_dir.is_file():
        return task_dir
    for name in ("task.json", "graph.json"):
        path = task_dir / name
        if path.exists():
            return path
    candidates = sorted(task_dir.glob("*.json"))
    preferred = [path for path in candidates if "task" in path.stem.lower() or "graph" in path.stem.lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"No task.json/graph.json or unique task-like json found in {task_dir}")


def _success_paths_json_path(json_path: Path, raw: dict) -> Path | None:
    task_root = json_path.parent
    candidates: list[Path] = []
    task_id = str(raw.get("task_id", "")).strip()
    if task_id:
        candidates.append(task_root / f"{task_id}_success_paths.json")

    stem = json_path.stem
    suffixes = (
        "_atomic_expanded_graph",
        "_atomic_graph",
        "_expanded_graph",
        "_graph",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            candidates.append(task_root / f"{stem[:-len(suffix)]}_success_paths.json")
    candidates.append(task_root / f"{stem}_success_paths.json")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _extract_success_traces_from_success_paths(payload: object) -> list[list[str]]:
    if not isinstance(payload, dict):
        return []
    raw_paths = payload.get("success_paths", [])
    if not isinstance(raw_paths, list):
        return []

    shortest_paths = [
        item for item in raw_paths
        if isinstance(item, dict) and bool(item.get("is_shortest_success_path", False))
    ]
    candidate_paths = shortest_paths or [item for item in raw_paths if isinstance(item, dict)]
    traces: list[list[str]] = []
    for item in candidate_paths:
        trace: list[str] = []
        atomic_steps = item.get("atomic_steps", [])
        if isinstance(atomic_steps, list):
            for step in atomic_steps:
                if isinstance(step, dict):
                    action = str(step.get("atomic_action", "")).strip()
                    if action:
                        trace.append(action)
                elif isinstance(step, str) and step.strip():
                    trace.append(step.strip())
        if not trace:
            subtask_steps = item.get("subtask_steps", [])
            if isinstance(subtask_steps, list):
                for step in subtask_steps:
                    if isinstance(step, dict):
                        action = str(step.get("action", "")).strip()
                        if action:
                            trace.append(action)
                    elif isinstance(step, str) and step.strip():
                        trace.append(step.strip())
        if trace:
            traces.append(trace)
    return traces


def _load_success_traces_from_sidecar(json_path: Path, raw: dict) -> list[list[str]]:
    sidecar_path = _success_paths_json_path(json_path, raw)
    if sidecar_path is None:
        return []
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return _extract_success_traces_from_success_paths(payload)


def _collect_observation_paths(task_root: Path, state_id: str) -> list[str]:
    observations_dir = task_root / "observations"
    if not observations_dir.exists():
        return []

    paths: list[Path] = []
    direct_files = [
        observations_dir / f"{state_id}.png",
        observations_dir / f"{state_id}.jpg",
        observations_dir / f"{state_id}.jpeg",
    ]
    nested_files = [
        observations_dir / state_id / "front.png",
        observations_dir / state_id / "left.png",
        observations_dir / state_id / "right.png",
        observations_dir / state_id / "top.png",
    ]
    paths.extend(path for path in direct_files if path.exists())
    paths.extend(path for path in nested_files if path.exists())

    nested_dir = observations_dir / state_id
    if nested_dir.is_dir():
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(sorted(nested_dir.glob(pattern)))

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def _infer_missing_node_depths(
    nodes: list[TongSimNode],
    dag_edges: list[TongSimEdge],
    *,
    raw_nodes: list[dict],
    initial_state: list[str],
) -> None:
    if not nodes or not dag_edges:
        return
    if any("depth" in raw_node for raw_node in raw_nodes):
        return

    node_by_id = {node.id: node for node in nodes}
    adjacency: dict[str, list[str]] = {node.id: [] for node in nodes}
    indegree: dict[str, int] = {node.id: 0 for node in nodes}
    for edge in dag_edges:
        if edge.from_node not in node_by_id or edge.to_node not in node_by_id:
            continue
        adjacency[edge.from_node].append(edge.to_node)
        indegree[edge.to_node] += 1

    initial_ids = [
        str(raw_node.get("id", "")).strip()
        for raw_node in raw_nodes
        if bool(raw_node.get("is_initial_state", False))
    ]
    initial_ids = [node_id for node_id in initial_ids if node_id in node_by_id]
    if not initial_ids:
        initial_ids = [node.id for node in nodes if node.state == initial_state]
    if not initial_ids:
        initial_ids = [node.id for node in nodes if indegree.get(node.id, 0) == 0]
    if not initial_ids and nodes:
        initial_ids = [nodes[0].id]

    inferred_depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in initial_ids)
    while queue:
        node_id, depth = queue.popleft()
        previous_depth = inferred_depths.get(node_id)
        if previous_depth is not None and previous_depth <= depth:
            continue
        inferred_depths[node_id] = depth
        for successor in adjacency.get(node_id, []):
            queue.append((successor, depth + 1))

    for node in nodes:
        if node.id in inferred_depths:
            node.depth = inferred_depths[node.id]


def load_tongsim_task(task_dir: Path, *, description_field: str = "description") -> TongSimTask:
    task_source = Path(task_dir)
    json_path = _task_json_path(task_source)
    task_root = json_path.parent
    with json_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "goal_state" not in raw and "final_state" in raw:
        raw["goal_state"] = raw["final_state"]

    raw_nodes = raw.get("nodes", [])
    normalized_initial_state = normalize_predicate_set(extract_predicate_list(raw.get("initial_state")))
    nodes = [TongSimNode.from_raw(node) for node in raw_nodes]
    raw_edges = raw.get("dag_edges", raw.get("edges", []))
    dag_edges = [TongSimEdge.from_raw(edge) for edge in raw_edges]
    _infer_missing_node_depths(
        nodes,
        dag_edges,
        raw_nodes=raw_nodes,
        initial_state=normalized_initial_state,
    )
    observations = {
        node.id: TongSimObservation(
            state_id=node.id,
            image_paths=_collect_observation_paths(task_root, node.id),
        )
        for node in nodes
    }

    success_traces = raw.get("success_traces", [])
    if not success_traces:
        success_traces = _load_success_traces_from_sidecar(json_path, raw)

    return TongSimTask(
        task_id=str(raw.get("task_id", json_path.stem)),
        description=select_task_description(raw, description_field=description_field),
        initial_state=normalized_initial_state,
        goal_state=normalize_predicate_set(extract_predicate_list(raw.get("goal_state"))),
        nodes=nodes,
        dag_edges=dag_edges,
        goal_nodes=[str(item) for item in raw.get("goal_nodes", [])],
        goal_paths=raw.get("goal_paths", []),
        success_traces=success_traces,
        detailed_success_traces=raw.get("detailed_success_traces", []),
        observations=observations,
        raw=raw,
    )
