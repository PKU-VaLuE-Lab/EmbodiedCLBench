from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .env import TongSimGraphEnv
from tongbench_eval.graph_env.grading.grader import grade_trace
from tongbench_eval.graph_env.io.loader import load_tongsim_task
from tongbench_eval.graph_env.io.predicate_normalizer import normalize_predicate_set


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _success_paths_json_path(task_json_path: Path, task: Any | None = None) -> Path | None:
    raw = getattr(task, "raw", {}) if task is not None else {}
    task_id = str(raw.get("task_id", getattr(task, "task_id", "")) if raw or task is not None else "").strip()
    candidates: list[Path] = []
    if task_id:
        candidates.append(task_json_path.parent / f"{task_id}_success_paths.json")
    stem = task_json_path.stem
    for suffix in ("_atomic_expanded_graph", "_atomic_graph", "_expanded_graph", "_graph"):
        if stem.endswith(suffix):
            candidates.append(task_json_path.parent / f"{stem[:-len(suffix)]}_success_paths.json")
    candidates.append(task_json_path.parent / f"{stem}_success_paths.json")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def _workspace_task_payload(task: Any) -> dict[str, Any]:
    raw = getattr(task, "raw", {}) or {}
    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "description": task.description,
        "task_scope": list(raw.get("task_scope", [])),
        "initial_state": list(task.initial_state),
        "goal_state": list(task.goal_state),
        "nodes": [
            {
                "id": node.id,
                "depth": node.depth,
                "state": list(node.state),
                "is_goal": node.is_goal,
                "prefix_count": node.prefix_count,
            }
            for node in task.nodes
        ],
        "dag_edges": [
            {
                "from": edge.from_node,
                "to": edge.to_node,
                "action": edge.action,
                "source": edge.source,
                "type": edge.type,
                "action_type": getattr(edge, "action_type", ""),
                "template_id": getattr(edge, "template_id", ""),
                "role_bindings": dict(getattr(edge, "role_bindings", {}) or {}),
                "is_success_edge": bool(getattr(edge, "is_success_edge", False)),
            }
            for edge in task.dag_edges
        ],
        "goal_nodes": list(task.goal_nodes),
        "goal_paths": list(task.goal_paths),
        "success_traces": list(task.success_traces),
        "detailed_success_traces": [],
    }
    for key in ("output_language", "search_mode", "graph_mode", "is_dag", "preprocess"):
        if key in raw:
            payload[key] = raw[key]
    return payload


def _state_signature(predicates: list[Any]) -> tuple[str, ...]:
    return tuple(sorted(normalize_predicate_set([str(item) for item in predicates])))


def _image_files_in_dir(directory: Path) -> list[Path]:
    files: list[Path] = []
    if not directory.is_dir():
        return files
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(path for path in directory.glob(pattern) if path.is_file()))
    return files


def _unique_state_dir(unique_state_image_dir: Path, node_id: str, node_index: Any | None = None) -> Path | None:
    if node_index is not None:
        try:
            indexed = unique_state_image_dir / f"{int(node_index):03d}_{node_id}"
            if indexed.is_dir():
                return indexed
        except (TypeError, ValueError):
            pass
    direct = unique_state_image_dir / node_id
    if direct.is_dir():
        return direct
    matches = sorted(path for path in unique_state_image_dir.glob(f"*_{node_id}") if path.is_dir())
    return matches[0] if matches else None


def _unique_state_image_index(unique_state_image_dir: Path) -> dict[tuple[str, ...], list[Path]]:
    unique_state_image_dir = Path(unique_state_image_dir)
    index: dict[tuple[str, ...], list[Path]] = {}
    manifest_path = unique_state_image_dir / "node_manifest.json"

    manifest_nodes: list[dict[str, Any]] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            manifest_nodes = [item for item in raw_nodes if isinstance(item, dict)]

    if not manifest_nodes:
        for info_path in sorted(unique_state_image_dir.glob("*/node_info.json")):
            payload = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = dict(payload)
                payload.setdefault("_image_dir", str(info_path.parent))
                manifest_nodes.append(payload)

    for node in manifest_nodes:
        signature = _state_signature(list(node.get("state", []) or []))
        if not signature:
            continue
        raw_image_dir = str(node.get("_image_dir", "") or "").strip()
        image_dir = Path(raw_image_dir) if raw_image_dir else (
            _unique_state_dir(unique_state_image_dir, str(node.get("node_id", "")), node.get("node_index")) or Path()
        )
        image_files = _image_files_in_dir(image_dir)
        if image_files:
            index.setdefault(signature, []).extend(image_files)
    return index


def _copy_unique_state_observations(workspace_dir: Path, task: Any, unique_state_image_dir: Path | None) -> None:
    if unique_state_image_dir is None:
        return
    unique_state_image_dir = Path(unique_state_image_dir)
    if not unique_state_image_dir.exists():
        raise FileNotFoundError(f"Unique-state image directory does not exist: {unique_state_image_dir}")
    image_index = _unique_state_image_index(unique_state_image_dir)
    if not image_index:
        raise FileNotFoundError(f"No unique-state images found under: {unique_state_image_dir}")

    observations_dir = workspace_dir / "observations"
    observations_dir.mkdir(parents=True, exist_ok=True)
    for node in getattr(task, "nodes", []):
        image_files = image_index.get(_state_signature(list(node.state)))
        if not image_files:
            continue
        node_dir = observations_dir / node.id
        node_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(image_files):
            destination = node_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, destination)


def _copy_task_assets(
    task_dir: Path,
    workspace_dir: Path,
    task: Any | None = None,
    unique_state_image_dir: Path | None = None,
) -> None:
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    task_json_path = _task_json_path(task_dir)
    if task is None:
        shutil.copy2(task_json_path, workspace_dir / "task.json")
    else:
        _write_json(workspace_dir / "task.json", _workspace_task_payload(task))
    success_paths_src = _success_paths_json_path(task_json_path, task)
    if success_paths_src is not None:
        shutil.copy2(success_paths_src, workspace_dir / success_paths_src.name)
    observations_src = task_json_path.parent / "observations"
    if observations_src.exists():
        shutil.copytree(observations_src, workspace_dir / "observations", dirs_exist_ok=True)
    if task is not None:
        _copy_unique_state_observations(workspace_dir, task, unique_state_image_dir)


def _write_workspace_env_script(workspace_dir: Path) -> None:
    source_dir = Path(__file__).resolve().parents[1] / "workspace"
    shutil.copy2(source_dir / "env.py", workspace_dir / "tongsim_env.py")
    parts_src = source_dir / "parts"
    if parts_src.exists():
        parts_dst = workspace_dir / "parts"
        if parts_dst.exists():
            shutil.rmtree(parts_dst)
        shutil.copytree(parts_src, parts_dst)


def _instruction_text(task: Any) -> str:
    return "\n".join(
        [
            "# TongSIM Offline DAG Task",
            "",
            task.description,
            "",
            "You are controlling a local offline embodied DAG environment.",
            "",
            "At each step:",
            "1. Run:",
            "   python tongsim_env.py observe",
            "2. Inspect the image paths in image_paths.",
            "3. Choose exactly one action_id from candidate_actions.",
            "4. Run:",
            "   python tongsim_env.py act --action_id \"ACTION_ID\"",
            "5. Repeat until done=true.",
            "6. Run:",
            "   python tongsim_env.py finish",
            "",
            "Rules:",
            "- Only use action_id values from candidate_actions.",
            "- Do not invent free-form actions.",
            "- Do not assume unavailable actions.",
            "- Use skill_memory.md if present.",
        ]
    ) + "\n"


def _read_jsonl_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _canonical_trace_action_id(entry: dict[str, Any]) -> str:
    for key in ("selected_action_id", "canonical_action"):
        value = entry.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() != "none":
            return str(value).strip()

    template_id = str(entry.get("selected_template_id", "") or "").strip()
    role_bindings = entry.get("selected_role_bindings", {}) or {}
    if not template_id or "{" not in template_id or not isinstance(role_bindings, dict):
        return template_id

    placeholders = re.findall(r"\{([^{}]+)\}", template_id)
    instantiated = template_id
    binding_values = [str(value) for value in role_bindings.values() if str(value).strip()]
    for index, role in enumerate(placeholders, start=1):
        object_id = str(role_bindings.get(role, "")).strip()
        if not object_id:
            object_id = str(role_bindings.get(f"r{index}", "")).strip()
        if not object_id and index <= len(binding_values):
            object_id = binding_values[index - 1]
        if object_id:
            instantiated = instantiated.replace(f"{{{role}}}", object_id)
    return instantiated if "{" not in instantiated else template_id


def _normalize_trace_for_grader(trace_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in trace_entries:
        normalized.append(
            {
                "step_index": entry.get("step_index"),
                "observation_image_paths": entry.get("observation_image_paths", []),
                "from_state": entry.get("from_state", entry.get("current_state_id")),
                "from_depth": entry.get("from_depth", entry.get("current_depth", 0)),
                "candidate_actions": entry.get("candidate_actions", []),
                "selected_action_id": _canonical_trace_action_id(entry),
                "valid": entry.get("valid", False),
                "to_state": entry.get("to_state", entry.get("next_state_id", entry.get("current_state_id"))),
                "to_depth": entry.get("to_depth", entry.get("next_depth", entry.get("current_depth", 0))),
                "reached_goal": entry.get("reached_goal", False),
                "current_predicates": entry.get("current_predicates", entry.get("current_state_predicates", [])),
                "to_state_predicates": entry.get("to_state_predicates", entry.get("current_state_predicates", [])),
                "is_virtual_state": entry.get("is_virtual_state", False),
                "observation_status": entry.get("observation_status"),
                "dag_alignment": entry.get("dag_alignment", {}),
            }
        )
    return normalized


def _write_trace_jsonl(path: Path, trace_entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in trace_entries:
            handle.write(json.dumps(entry) + "\n")


def _oracle_trace(task_dir: Path, max_steps: int) -> list[dict[str, Any]]:
    task = load_tongsim_task(task_dir)
    env = TongSimGraphEnv(task=task, task_dir=task_dir, max_steps=max_steps)
    env.reset()
    for trace in task.success_traces:
        if isinstance(trace, list) and trace:
            for action in trace:
                env.step(str(action))
            return [
                {
                    "step_index": entry["step_index"],
                    "observation_image_paths": entry["observation_image_paths"],
                    "current_state_id": entry["from_state"],
                    "current_depth": entry["from_depth"],
                    "current_state_predicates": task.node_by_id(entry["from_state"]).state,
                    "candidate_actions": entry["candidate_actions"],
                    "selected_action_id": entry["selected_action_id"],
                    "valid": entry["valid"],
                    "next_state_id": entry["to_state"],
                    "next_depth": entry["to_depth"],
                    "reached_goal": entry["reached_goal"],
                    "from_state": entry["from_state"],
                    "from_depth": entry["from_depth"],
                    "to_state": entry["to_state"],
                    "to_depth": entry["to_depth"],
                    "current_predicates": entry["current_predicates"],
                }
                for entry in env.trace
            ]
    return []


def _ensure_workspace(
    task_dir: Path,
    workspace_dir: Path,
    task: Any,
    memory_file: Path | None,
    unique_state_image_dir: Path | None = None,
) -> None:
    _copy_task_assets(task_dir, workspace_dir, task=task, unique_state_image_dir=unique_state_image_dir)
    _write_workspace_env_script(workspace_dir)
    (workspace_dir / "INSTRUCTION.md").write_text(_instruction_text(task), encoding="utf-8")
    if memory_file and Path(memory_file).exists():
        shutil.copy2(memory_file, workspace_dir / "skill_memory.md")


def prepare_tongsim_workspace(
    task_dir: Path,
    workspace_dir: Path,
    memory_file: Path | None = None,
    unique_state_image_dir: Path | None = None,
    description_field: str = "description",
) -> Any:
    task = load_tongsim_task(task_dir, description_field=description_field)
    _ensure_workspace(Path(task_dir), Path(workspace_dir), task, memory_file, unique_state_image_dir=unique_state_image_dir)
    return task


def _final_state_from_trace(task: Any, trace: list[dict[str, Any]]) -> dict[str, Any]:
    if trace:
        final_state_id = str(trace[-1].get("next_state_id", trace[-1].get("to_state")))
        final_node = task.node_by_id(final_state_id)
        predicates = list(trace[-1].get("current_predicates", trace[-1].get("current_state_predicates", final_node.state)))
        step_count = len(trace)
        invalid_action_count = sum(0 if entry.get("valid") else 1 for entry in trace)
        reached_goal = bool(trace[-1].get("reached_goal", task.is_goal_node(final_state_id)))
    else:
        final_node = task.node_by_id(task.initial_node_id())
        final_state_id = final_node.id
        predicates = list(final_node.state)
        step_count = 0
        invalid_action_count = 0
        reached_goal = task.is_goal_node(final_state_id)
    return {
        "task_id": task.task_id,
        "current_state_id": final_state_id,
        "current_depth": final_node.depth,
        "reached_goal": reached_goal,
        "done": reached_goal,
        "step_count": step_count,
        "invalid_action_count": invalid_action_count,
        "current_state_predicates": predicates,
    }


def _copy_outputs_from_workspace(workspace_dir: Path, task_output_dir: Path) -> None:
    task_output_dir.mkdir(parents=True, exist_ok=True)
    for relative in ("trace.jsonl", "env_state.json", "results/final_state.json"):
        source = workspace_dir / relative
        if source.exists():
            target_name = source.name if source.name != "final_state.json" else "final_state.json"
            shutil.copy2(source, task_output_dir / target_name)


def _default_usage(duration: float, timed_out: bool, grade: dict[str, Any]) -> dict[str, Any]:
    return {
        "time_sec": duration,
        "timeout": timed_out,
        "valid_action_count": int(grade.get("valid_action_count", 0)),
        "invalid_action_count": int(grade.get("invalid_action_count", 0)),
    }


def _score_details(grade: dict[str, Any], final_state_id: str = "") -> dict[str, Any]:
    return {
        "success_score": grade.get("success_score", 0.0),
        "progress_score": grade.get("progress_score", 0.0),
        "efficiency_score": grade.get("efficiency_score", 0.0),
        "path_quality_score": grade.get("path_quality_score", 0.0),
        "node_progress": grade.get("node_progress", 0.0),
        "predicate_progress": grade.get("predicate_progress", 0.0),
        "invalid_penalty": grade.get("invalid_penalty", 0.0),
        "reached_goal": grade.get("reached_goal", False),
        "final_state_id": grade.get("final_state_id", final_state_id),
        "error_type": grade.get("error_type", "none"),
        "breakpoint_state_id": grade.get("breakpoint_state_id", ""),
        "breakpoint_depth": grade.get("breakpoint_depth", 0),
    }


def _score_payload(task: Any, grade: dict[str, Any]) -> dict[str, Any]:
    _ = task
    reached_goal = bool(grade.get("reached_goal", False))
    return {
        "SR": int(reached_goal),
        "ER": int(grade.get("extra_steps", 0)) if reached_goal else None,
    }


def create_skill_memory(
    previous_task_dir: Path,
    previous_output_dir: Path,
    existing_memory: str | None = None,
) -> str:
    task = load_tongsim_task(previous_task_dir)
    score_path = previous_output_dir / "score.json"
    grade_path = previous_output_dir / "task_output" / "grade.json"
    trace_path = previous_output_dir / "task_output" / "trace.jsonl"
    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.exists() else {}
    grade = json.loads(grade_path.read_text(encoding="utf-8")) if grade_path.exists() else {}
    trace = _read_jsonl_trace(trace_path)

    successful_actions = [
        entry.get("selected_action_id", "")
        for entry in trace
        if entry.get("valid") and entry.get("reached_goal")
    ]
    invalid_actions = [
        entry.get("selected_action_id", "")
        for entry in trace
        if not entry.get("valid")
    ]
    actual_actions = [entry.get("selected_action_id", "") for entry in trace if entry.get("valid")]

    pattern_rules: list[str] = []
    action_text = " ".join(actual_actions).lower()
    goal_text = " ".join(task.goal_state).lower()
    if "place(" in action_text:
        pattern_rules.append("- Acquire object before placing it on a target surface.")
    if "reach(" in action_text and "place(" in action_text:
        pattern_rules.append("- Reach the target surface before placing a held object.")
    if "clean(" in goal_text or "dirty(" in " ".join(task.initial_state).lower():
        pattern_rules.append("- Clean a dirty object before final placement when the goal requires a clean object.")
    if not pattern_rules:
        pattern_rules.append("- Follow candidate_actions strictly and prefer actions that increase depth toward goal nodes.")

    sections: list[str] = []
    if existing_memory:
        sections.append(existing_memory.strip())
    sections.extend(
        [
            "# Skill Memory" if not sections else "",
            f"## {task.task_id}",
            f"Score: {score.get('overall_score', 0.0):.2f}" if score else "Score: 0.00",
            f"Result: {'success' if grade.get('reached_goal') else 'failure'}",
            "",
            f"Task: {task.description}",
            "",
            "Successful action pattern:",
        ]
    )
    if actual_actions:
        for index, action in enumerate(actual_actions, start=1):
            sections.append(f"{index}. {action}")
    else:
        sections.append("1. No valid actions were completed.")
    sections.append("")
    if invalid_actions:
        sections.append("Invalid actions:")
        for action in invalid_actions:
            sections.append(f"- {action}")
        sections.append("")
    if not grade.get("reached_goal"):
        sections.append("Breakpoint diagnosis:")
        sections.append(f"- Error type: {grade.get('error_type', 'unknown')}")
        sections.append(f"- Breakpoint state: {grade.get('breakpoint_state_id', '')}")
        sections.append(f"- Expected next actions: {', '.join(grade.get('expected_next_actions', []))}")
        sections.append("")
    sections.append("Useful rules:")
    sections.extend(pattern_rules)
    sections.append("- Avoid selecting an action that is not in candidate_actions.")
    if any("place(" in action.lower() for action in actual_actions + invalid_actions):
        sections.append("- Do not place an object before reaching or preparing the target state if the graph suggests another prerequisite.")
    sections.append("")
    return "\n".join(section for section in sections if section != "").strip() + "\n"


def run_tongsim_task(
    task_dir: Path,
    output_dir: Path,
    agent_command: list[str] | str | None = None,
    workspace_dir: Path | None = None,
    max_steps: int = 30,
    timeout_sec: int = 600,
    memory_file: Path | None = None,
    unique_state_image_dir: Path | None = None,
) -> dict[str, Any]:
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_output_dir = output_dir / "task_output"
    workspace_dir = Path(workspace_dir) if workspace_dir is not None else output_dir / "workspace"
    task = load_tongsim_task(task_dir)
    start_time = time.time()
    timed_out = False
    _ensure_workspace(task_dir, workspace_dir, task, memory_file, unique_state_image_dir=unique_state_image_dir)

    if agent_command is None:
        trace_entries = _oracle_trace(task_dir, max_steps=max_steps)
        final_state = _final_state_from_trace(task, trace_entries)
        env_state = dict(final_state)
        env_state["max_steps"] = max_steps
        _write_trace_jsonl(task_output_dir / "trace.jsonl", trace_entries)
        _write_json(task_output_dir / "final_state.json", final_state)
        _write_json(task_output_dir / "env_state.json", env_state)
        _write_json(workspace_dir / "env_state.json", env_state)
        _write_json(workspace_dir / "results" / "final_state.json", final_state)
        _write_trace_jsonl(workspace_dir / "trace.jsonl", trace_entries)
        (output_dir / "agent.log").write_text("oracle rollout\n", encoding="utf-8")
    else:
        log_path = output_dir / "agent.log"
        env = os.environ.copy()
        env["TONGSIM_WORKSPACE_DIR"] = str(workspace_dir.resolve())
        env["TONGSIM_MAX_STEPS"] = str(max_steps)
        try:
            completed = subprocess.run(
                agent_command,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                shell=isinstance(agent_command, str),
            )
            log_text = ""
            if completed.stdout:
                log_text += completed.stdout
            if completed.stderr:
                if log_text:
                    log_text += "\n"
                log_text += completed.stderr
            log_path.write_text(log_text, encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            log_text = (exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")
            log_path.write_text(log_text + "\nTIMEOUT\n", encoding="utf-8")

        _copy_outputs_from_workspace(workspace_dir, task_output_dir)
        trace_entries = _read_jsonl_trace(workspace_dir / "trace.jsonl")
        final_state_path = workspace_dir / "results" / "final_state.json"
        if final_state_path.exists():
            shutil.copy2(final_state_path, task_output_dir / "final_state.json")
        else:
            final_state = _final_state_from_trace(task, trace_entries)
            _write_json(task_output_dir / "final_state.json", final_state)
        env_state_path = workspace_dir / "env_state.json"
        if env_state_path.exists():
            shutil.copy2(env_state_path, task_output_dir / "env_state.json")
    trace_entries = _read_jsonl_trace(task_output_dir / "trace.jsonl")
    normalized_trace = _normalize_trace_for_grader(trace_entries)
    if not (task_output_dir / "final_state.json").exists():
        _write_json(task_output_dir / "final_state.json", _final_state_from_trace(task, trace_entries))
    if not (task_output_dir / "env_state.json").exists():
        _write_json(task_output_dir / "env_state.json", _final_state_from_trace(task, trace_entries))

    grade = grade_trace(task, normalized_trace)
    _write_json(task_output_dir / "grade.json", grade)
    score = _score_payload(task, grade)
    _write_json(output_dir / "score.json", score)
    usage = _default_usage(time.time() - start_time, timed_out, grade)
    _write_json(output_dir / "usage.json", usage)
    return {
        "task_id": task.task_id,
        "task_dir": str(task_dir),
        "output_dir": str(output_dir),
        "score": score,
        "usage": usage,
        "grade": grade,
        "timed_out": timed_out,
    }
