from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import AgentDecision, VLMBackend
from .prompt_builder import build_action_selection_prompt
from ...utils.python_utils import current_python


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_env_command(workspace_dir: Path, *args: str) -> dict[str, Any]:
    command = [current_python(), "tongsim_env.py", *args]
    try:
        result = subprocess.run(
            command,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        details: list[str] = [
            f"TongSIM workspace env command failed with exit code {exc.returncode}.",
            f"workspace_dir: {workspace_dir}",
            f"command: {command}",
        ]
        if stdout:
            details.extend(["stdout:", stdout])
        if stderr:
            details.extend(["stderr:", stderr])
        raise RuntimeError("\n".join(details)) from exc
    return json.loads(result.stdout)


def _task_json_path(workspace_dir: Path) -> Path:
    for name in ("task.json", "graph.json"):
        path = workspace_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No task.json or graph.json found in {workspace_dir}")


def _load_task(workspace_dir: Path) -> dict[str, Any]:
    with _task_json_path(workspace_dir).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_prompt_log(
    path: Path,
    step_index: int,
    observation: dict[str, Any],
    prompt: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== STEP {step_index} =====\n")
        handle.write(f"task_id: {observation.get('task_id', '')}\n")
        handle.write(f"current_state_id: {observation.get('current_state_id', '')}\n")
        handle.write(f"current_depth: {observation.get('current_depth', 0)}\n")
        handle.write(f"action_interface: {observation.get('action_interface', 'full_action')}\n")
        if observation.get("action_library_mode") is not None:
            handle.write(f"action_library_mode: {observation.get('action_library_mode')}\n")
        handle.write("prompt:\n")
        handle.write(prompt.rstrip())
        handle.write("\n\n")


def _decision_summary(
    backend: VLMBackend,
    task: dict[str, Any],
    decision_entries: list[dict[str, Any]],
    final_state: dict[str, Any],
    failure_type_override: str | None = None,
) -> dict[str, Any]:
    def _env_result(entry: dict[str, Any]) -> dict[str, Any]:
        return entry.get("env_step_result", {})

    def _env_valid(entry: dict[str, Any]) -> bool:
        return bool(_env_result(entry).get("valid", False))

    def _transition_or_execution_status(entry: dict[str, Any]) -> str | None:
        result = _env_result(entry)
        return result.get("transition_status") or result.get("execution_status")

    parse_error_steps = [entry["step_index"] for entry in decision_entries if entry.get("parse_error")]
    invalid_steps = [entry["step_index"] for entry in decision_entries if not entry.get("selected_action_valid", False)]
    not_executable_steps = [
        entry["step_index"]
        for entry in decision_entries
        if _transition_or_execution_status(entry) == "not_executable_by_state"
    ]
    executable_no_transition_steps = [
        entry["step_index"]
        for entry in decision_entries
        if _transition_or_execution_status(entry) == "executable_but_no_transition"
    ]
    matched_steps = [
        entry["step_index"]
        for entry in decision_entries
        if _transition_or_execution_status(entry) == "matched_transition"
    ]
    no_progress_steps = [
        entry["step_index"]
        for entry in decision_entries
        if _transition_or_execution_status(entry) == "state_unchanged_no_progress"
    ]
    return {
        "task_id": str(task.get("task_id", workspace_dir_name(task))),
        "model_backend": backend.name,
        "action_interface": decision_entries[0].get("action_interface") if decision_entries else "full_action",
        "action_library_mode": decision_entries[0].get("action_library_mode") if decision_entries else None,
        "total_steps": len(decision_entries),
        "valid_decisions": sum(1 for entry in decision_entries if _env_valid(entry)),
        "invalid_decisions": sum(1 for entry in decision_entries if not _env_valid(entry)),
        "parse_errors": len(parse_error_steps),
        "final_done": bool(final_state.get("done", False)),
        "final_state_id": str(final_state.get("current_state_id", "")),
        "model_selected_invalid_action_count": len(invalid_steps),
        "parse_error_count": len(parse_error_steps),
        "first_parse_error_step": parse_error_steps[0] if parse_error_steps else None,
        "first_invalid_selection_step": invalid_steps[0] if invalid_steps else None,
        "invalid_selected_action_ids": [
            entry.get("parsed_decision", {}).get("selected_action_id")
            or json.dumps(
                {
                    "action_level": entry.get("parsed_decision", {}).get("action_level"),
                    "template_id": entry.get("parsed_decision", {}).get("template_id"),
                    "action_type": entry.get("parsed_decision", {}).get("action_type"),
                    "object_id": entry.get("parsed_decision", {}).get("object_id"),
                    "target_id": entry.get("parsed_decision", {}).get("target_id"),
                    "role_bindings": entry.get("parsed_decision", {}).get("role_bindings", {}),
                },
                ensure_ascii=False,
            )
            for entry in decision_entries
            if not entry.get("selected_action_valid", False)
        ],
        "not_executable_by_state_count": len(not_executable_steps),
        "executable_but_no_transition_count": len(executable_no_transition_steps),
        "matched_transition_count": len(matched_steps),
        "no_progress_action_count": len(no_progress_steps),
        "first_not_executable_step": not_executable_steps[0] if not_executable_steps else None,
        "first_executable_but_no_transition_step": executable_no_transition_steps[0] if executable_no_transition_steps else None,
        "template_executable_count": sum(
            1
            for entry in decision_entries
            if _env_result(entry).get("executable_by_state") is True
        ),
        "template_not_executable_count": sum(
            1
            for entry in decision_entries
            if _env_result(entry).get("executable_by_state") is False
        ),
        "atomic_action_count": sum(1 for entry in decision_entries if entry.get("parsed_decision", {}).get("action_level") == "atomic"),
        "subtask_action_count": sum(1 for entry in decision_entries if entry.get("parsed_decision", {}).get("action_level") == "subtask"),
        "virtual_state_count": sum(1 for entry in decision_entries if _env_result(entry).get("is_virtual_state")),
        "rendered_observation_coverage": (
            sum(1 for entry in decision_entries if entry.get("image_paths")) / max(1, len(decision_entries))
        ),
        "dag_alignment_match_count": sum(
            1 for entry in decision_entries
            if _env_result(entry).get("dag_alignment", {}).get("status") == "matched_reference_edge"
        ),
        "dag_alignment_miss_count": sum(
            1 for entry in decision_entries
            if _env_result(entry).get("dag_alignment", {}).get("status") == "no_matching_reference_edge"
        ),
        "model_decision_failure_type": failure_type_override or "none",
        "final_state_predicates": list(final_state.get("current_state_predicates", [])),
    }


def workspace_dir_name(task: dict[str, Any]) -> str:
    return str(task.get("task_id", "unknown_task"))


def run_vlm_agent_loop(
    workspace_dir: Path,
    backend: VLMBackend,
    max_steps: int = 30,
    decision_log_path: Path | None = None,
    prompt_log_path: Path | None = None,
    action_interface: str = "full_action",
    action_library_mode: str = "atomic_only",
    atomic_template_path: Path | None = None,
    subtask_template_path: Path | None = None,
    debug_expose_valid_bindings: bool = False,
    max_invalid_repeats: int = 5,
    max_total_invalid_actions: int = 15,
    include_state_predicates_in_prompt: bool = True,
    include_goal_state_in_prompt: bool = True,
) -> dict[str, Any]:
    workspace_dir = Path(workspace_dir)
    decision_log_path = decision_log_path or (workspace_dir / "decision_log.jsonl")
    prompt_log_path = prompt_log_path or (workspace_dir / "prompt_log.txt")
    if decision_log_path.exists():
        decision_log_path.unlink()
    if prompt_log_path.exists():
        prompt_log_path.unlink()

    task = _load_task(workspace_dir)
    decision_entries: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None
    total_invalid_actions = 0
    invalid_repeat_count = 0
    previous_invalid_signature: str | None = None
    failure_type_override: str | None = None

    for step_index in range(max_steps):
        observe_args = ["observe", "--action-interface", action_interface]
        if action_interface == "library_factorized":
            observe_args.extend(["--action-library-mode", action_library_mode])
            if atomic_template_path is not None:
                observe_args.extend(["--atomic-template-path", str(atomic_template_path)])
            if subtask_template_path is not None:
                observe_args.extend(["--subtask-template-path", str(subtask_template_path)])
        if debug_expose_valid_bindings:
            observe_args.append("--debug-expose-valid-bindings")
        observation = _run_env_command(workspace_dir, *observe_args)
        observation["task_id"] = str(task.get("task_id", workspace_dir.name))
        observation["success_traces"] = task.get("success_traces", [])
        observation["goal_paths"] = task.get("goal_paths", [])
        observation["action_interface"] = action_interface
        previous_invalid_action_report = observation.get("previous_invalid_action_report")
        previous_action_report = observation.get("previous_action_report")
        if observation.get("done"):
            break

        image_paths = [Path(path) for path in observation.get("image_paths", [])]
        prompt = build_action_selection_prompt(
            observation,
            image_paths,
            action_interface=action_interface,
            include_state_predicates=include_state_predicates_in_prompt,
            include_goal_state=include_goal_state_in_prompt,
        )
        _append_prompt_log(prompt_log_path, step_index, observation, prompt)
        decision = backend.choose_action(observation, prompt, image_paths)
        if action_interface == "library_factorized":
            action_candidates = observation.get("model_facing_action_candidates", [])
            object_choice_ids = {item["object_id"] for item in observation.get("model_facing_object_candidates", [])}
            selected_action_valid = decision.parse_error is None
            if selected_action_valid:
                chosen_candidate = next(
                    (
                        candidate for candidate in action_candidates
                        if str(candidate.get("action_level", "")) == str(decision.action_level or "")
                        and str(candidate.get("template_id", "")) == str(decision.template_id or "")
                        and str(candidate.get("action_type", "")) == str(decision.action_type or "")
                    ),
                    None,
                )
                selected_action_valid = chosen_candidate is not None
                if selected_action_valid:
                    for role in chosen_candidate.get("required_roles", []):
                        if decision.role_bindings.get(str(role)) not in object_choice_ids:
                            selected_action_valid = False
                            break
            act_args = [
                "act",
                "--action-interface",
                "library_factorized",
                "--action-library-mode",
                action_library_mode,
                "--action_level",
                decision.action_level or "",
                "--template_id",
                decision.template_id or "",
                "--action_type",
                decision.action_type or "",
                "--role_bindings",
                json.dumps(decision.role_bindings, ensure_ascii=False),
            ]
            if decision.parse_error is not None:
                act_args.extend(["--parse-error", decision.parse_error])
                act_args.extend(["--raw-response", decision.raw_response])
                act_args.extend(["--parsed-json", json.dumps(decision.parsed_json or {}, ensure_ascii=False)])
                if decision.parse_error_details is not None:
                    act_args.extend(["--parse-error-details", json.dumps(decision.parse_error_details, ensure_ascii=False)])
            if atomic_template_path is not None:
                act_args.extend(["--atomic-template-path", str(atomic_template_path)])
            if subtask_template_path is not None:
                act_args.extend(["--subtask-template-path", str(subtask_template_path)])
            env_step_result = _run_env_command(workspace_dir, *act_args)
        elif action_interface in {"factorized", "global_factorized"}:
            allowed_action_types = {item["action_type"] for item in observation.get("action_types", [])}
            object_choice_ids = {item["object_id"] for item in observation.get("object_choices", [])}
            requirements = {
                item["action_type"]: list(item.get("required_args", []))
                for item in observation.get("action_types", [])
            }
            selected_action_valid = decision.parse_error is None and decision.action_type in allowed_action_types
            if selected_action_valid:
                required_args = requirements.get(decision.action_type or "", [])
                if "object_id" in required_args and decision.object_id not in object_choice_ids:
                    selected_action_valid = False
                if "target_id" in required_args and decision.target_id not in object_choice_ids:
                    selected_action_valid = False
            if selected_action_valid:
                act_args = ["act", "--action-interface", action_interface, "--action_type", decision.action_type or ""]
                if decision.object_id:
                    act_args.extend(["--object_id", decision.object_id])
                if decision.target_id:
                    act_args.extend(["--target_id", decision.target_id])
            else:
                act_args = ["act", "--action-interface", action_interface, "--action_type", "__PARSE_ERROR__"]
            env_step_result = _run_env_command(workspace_dir, *act_args)
        else:
            candidate_action_ids = [item["action_id"] for item in observation.get("candidate_actions", [])]
            selected_action_valid = (
                decision.selected_action_id is not None and decision.selected_action_id in candidate_action_ids
            )
            action_id = decision.selected_action_id if selected_action_valid else "__INVALID_MODEL_OUTPUT__"
            env_step_result = _run_env_command(workspace_dir, "act", "--action_id", action_id)

        invalid_action_repeated = False
        if action_interface in {"factorized", "global_factorized", "library_factorized"} and not env_step_result.get("valid", False):
            total_invalid_actions += 1
            current_invalid_signature = json.dumps(
                {
                    "action_level": decision.action_level,
                    "template_id": decision.template_id,
                    "action_type": decision.action_type,
                    "object_id": decision.object_id,
                    "target_id": decision.target_id,
                    "role_bindings": decision.role_bindings,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if current_invalid_signature == previous_invalid_signature:
                invalid_repeat_count += 1
                invalid_action_repeated = True
            else:
                invalid_repeat_count = 1
                previous_invalid_signature = current_invalid_signature
            if invalid_repeat_count > max_invalid_repeats:
                failure_type_override = "repeated_invalid_factorized_action"
            elif total_invalid_actions > max_total_invalid_actions:
                failure_type_override = "too_many_invalid_actions"
        else:
            invalid_repeat_count = 0
            if env_step_result.get("valid", False):
                previous_invalid_signature = None

        entry = {
            "step_index": step_index,
            "task_id": observation["task_id"],
            "current_state_id": observation.get("current_state_id", ""),
            "current_depth": observation.get("current_depth", 0),
            "image_paths": [str(path) for path in image_paths],
            "state_predicates": observation.get("state_predicates", []),
            "normalized_current_state_predicates": observation.get("normalized_current_state_predicates", observation.get("state_predicates", [])),
            "goal_state": observation.get("goal_state", []),
            "normalized_goal_state": observation.get("normalized_goal_state", observation.get("goal_state", [])),
            "action_interface": action_interface,
            "action_library_mode": action_library_mode if action_interface == "library_factorized" else None,
            "render_node_id": observation.get("render_node_id"),
            "previous_invalid_action_report": previous_invalid_action_report,
            "previous_action_report": previous_action_report,
            "candidate_actions": observation.get("candidate_actions", []),
            "action_types": observation.get("action_types", []),
            "object_choices": observation.get("object_choices", []),
            "model_facing_action_candidates": observation.get("model_facing_action_candidates", []),
            "model_facing_object_candidates": observation.get("model_facing_object_candidates", []),
            "observation_status": observation.get("observation_status"),
            "observation_warning": observation.get("observation_warning"),
            "model_backend": backend.name,
            "prompt": prompt,
            "raw_response": decision.raw_response,
            "model_selected_action_level": decision.action_level,
            "model_selected_template_id": decision.template_id,
            "model_selected_action_type": decision.action_type,
            "model_selected_object_id": decision.object_id,
            "model_selected_target_id": decision.target_id,
            "parsed_decision": {
                "selected_action_id": decision.selected_action_id,
                "action_level": decision.action_level,
                "template_id": decision.template_id,
                "action_type": decision.action_type,
                "object_id": decision.object_id,
                "target_id": decision.target_id,
                "role_bindings": decision.role_bindings,
                "parse_error_details": decision.parse_error_details,
                "confidence": decision.confidence,
                "brief_reason": decision.brief_reason,
                "perceived_state": decision.perceived_state,
                "perceived_state_predicates": decision.perceived_state_predicates,
                "uncertain_predicates": decision.uncertain_predicates,
                "visual_evidence": decision.visual_evidence,
                "rejected_actions": decision.rejected_actions,
                "rejected_options": decision.rejected_options,
            },
            "parse_error": decision.parse_error,
            "parse_error_details": decision.parse_error_details,
            "selected_action_valid": selected_action_valid,
            "env_step_result": env_step_result,
            "env_error_type": env_step_result.get("error_type") or env_step_result.get("diagnostic", {}).get("error_type"),
            "constraint_violations": env_step_result.get("constraint_violations", []),
            "precondition_violations": env_step_result.get("precondition_violations", []),
            "transition_status": env_step_result.get("transition_status"),
            "execution_status": env_step_result.get("execution_status"),
            "executable_by_state": env_step_result.get("executable_by_state"),
            "matched_transition": env_step_result.get("matched_transition"),
            "invalid_action_repeated": invalid_action_repeated,
            "invalid_repeat_count": invalid_repeat_count if not env_step_result.get("valid", False) else 0,
            "timestamp": _timestamp(),
        }
        decision_entries.append(entry)
        _append_jsonl(decision_log_path, entry)

        if env_step_result.get("done") or failure_type_override is not None:
            break

    final_state = _run_env_command(workspace_dir, "finish")
    summary = _decision_summary(
        backend,
        task,
        decision_entries,
        final_state,
        failure_type_override=failure_type_override,
    )
    summary_path = decision_log_path.with_name("decision_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "decision_log_path": str(decision_log_path),
        "prompt_log_path": str(prompt_log_path),
        "decision_summary_path": str(summary_path),
        "decision_summary": summary,
        "final_state": final_state,
    }
