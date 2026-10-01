from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tongbench_eval.graph_env.grading.grader import grade_trace
from tongbench_eval.graph_env.interface.natural_language_interface import render_action_result_text, render_observation_text
from tongbench_eval.graph_env.io.loader import load_tongsim_task
from tongbench_eval.protocol.choice_interface import (
    build_choice_observation,
    load_choice_bank,
    normalize_choice_label,
    render_choice_action_result_text,
    render_choice_observation_text,
)
from tongbench_eval.utils.python_utils import current_python
from .runner import _canonical_trace_action_id, _score_details, prepare_tongsim_workspace


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CLIENT_SCRIPT = PACKAGE_ROOT / "mcp" / "client.py"
MCP_SERVER_SCRIPT = PACKAGE_ROOT / "mcp" / "server.py"


@dataclass(frozen=True)
class FrameworkSession:
    session_id: str
    task_dir: str
    task_id: str
    output_dir: str
    private_workspace_dir: str
    public_workspace_dir: str
    bridge_dir: str
    action_interface: str
    action_library_mode: str
    atomic_template_path: str | None
    subtask_template_path: str | None
    exclusive_in_view: bool
    max_steps: int
    public_interface_mode: str
    decision_only: bool = False
    generic_invalid_action_feedback: bool = False
    description_field: str = "description"
    protocol_surface: str = "legacy"
    choice_count: int = 4
    choice_seed: int = 0
    choice_bank_path: str | None = None
    write_human_trace_md: bool = False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _normalize_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    selected_action = payload.get("selected_action")
    if isinstance(selected_action, dict):
        action_fields = dict(selected_action)
    else:
        action_fields = dict(payload)
    result = {
        "action_id": action_fields.get("action_id"),
        "action_level": action_fields.get("action_level"),
        "template_id": action_fields.get("template_id"),
        "action_type": action_fields.get("action_type"),
        "object_id": action_fields.get("object_id"),
        "target_id": action_fields.get("target_id"),
        "role_bindings": action_fields.get("role_bindings"),
        "parse_error": payload.get("parse_error"),
        "raw_response": payload.get("raw_response"),
        "parsed_json": payload.get("parsed_json", payload),
        "parse_error_details": payload.get("parse_error_details"),
    }
    return result


def _decision_analysis_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for key in ("perceived_state_predicates", "uncertain_predicates", "visual_evidence"):
        if key in payload:
            analysis[key] = list(payload.get(key, []) or [])
    for key in (
        "brief_reason",
        "image_description",
        "decision_summary",
        "perceived_state",
    ):
        if key in payload:
            analysis[key] = payload.get(key)
    return analysis


def _summarize_feedback(
    report: dict[str, Any] | None,
    *,
    generic_invalid_action_feedback: bool = False,
) -> str | None:
    if not isinstance(report, dict) or not report:
        return None
    error_type = str(report.get("error_type", "") or "").strip()
    if error_type == "no_progress_action":
        return "The previous action was allowed but made no progress."
    if error_type == "parse_error":
        if generic_invalid_action_feedback:
            return "The selected action failed."
        return "The previous action was rejected because its template or object bindings were invalid. Re-check the exact template_id and object_id values."
    if error_type == "invalid_action":
        if generic_invalid_action_feedback:
            return "The selected action failed."
        missing = [str(item) for item in report.get("missing_preconditions", []) if str(item).strip()]
        if any(item.startswith("in_view(") for item in missing):
            return "The previous action could not be executed because the required object was not currently in view."
        if any(item.startswith("with_reach(") for item in missing):
            return "The previous action could not be executed because the required object was not currently reachable."
        if any(item.startswith("holding(") for item in missing):
            return "The previous action could not be executed because the required object was not being held."
        if "reason" in report:
            return str(report["reason"])
        return "The previous action could not be executed from the current situation."
    if error_type == "episode_already_done":
        return "The episode is already complete, so no more actions are needed."
    reason = str(report.get("reason", "") or "").strip()
    return reason or None


def _public_action_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_candidates = payload.get("model_facing_action_candidates")
    if isinstance(model_candidates, list) and model_candidates:
        normalized: list[dict[str, Any]] = []
        for item in model_candidates:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "action_level": item.get("action_level"),
                    "template_id": item.get("template_id"),
                    "action_type": item.get("action_type"),
                    "description": item.get("description"),
                    "required_roles": list(item.get("required_roles", [])),
                }
            )
        return normalized

    candidate_actions = payload.get("candidate_actions")
    if isinstance(candidate_actions, list):
        normalized = []
        for item in candidate_actions:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "action_id": item.get("action_id"),
                    "text": item.get("text"),
                    "type": item.get("type"),
                }
            )
        return normalized
    return []


def _public_object_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    model_objects = payload.get("model_facing_object_candidates")
    if isinstance(model_objects, list) and model_objects:
        normalized: list[dict[str, Any]] = []
        for item in model_objects:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if object_id:
                normalized.append({"object_id": object_id})
        return normalized

    object_choices = payload.get("object_choices")
    if isinstance(object_choices, list):
        normalized = []
        for item in object_choices:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("object_id", "")).strip()
            if object_id:
                normalized.append({"object_id": object_id})
        return normalized
    return []


def _public_observation_payload(
    payload: dict[str, Any],
    *,
    generic_invalid_action_feedback: bool = False,
) -> dict[str, Any]:
    return {
        "task_instruction": payload.get("task_instruction"),
        "image_paths": list(payload.get("image_paths", []) or []),
        "action_candidates": _public_action_candidates(payload),
        "object_candidates": _public_object_candidates(payload),
        "previous_action_feedback": _summarize_feedback(
            payload.get("previous_action_report") or payload.get("previous_invalid_action_report"),
            generic_invalid_action_feedback=generic_invalid_action_feedback,
        ),
        "done": bool(payload.get("done", False)),
    }


def _render_public_observation_payload(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if payload.get("protocol_surface") == "safe_choice":
        return {
            "observation_text": render_choice_observation_text(payload),
            "image_paths": list(payload.get("image_paths", []) or []),
            "done": bool(payload.get("done", False)),
            "choices": list(payload.get("choices", []) or []),
        }
    if mode != "natural_language":
        return payload
    return {
        "observation_text": render_observation_text(payload),
        "image_paths": list(payload.get("image_paths", []) or []),
        "done": bool(payload.get("done", False)),
    }


def _public_action_result_payload(
    payload: dict[str, Any],
    *,
    generic_invalid_action_feedback: bool = False,
) -> dict[str, Any]:
    feedback = _summarize_feedback(
        payload.get("diagnostic"),
        generic_invalid_action_feedback=generic_invalid_action_feedback,
    )
    if feedback is None:
        if bool(payload.get("done", False)) and bool(payload.get("reached_goal", False)):
            feedback = "Action accepted. The task is complete. Stop using environment tools."
        elif bool(payload.get("done", False)):
            feedback = "The episode ended without completing the task. Stop using environment tools."
        elif bool(payload.get("valid", False)):
            feedback = "Action accepted."
        else:
            feedback = "Action rejected."
    elif bool(payload.get("done", False)):
        feedback = f"{feedback} The episode has ended. Stop using environment tools."
    return {
        "accepted": bool(payload.get("valid", False)),
        "done": bool(payload.get("done", False)),
        "action_feedback": feedback,
    }


def _render_public_action_result_payload(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    if payload.get("protocol_surface") == "safe_choice":
        rendered = dict(payload)
        rendered.pop("protocol_surface", None)
        rendered["result_text"] = render_choice_action_result_text(rendered)
        return rendered
    if mode != "natural_language":
        return payload
    rendered = dict(payload)
    rendered["result_text"] = render_action_result_text(rendered)
    return rendered


def _did_environment_update(payload: dict[str, Any]) -> bool:
    if "state_changed" in payload:
        return bool(payload.get("state_changed", False))
    from_state = str(payload.get("from_state", "") or payload.get("current_state_id", "") or "")
    to_state = str(payload.get("to_state", "") or payload.get("next_state_id", "") or "")
    if from_state and to_state:
        return from_state != to_state
    return str(payload.get("execution_status", "") or "").strip() == "state_updated_by_template_effect"


class TongSimFrameworkRuntime:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.runtime_root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        *,
        task_dir: Path,
        output_dir: Path,
        action_interface: str,
        action_library_mode: str,
        atomic_template_path: Path | None,
        subtask_template_path: Path | None,
        exclusive_in_view: bool,
        max_steps: int,
        public_interface_mode: str = "structured",
        memory_file: Path | None = None,
        public_workspace_dir: Path | None = None,
        unique_state_image_dir: Path | None = None,
        decision_only: bool = False,
        generic_invalid_action_feedback: bool = False,
        description_field: str = "description",
        protocol_surface: str = "legacy",
        choice_count: int = 4,
        choice_seed: int = 0,
        choice_bank_path: Path | None = None,
        write_human_trace_md: bool = False,
    ) -> FrameworkSession:
        task = load_tongsim_task(task_dir, description_field=description_field)
        session_id = f"{task.task_id}_{uuid.uuid4().hex[:8]}"
        session_root = self.sessions_dir / session_id
        private_workspace_dir = session_root / "private_workspace"
        bridge_dir = session_root / "bridge"
        public_workspace_dir = Path(public_workspace_dir) if public_workspace_dir else Path(output_dir) / "agent_workspace"

        prepare_tongsim_workspace(
            task_dir,
            private_workspace_dir,
            memory_file=memory_file,
            unique_state_image_dir=unique_state_image_dir,
            description_field=description_field,
        )
        bridge_dir.mkdir(parents=True, exist_ok=True)
        (bridge_dir / "requests").mkdir(exist_ok=True)
        (bridge_dir / "responses").mkdir(exist_ok=True)
        (bridge_dir / "current_observation").mkdir(exist_ok=True)
        self._prepare_public_workspace(
            public_workspace_dir,
            session_id,
            decision_only=decision_only,
            generic_invalid_action_feedback=generic_invalid_action_feedback,
            protocol_surface=protocol_surface,
        )

        session = FrameworkSession(
            session_id=session_id,
            task_dir=str(Path(task_dir).resolve()),
            task_id=task.task_id,
            output_dir=str(Path(output_dir).resolve()),
            private_workspace_dir=str(private_workspace_dir.resolve()),
            public_workspace_dir=str(public_workspace_dir.resolve()),
            bridge_dir=str(bridge_dir.resolve()),
            action_interface=action_interface,
            action_library_mode=action_library_mode,
            atomic_template_path=str(atomic_template_path.resolve()) if atomic_template_path else None,
            subtask_template_path=str(subtask_template_path.resolve()) if subtask_template_path else None,
            exclusive_in_view=exclusive_in_view,
            max_steps=max_steps,
            public_interface_mode=public_interface_mode,
            decision_only=decision_only,
            generic_invalid_action_feedback=generic_invalid_action_feedback,
            description_field=description_field,
            protocol_surface=str(protocol_surface or "legacy"),
            choice_count=int(choice_count or 4),
            choice_seed=int(choice_seed or 0),
            choice_bank_path=str(choice_bank_path.resolve()) if choice_bank_path else None,
            write_human_trace_md=bool(write_human_trace_md),
        )
        _write_json(session_root / "session.json", asdict(session))
        self._write_runtime_meta(
            session_id,
            {
                "observation_version": 0,
                "fresh_observation_available": False,
                "last_done": False,
                "last_reached_goal": False,
                "finalized": False,
            },
        )
        return session

    def load_session(self, session_id: str) -> FrameworkSession:
        session_path = self.sessions_dir / session_id / "session.json"
        if not session_path.exists():
            raise FileNotFoundError(f"Unknown TongSIM framework session: {session_id}")
        return FrameworkSession(**_read_json(session_path))

    def _safe_choice_observation_payload(
        self,
        session: FrameworkSession,
        raw_payload: dict[str, Any],
        *,
        observation_version: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        task = load_tongsim_task(Path(session.task_dir), description_field=session.description_field)
        choice_state = build_choice_observation(
            task=task,
            observation_payload=raw_payload,
            choice_count=session.choice_count,
            seed=session.choice_seed,
            observation_version=observation_version,
            choice_bank=load_choice_bank(session.choice_bank_path) if session.choice_bank_path else None,
            require_choice_bank=bool(session.choice_bank_path),
        )
        public_payload = {
            "protocol_surface": "safe_choice",
            "task_instruction": raw_payload.get("task_instruction"),
            "image_paths": list(raw_payload.get("image_paths", []) or []),
            "choices": list(choice_state.get("choices", []) or []),
            "previous_action_feedback": _summarize_feedback(
                raw_payload.get("previous_action_report") or raw_payload.get("previous_invalid_action_report"),
                generic_invalid_action_feedback=session.generic_invalid_action_feedback,
            ),
            "done": bool(raw_payload.get("done", False)),
        }
        return public_payload, choice_state

    def _append_human_trace_event(self, session: FrameworkSession, payload: dict[str, Any]) -> None:
        if not bool(session.write_human_trace_md):
            return
        _append_jsonl(
            Path(session.private_workspace_dir) / "human_trace_events.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
        )

    def observe(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        payload = self._run_env_command(session, "observe")
        raw_image_paths = list(payload.get("image_paths", []) or [])
        payload["image_paths"] = self._stage_observation_images(session, payload.get("image_paths", []))
        meta = self._read_runtime_meta(session_id)
        next_observation_version = int(meta.get("observation_version", 0)) + 1
        if session.protocol_surface == "safe_choice":
            public_observation, choice_state = self._safe_choice_observation_payload(
                session,
                payload,
                observation_version=next_observation_version,
            )
            public_payload = _render_public_observation_payload(
                public_observation,
                session.public_interface_mode,
            )
            choice_meta = {"choice_observation": choice_state}
            self._append_human_trace_event(
                session,
                {
                    "event": "observe",
                    "observation_version": next_observation_version,
                    "task_instruction": public_observation.get("task_instruction"),
                    "observation_text": public_payload.get("observation_text"),
                    "previous_action_feedback": public_observation.get("previous_action_feedback"),
                    "image_paths": list(public_observation.get("image_paths", []) or []),
                    "host_image_paths": raw_image_paths,
                    "choices": list(choice_state.get("choices", []) or []),
                    "hidden_choices": dict(choice_state.get("hidden_choices", {}) or {}),
                    "done": bool(public_observation.get("done", False)),
                },
            )
        else:
            public_payload = _render_public_observation_payload(
                _public_observation_payload(
                    payload,
                    generic_invalid_action_feedback=session.generic_invalid_action_feedback,
                ),
                session.public_interface_mode,
            )
            choice_meta = {}
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                **choice_meta,
                "observation_version": next_observation_version,
                "fresh_observation_available": True,
                "last_done": bool(payload.get("done", False)),
                "last_reached_goal": bool(payload.get("reached_goal", False)),
            },
        )
        self._auto_finalize_if_needed(session, session_id)
        return public_payload

    def act(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        session = self.load_session(session_id)
        meta = self._read_runtime_meta(session_id)
        if bool(meta.get("last_done", False)):
            return {
                "accepted": False,
                "done": True,
                "action_feedback": "The episode is already complete. Stop using environment tools.",
            }
        if not bool(meta.get("fresh_observation_available", False)):
            return {
                "accepted": False,
                "done": False,
                "action_feedback": "You must call observe to get the current observation before the first action. Call observe now.",
            }
        if session.protocol_surface == "safe_choice":
            return self._act_safe_choice(session, session_id, payload, meta)
        normalized = _normalize_action_payload(payload)
        raw_result = self._run_env_command(session, "act", normalized)
        public_payload = _public_action_result_payload(
            raw_result,
            generic_invalid_action_feedback=session.generic_invalid_action_feedback,
        )
        auto_observation: dict[str, Any] | None = None
        next_observation_version = int(meta.get("observation_version", 0))
        keep_current_observation = bool(meta.get("fresh_observation_available", False))
        if (
            bool(raw_result.get("valid", False))
            and _did_environment_update(raw_result)
            and not bool(public_payload.get("done", False))
        ):
            refreshed = self._run_env_command(session, "observe")
            raw_refreshed_image_paths = list(refreshed.get("image_paths", []) or [])
            refreshed["image_paths"] = self._stage_observation_images(session, refreshed.get("image_paths", []))
            auto_observation = _render_public_observation_payload(
                _public_observation_payload(
                    refreshed,
                    generic_invalid_action_feedback=session.generic_invalid_action_feedback,
                ),
                session.public_interface_mode,
            )
            next_observation_version += 1
            keep_current_observation = True
            public_payload["next_observation"] = auto_observation
            public_payload["action_feedback"] = f"{public_payload['action_feedback']} Observation refreshed automatically."
        public_payload = _render_public_action_result_payload(public_payload, session.public_interface_mode)
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                "observation_version": next_observation_version,
                "fresh_observation_available": keep_current_observation and not bool(public_payload.get("done", False)),
                "last_done": bool(
                    (auto_observation or {}).get("done", public_payload.get("done", False))
                ),
                "last_reached_goal": bool(
                    raw_result.get("reached_goal", False)
                ),
            },
        )
        self._append_framework_decision_log(
            session,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "observation_version": int(meta.get("observation_version", 0)),
                "selected_action": {
                    "action_id": normalized.get("action_id"),
                    "action_level": normalized.get("action_level"),
                    "template_id": normalized.get("template_id"),
                    "action_type": normalized.get("action_type"),
                    "object_id": normalized.get("object_id"),
                    "target_id": normalized.get("target_id"),
                    "role_bindings": dict(normalized.get("role_bindings") or {}),
                },
                "decision_analysis": _decision_analysis_from_payload(payload),
                "result": dict(public_payload),
            },
        )
        self._auto_finalize_if_needed(session, session_id)
        return public_payload

    def _act_safe_choice(
        self,
        session: FrameworkSession,
        session_id: str,
        payload: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        label = normalize_choice_label(payload.get("option_id") or payload.get("choice") or payload.get("label"))
        choice_observation = meta.get("choice_observation") if isinstance(meta.get("choice_observation"), dict) else {}
        hidden_choices = choice_observation.get("hidden_choices") if isinstance(choice_observation, dict) else {}
        selected_choice = dict(hidden_choices.get(label, {}) or {}) if isinstance(hidden_choices, dict) else {}
        if not label or not selected_choice:
            return _render_public_action_result_payload(
                {
                    "protocol_surface": "safe_choice",
                    "accepted": False,
                    "done": False,
                    "selected_choice": label,
                    "action_feedback": "Choose one of the listed option letters from the latest observation.",
                },
                session.public_interface_mode,
            )

        hidden_action_id = str(selected_choice.get("action_id") or f"__INVALID_CHOICE_{label}__")
        raw_result = self._run_env_command(
            session,
            "act",
            {
                "action_id": hidden_action_id,
                "parsed_json": {
                    "selected_choice": label,
                    "selected_choice_text": selected_choice.get("text"),
                    "choice_kind": selected_choice.get("kind"),
                    "brief_reason": payload.get("brief_reason", ""),
                },
            },
        )
        public_payload = _public_action_result_payload(
            raw_result,
            generic_invalid_action_feedback=session.generic_invalid_action_feedback,
        )
        public_payload["protocol_surface"] = "safe_choice"
        public_payload["selected_choice"] = label
        public_payload["selected_choice_text"] = selected_choice.get("text")
        auto_observation: dict[str, Any] | None = None
        next_observation_version = int(meta.get("observation_version", 0))
        keep_current_observation = bool(meta.get("fresh_observation_available", False))
        next_choice_meta = {"choice_observation": choice_observation}
        if (
            bool(raw_result.get("valid", False))
            and _did_environment_update(raw_result)
            and not bool(public_payload.get("done", False))
        ):
            refreshed = self._run_env_command(session, "observe")
            raw_refreshed_image_paths = list(refreshed.get("image_paths", []) or [])
            refreshed["image_paths"] = self._stage_observation_images(session, refreshed.get("image_paths", []))
            next_observation_version += 1
            public_observation, new_choice_state = self._safe_choice_observation_payload(
                session,
                refreshed,
                observation_version=next_observation_version,
            )
            auto_observation = _render_public_observation_payload(
                public_observation,
                session.public_interface_mode,
            )
            next_choice_meta = {"choice_observation": new_choice_state}
            keep_current_observation = True
            public_payload["next_observation"] = auto_observation
            public_payload["action_feedback"] = f"{public_payload['action_feedback']} Observation refreshed automatically."
            self._append_human_trace_event(
                session,
                {
                    "event": "observe",
                    "observation_version": next_observation_version,
                    "task_instruction": public_observation.get("task_instruction"),
                    "observation_text": auto_observation.get("observation_text"),
                    "previous_action_feedback": public_observation.get("previous_action_feedback"),
                    "image_paths": list(public_observation.get("image_paths", []) or []),
                    "host_image_paths": raw_refreshed_image_paths,
                    "choices": list(new_choice_state.get("choices", []) or []),
                    "hidden_choices": dict(new_choice_state.get("hidden_choices", {}) or {}),
                    "done": bool(public_observation.get("done", False)),
                },
            )
        public_payload = _render_public_action_result_payload(public_payload, session.public_interface_mode)
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                **next_choice_meta,
                "observation_version": next_observation_version,
                "fresh_observation_available": keep_current_observation and not bool(public_payload.get("done", False)),
                "last_done": bool((auto_observation or {}).get("done", public_payload.get("done", False))),
                "last_reached_goal": bool(raw_result.get("reached_goal", False)),
            },
        )
        self._append_framework_decision_log(
            session,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "observation_version": int(meta.get("observation_version", 0)),
                "selected_choice": label,
                "selected_choice_text": selected_choice.get("text"),
                "hidden_choice": selected_choice,
                "selected_action": {
                    "action_id": hidden_action_id,
                    "action_level": None,
                    "template_id": None,
                    "action_type": selected_choice.get("action_type"),
                    "object_id": selected_choice.get("object_id"),
                    "target_id": selected_choice.get("target_id"),
                    "role_bindings": {},
                },
                "decision_analysis": _decision_analysis_from_payload(payload),
                "result": dict(public_payload),
            },
        )
        self._append_human_trace_event(
            session,
            {
                "event": "select_option",
                "observation_version": int(meta.get("observation_version", 0)),
                "selected_choice": label,
                "selected_choice_text": selected_choice.get("text"),
                "hidden_choice": selected_choice,
                "result": dict(public_payload),
            },
        )
        self._auto_finalize_if_needed(session, session_id)
        return public_payload

    def status(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        payload = self._run_env_command(session, "status")
        meta = self._read_runtime_meta(session_id)
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                "last_done": bool(payload.get("done", False)),
                "last_reached_goal": bool(payload.get("reached_goal", False)),
            },
        )
        self._auto_finalize_if_needed(session, session_id)
        return {
            "done": bool(payload.get("done", False)),
            "step_count": int(payload.get("step_count", 0)),
        }

    def finish(self, session_id: str) -> dict[str, Any]:
        session = self.load_session(session_id)
        result = self._run_env_command(session, "finish")
        meta = self._read_runtime_meta(session_id)
        is_complete = bool(result.get("done", False) or result.get("reached_goal", False))
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                "finalized": is_complete,
                "last_done": bool(result.get("done", meta.get("last_done", False))),
                "last_reached_goal": bool(result.get("reached_goal", meta.get("last_reached_goal", False))),
            },
        )
        return result

    def export_outputs(self, session_id: str, output_dir: Path) -> dict[str, Any]:
        session = self.load_session(session_id)
        private_workspace = Path(session.private_workspace_dir)
        task_output_dir = Path(output_dir) / "task_output"
        task_output_dir.mkdir(parents=True, exist_ok=True)

        for relative in (
            "trace.jsonl",
            "env_state.json",
            "decision_log.jsonl",
            "decision_summary.json",
            "human_trace_events.jsonl",
            "prompt_log.txt",
            "results/final_state.json",
        ):
            source = private_workspace / relative
            if source.exists():
                target_name = "final_state.json" if source.name == "final_state.json" else source.name
                shutil.copy2(source, task_output_dir / target_name)

        trace = self._read_trace(task_output_dir / "trace.jsonl")
        task = load_tongsim_task(Path(session.task_dir), description_field=session.description_field)
        grade = grade_trace(task, self._normalize_trace_for_grader(trace))
        final_state = (
            _read_json(task_output_dir / "final_state.json")
            if (task_output_dir / "final_state.json").exists()
            else {}
        )
        score = {
            "SR": int(bool(grade.get("reached_goal", False))),
            "ER": int(grade.get("extra_steps", 0))
            if bool(grade.get("reached_goal", False))
            else None,
        }
        _write_json(task_output_dir / "grade.json", grade)
        _write_json(Path(output_dir) / "score.json", score)
        completed = bool(grade.get("reached_goal", False))
        outcome = {
            "schema_version": "tongbench_task_outcome_v1",
            "task_id": task.task_id,
            "completed": completed,
            "status": "completed" if completed else "not_completed",
            "message": (
                "Task outcome: completed successfully."
                if completed
                else "Task outcome: not completed."
            ),
        }
        _write_json(task_output_dir / "task_outcome.json", outcome)
        (task_output_dir / "task_outcome.md").write_text(
            f"# Task Outcome\n\n{outcome['message']}\n",
            encoding="utf-8",
        )
        if bool(session.write_human_trace_md):
            try:
                from tongbench_eval.reports.human_trace import write_human_trace_markdown

                write_human_trace_markdown(
                    output_dir=Path(output_dir),
                    task_id=task.task_id,
                    title=f"{task.task_id} safe-choice trace",
                )
            except Exception as exc:
                _write_json(
                    Path(output_dir) / "human_trace_error.json",
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
        return {"grade": grade, "score": score, "outcome": outcome}

    def _prepare_public_workspace(
        self,
        public_workspace_dir: Path,
        session_id: str,
        *,
        decision_only: bool = False,
        generic_invalid_action_feedback: bool = False,
        protocol_surface: str = "legacy",
    ) -> None:
        if public_workspace_dir.exists():
            shutil.rmtree(public_workspace_dir)
        public_workspace_dir.mkdir(parents=True, exist_ok=True)
        exec_dir = public_workspace_dir / "exec"
        exec_dir.mkdir(parents=True, exist_ok=True)
        for target_dir in (public_workspace_dir, exec_dir):
            shutil.copy2(CLIENT_SCRIPT, target_dir / "tongsim_client.py")
            shutil.copy2(MCP_SERVER_SCRIPT, target_dir / "tongsim_mcp_server.py")
            shutil.copy2(MCP_SERVER_SCRIPT.with_name("context_compact.py"), target_dir / "context_compact.py")
            (target_dir / "current_observation").mkdir(exist_ok=True)
            _write_json(
                target_dir / ".tongsim_session.json",
                {
                    "session_id": session_id,
                    "bridge_mount": "/tongsim_bridge",
                    "observation_subdir": "current_observation",
                    "decision_only": bool(decision_only),
                    "generic_invalid_action_feedback": bool(generic_invalid_action_feedback),
                    "protocol_surface": str(protocol_surface or "legacy"),
                },
            )

    def _session_root(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _runtime_meta_path(self, session_id: str) -> Path:
        return self._session_root(session_id) / "runtime_meta.json"

    def _framework_decision_log_path(self, session: FrameworkSession) -> Path:
        return Path(session.private_workspace_dir) / "decision_log.jsonl"

    def _read_runtime_meta(self, session_id: str) -> dict[str, Any]:
        path = self._runtime_meta_path(session_id)
        if not path.exists():
            return {
                "observation_version": 0,
                "fresh_observation_available": False,
                "last_done": False,
                "last_reached_goal": False,
                "finalized": False,
            }
        return _read_json(path)

    def _write_runtime_meta(self, session_id: str, payload: dict[str, Any]) -> None:
        _write_json(self._runtime_meta_path(session_id), payload)

    def _append_framework_decision_log(self, session: FrameworkSession, payload: dict[str, Any]) -> None:
        _append_jsonl(self._framework_decision_log_path(session), payload)

    def _auto_finalize_if_needed(self, session: FrameworkSession, session_id: str) -> None:
        meta = self._read_runtime_meta(session_id)
        if not bool(meta.get("last_done", False)) or bool(meta.get("finalized", False)):
            return
        self._run_env_command(session, "finish")
        self._write_runtime_meta(
            session_id,
            {
                **meta,
                "finalized": True,
            },
        )

    def _run_env_command(
        self,
        session: FrameworkSession,
        command: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        args = [
            current_python(),
            "tongsim_env.py",
            command,
        ]
        if command in {"observe", "act"}:
            args.extend(["--action-interface", session.action_interface])
            args.extend(["--action-library-mode", session.action_library_mode])
            args.extend(["--exclusive-in-view", "true" if session.exclusive_in_view else "false"])
            if session.atomic_template_path:
                args.extend(["--atomic-template-path", session.atomic_template_path])
            if session.subtask_template_path:
                args.extend(["--subtask-template-path", session.subtask_template_path])
        if command == "act" and payload:
            for key, flag in (
                ("action_id", "--action_id"),
                ("action_level", "--action_level"),
                ("template_id", "--template_id"),
                ("action_type", "--action_type"),
                ("object_id", "--object_id"),
                ("target_id", "--target_id"),
                ("parse_error", "--parse-error"),
                ("raw_response", "--raw-response"),
            ):
                value = payload.get(key)
                if value is not None:
                    args.extend([flag, str(value)])
            if payload.get("role_bindings") is not None:
                args.extend(["--role_bindings", json.dumps(payload["role_bindings"], ensure_ascii=False)])
            if payload.get("parsed_json") is not None:
                args.extend(["--parsed-json", json.dumps(payload["parsed_json"], ensure_ascii=False)])
            if payload.get("parse_error_details") is not None:
                args.extend(["--parse-error-details", json.dumps(payload["parse_error_details"], ensure_ascii=False)])

        completed = subprocess.run(
            args,
            cwd=Path(session.private_workspace_dir),
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "TONGSIM_MAX_STEPS": str(session.max_steps),
            },
        )
        if completed.returncode != 0:
            details = [
                f"TongSIM framework runtime command failed with exit code {completed.returncode}.",
                f"session_id: {session.session_id}",
                f"command: {args}",
            ]
            if completed.stdout:
                details.append("stdout:")
                details.append(completed.stdout.strip())
            if completed.stderr:
                details.append("stderr:")
                details.append(completed.stderr.strip())
            raise RuntimeError("\n".join(details))
        return json.loads(completed.stdout)

    def _stage_observation_images(self, session: FrameworkSession, image_paths: list[str]) -> list[str]:
        bridge_dir = Path(session.bridge_dir)
        bridge_observation_dir = bridge_dir / "current_observation"
        if bridge_observation_dir.exists():
            shutil.rmtree(bridge_observation_dir)
        bridge_observation_dir.mkdir(parents=True, exist_ok=True)

        rewritten: list[str] = []
        private_workspace = Path(session.private_workspace_dir)
        bridge_mount = self._public_bridge_mount(session)
        for index, raw_path in enumerate(image_paths):
            source = Path(raw_path)
            if not source.is_absolute():
                source = (private_workspace / raw_path).resolve()
            if not source.exists():
                continue
            destination = bridge_observation_dir / f"{index:02d}_{source.name}"
            shutil.copy2(source, destination)
            rewritten.append(f"{bridge_mount}/current_observation/{destination.name}")
        return rewritten

    @staticmethod
    def _public_bridge_mount(session: FrameworkSession) -> str:
        for config_path in (
            Path(session.public_workspace_dir) / ".tongsim_session.json",
            Path(session.public_workspace_dir) / "exec" / ".tongsim_session.json",
        ):
            if not config_path.exists():
                continue
            try:
                payload = _read_json(config_path)
            except Exception:
                continue
            bridge_mount = str(payload.get("bridge_mount", "") or "").strip()
            if bridge_mount:
                return bridge_mount.rstrip("/")
        return "/tongsim_bridge"

    @staticmethod
    def _read_trace(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    @staticmethod
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
