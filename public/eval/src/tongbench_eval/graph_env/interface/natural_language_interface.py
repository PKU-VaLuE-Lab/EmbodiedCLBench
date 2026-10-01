from __future__ import annotations

from typing import Any


def render_observation_text(payload: dict[str, Any]) -> str:
    task_instruction = str(payload.get("task_instruction", "") or "").strip()
    image_paths = list(payload.get("image_paths", []) or [])
    action_candidates = list(payload.get("action_candidates", []) or [])
    object_candidates = list(payload.get("object_candidates", []) or [])
    previous_action_feedback = str(payload.get("previous_action_feedback", "") or "").strip()

    lines: list[str] = []
    if task_instruction:
        lines.append(f"Task instruction: {task_instruction}")

    if image_paths:
        lines.append("Current observation image: attached.")
    else:
        lines.append("Current observation image: none.")

    if action_candidates:
        lines.append("Available actions:")
        for candidate in action_candidates:
            template_id = str(candidate.get("template_id", "") or "").strip()
            if template_id:
                lines.append(f"- {template_id}")
    else:
        lines.append("Available actions: none.")

    if object_candidates:
        lines.append("Available objects:")
        for candidate in object_candidates:
            object_id = str(candidate.get("object_id", "") or "").strip()
            if object_id:
                lines.append(f"- {object_id}")
    else:
        lines.append("Available objects: none.")

    if previous_action_feedback:
        lines.append(f"Previous action feedback: {previous_action_feedback}")

    return "\n".join(lines)


def render_action_result_text(payload: dict[str, Any]) -> str:
    accepted = bool(payload.get("accepted", False))
    done = bool(payload.get("done", False))
    reached_goal = bool(payload.get("reached_goal", False))
    action_feedback = str(payload.get("action_feedback", "") or "").strip()

    lines = [
        f"Action accepted: {'yes' if accepted else 'no'}.",
        f"Episode done: {'yes' if done else 'no'}.",
        f"Goal reached: {'yes' if reached_goal else 'no'}.",
    ]
    if action_feedback:
        lines.append(f"Feedback: {action_feedback}")
    if isinstance(payload.get('next_observation'), dict):
        lines.append("A refreshed observation is included below in next_observation.")
    return "\n".join(lines)
