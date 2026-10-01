from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_VISIBLE_ACTION_TYPES = (
    "look_at",
    "walk_to",
    "switch_on",
    "switch_off",
    "open",
    "close",
)

def normalize_choice_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"[A-Z]", text)
    return match.group(0) if match else ""


def build_choice_observation(
    *,
    task: Any,
    observation_payload: dict[str, Any],
    choice_count: int = 4,
    seed: int = 0,
    observation_version: int = 0,
    choice_bank: dict[tuple[str, str], list[dict[str, Any]]] | None = None,
    require_choice_bank: bool = False,
) -> dict[str, Any]:
    _ = require_choice_bank
    current_state_id = str(observation_payload.get("current_state_id", "") or "")
    choice_count = max(2, min(int(choice_count or 4), len(CHOICE_LABELS)))
    candidate_edges = _outgoing_edge_options(task, current_state_id)
    bank_candidates = _choice_bank_option_candidates(
        task=task,
        state_id=current_state_id,
        candidate_edges=candidate_edges,
        choice_bank=choice_bank,
    )
    if not bank_candidates:
        task_id = str(getattr(task, "task_id", "") or _raw_task(task).get("task_id", ""))
        raise ValueError(
            f"Missing choice bank record for task_id={task_id} state_id={current_state_id}. "
            "Safe-choice no longer auto-generates options; create the option bank data file explicitly."
        )
    selected = list(bank_candidates[:choice_count])
    if len(selected) < choice_count:
        task_id = str(getattr(task, "task_id", "") or _raw_task(task).get("task_id", ""))
        raise ValueError(
            f"Choice bank record for task_id={task_id} state_id={current_state_id} "
            f"has {len(selected)} choices, expected at least {choice_count}."
        )

    choices: list[dict[str, str]] = []
    hidden_choices: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selected[:choice_count]):
        label = CHOICE_LABELS[index]
        public = {
            "label": label,
            "text": str(item["text"]),
        }
        choices.append(public)
        hidden_choices[label] = {
            "label": label,
            "text": str(item["text"]),
            "kind": str(item.get("kind", "invalid")),
            "action_id": str(item.get("action_id") or f"__INVALID_CHOICE_{label}__"),
            "action_type": str(item.get("action_type") or ""),
            "object_id": item.get("object_id"),
            "target_id": item.get("target_id"),
            "to_state": item.get("to_state"),
            "is_success_edge": bool(item.get("is_success_edge", False)),
        }

    return {
        "choices": choices,
        "hidden_choices": hidden_choices,
        "choice_count": choice_count,
        "choice_seed": int(seed or 0),
        "observation_version": int(observation_version),
        "current_state_id": current_state_id,
        "choice_source": "choice_bank",
    }


def load_choice_bank(path: str | Path | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if path is None:
        return {}
    choice_path = Path(path)
    if not choice_path.is_file():
        raise FileNotFoundError(f"Missing choice bank JSONL: {choice_path}")
    bank: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line_number, raw_line in enumerate(choice_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Choice bank line {line_number} is not a JSON object: {choice_path}")
        task_id = str(payload.get("task_id") or "").strip()
        state_id = str(payload.get("state_id") or payload.get("current_state_id") or "").strip()
        choices = payload.get("choices")
        if not task_id or not state_id or not isinstance(choices, list):
            raise ValueError(f"Choice bank line {line_number} missing task_id/state_id/choices: {choice_path}")
        normalized_choices = [dict(item) for item in choices if isinstance(item, dict)]
        if not normalized_choices:
            raise ValueError(f"Choice bank line {line_number} has no usable choices: {choice_path}")
        bank[(task_id, state_id)] = normalized_choices
    return bank


def render_choice_observation_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    task_instruction = str(payload.get("task_instruction", "") or "").strip()
    previous_feedback = str(payload.get("previous_action_feedback", "") or "").strip()
    if task_instruction:
        lines.append(f"Task: {task_instruction}")
    lines.append("Current image: attached." if payload.get("image_paths") else "Current image: none.")
    if previous_feedback:
        lines.append(f"Previous result: {previous_feedback}")
    lines.append("Choices:")
    for choice in payload.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label", "") or "").strip()
        text = str(choice.get("text", "") or "").strip()
        if label and text:
            lines.append(f"{label}. {text}")
    lines.append("")
    lines.append("Select one option by its letter.")
    return "\n".join(lines)


def render_choice_action_result_text(payload: dict[str, Any]) -> str:
    accepted = bool(payload.get("accepted", False))
    done = bool(payload.get("done", False))
    reached_goal = bool(payload.get("reached_goal", False))
    feedback = str(payload.get("action_feedback", "") or "").strip()
    lines = [
        f"Choice accepted: {'yes' if accepted else 'no'}.",
        f"Task complete: {'yes' if done else 'no'}.",
        f"Goal reached: {'yes' if reached_goal else 'no'}.",
    ]
    if feedback:
        lines.append(f"Feedback: {feedback}")
    if isinstance(payload.get("next_observation"), dict):
        lines.append("A refreshed observation is included below.")
    return "\n".join(lines)


def _raw_task(task: Any) -> dict[str, Any]:
    raw = getattr(task, "raw", None)
    return raw if isinstance(raw, dict) else {}


def _raw_edges(task: Any) -> list[dict[str, Any]]:
    raw = _raw_task(task)
    edges = raw.get("dag_edges", raw.get("edges", []))
    return [item for item in edges if isinstance(item, dict)]


def _task_id(task: Any) -> str:
    return str(getattr(task, "task_id", "") or _raw_task(task).get("task_id", "")).strip()


def _edge_action_id(edge: dict[str, Any]) -> str:
    return str(
        edge.get("action")
        or edge.get("atomic_action")
        or edge.get("parent_action")
        or edge.get("action_id")
        or ""
    ).strip()


def _outgoing_edge_options(task: Any, state_id: str) -> list[dict[str, Any]]:
    seen_index: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for edge in _raw_edges(task):
        if str(edge.get("from", "") or "") != state_id:
            continue
        action_id = _edge_action_id(edge)
        if not action_id:
            continue
        if action_id in seen_index:
            existing = output[seen_index[action_id]]
            existing["is_success_edge"] = bool(existing.get("is_success_edge", False) or edge.get("is_success_edge", False))
            continue
        seen_index[action_id] = len(output)
        parsed = parse_action_id(action_id, edge)
        output.append(
            {
                "kind": "valid",
                "action_id": action_id,
                "action_type": parsed["action_type"],
                "object_id": parsed.get("object_id"),
                "target_id": parsed.get("target_id"),
                "to_state": str(edge.get("to", "") or ""),
                "is_success_edge": bool(edge.get("is_success_edge", False)),
                "text": naturalize_action(action_id, edge),
            }
        )
    return output


def _choice_bank_option_candidates(
    *,
    task: Any,
    state_id: str,
    candidate_edges: list[dict[str, Any]],
    choice_bank: dict[tuple[str, str], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    if not choice_bank:
        return []
    raw_items = choice_bank.get((_task_id(task), state_id)) or []
    if not raw_items:
        return []
    edge_by_action = {str(item.get("action_id") or ""): item for item in candidate_edges}
    candidates: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_items):
        text = str(raw_item.get("text") or raw_item.get("option_text") or "").strip()
        if not text:
            continue
        kind = str(raw_item.get("kind") or "").strip().lower() or "invalid"
        action_id = str(raw_item.get("action_id") or "").strip()
        if kind == "valid":
            edge = edge_by_action.get(action_id)
            if edge is None:
                raise ValueError(f"Choice bank valid action is not outgoing from state {state_id}: {action_id}")
            candidate = dict(edge)
            candidate["text"] = text
            candidate["kind"] = "valid"
            candidate["choice_design_note"] = raw_item.get("design_note") or raw_item.get("rationale_zh")
            candidates.append(candidate)
            continue
        parsed = parse_action_id(action_id) if action_id else {}
        candidates.append(
            {
                "kind": "invalid",
                "action_id": action_id or f"__INVALID_CHOICE_BANK_{index}__",
                "action_type": str(raw_item.get("action_type") or parsed.get("action_type") or ""),
                "object_id": raw_item.get("object_id") or parsed.get("object_id"),
                "target_id": raw_item.get("target_id") or parsed.get("target_id"),
                "to_state": None,
                "is_success_edge": False,
                "text": text,
                "choice_design_note": raw_item.get("design_note") or raw_item.get("rationale_zh"),
            }
        )
    return _dedupe_option_texts(candidates)


def parse_action_id(action_id: str, edge: dict[str, Any] | None = None) -> dict[str, Any]:
    edge = edge or {}
    action_type = str(edge.get("action_type", "") or "").strip()
    role_bindings = dict(edge.get("role_bindings", {}) or {})
    object_id = role_bindings.get("object") or role_bindings.get("r1")
    target_id = role_bindings.get("target") or role_bindings.get("object2") or role_bindings.get("r2")
    if action_type and object_id:
        return {"action_type": action_type, "object_id": object_id, "target_id": target_id}

    if action_id.startswith("reach_{object}__"):
        return {"action_type": "walk_to", "object_id": action_id.split("__", 1)[1], "target_id": None}

    place_match = re.match(r"^place_(.+?)_(on|in)_(.+)$", action_id)
    if place_match:
        return {
            "action_type": "place",
            "object_id": place_match.group(1),
            "target_id": place_match.group(3),
        }

    for prefix in sorted(
        (*DEFAULT_VISIBLE_ACTION_TYPES, "pick_up", "drop", "plug_in", "unplug", "stand_up"),
        key=len,
        reverse=True,
    ):
        marker = f"{prefix}_"
        if action_id.startswith(marker):
            return {"action_type": prefix, "object_id": action_id[len(marker):], "target_id": None}
    return {"action_type": action_type or "interact", "object_id": object_id, "target_id": target_id}


def naturalize_action(action_id: str, edge: dict[str, Any] | None = None) -> str:
    parsed = parse_action_id(action_id, edge)
    return naturalize_action_from_parts(parsed["action_type"], parsed.get("object_id"), parsed.get("target_id"))


def naturalize_action_from_parts(action_type: str, object_id: str | None, target_id: str | None = None) -> str:
    obj = object_display_name(object_id)
    target = object_display_name(target_id)
    if action_type == "look_at" and obj:
        return f"Inspect the {obj}."
    if action_type == "walk_to" and obj:
        return f"Move closer to the {obj}."
    if action_type == "switch_on" and obj:
        return f"Turn on the {obj}."
    if action_type == "switch_off" and obj:
        return f"Turn off the {obj}."
    if action_type == "open" and obj:
        return f"Open the {obj}."
    if action_type == "close" and obj:
        return f"Close the {obj}."
    if action_type == "pick_up" and obj:
        return f"Pick up the {obj}."
    if action_type == "place" and obj and target:
        return f"Place the {obj} on the {target}."
    if obj:
        return f"Interact with the {obj}."
    return "Make another household adjustment."


def object_display_name(object_id: Any) -> str:
    text = str(object_id or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    keyword_map = (
        ("aircondition", "air conditioner"),
        ("tv_hanged", "wall-mounted television"),
        ("television", "television"),
        ("tv", "television"),
        ("ceilinglamp", "ceiling light"),
        ("tablelamp", "table lamp"),
        ("nightstand", "nightstand"),
        ("bedside", "bedside cabinet"),
        ("diswasher", "dishwasher"),
        ("dishwasher", "dishwasher"),
        ("window", "window"),
        ("bookshelf", "bookshelf"),
        ("bookshelves", "bookshelf"),
        ("trashcan", "trash can"),
        ("coffeetable", "coffee table"),
        ("remotecontroller", "remote control"),
        ("cup", "cup"),
        ("plate", "plate"),
        ("drawer", "drawer"),
        ("cabinet", "cabinet"),
    )
    compact = re.sub(r"[^a-z]", "", lowered)
    for key, label in keyword_map:
        if key in compact or key in lowered:
            return label
    cleaned = re.sub(r"^BP_", "", text)
    cleaned = re.sub(r"_[A-Z]+_\\d+$", "", cleaned)
    cleaned = re.sub(r"_[A-Z]?(?:\\d+)$", "", cleaned)
    cleaned = re.sub(r"[_\\-]+", " ", cleaned)
    cleaned = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", cleaned)
    cleaned = re.sub(r"\\b\\d+\\b", "", cleaned)
    cleaned = re.sub(r"\\s+", " ", cleaned).strip().lower()
    return cleaned or "object"


def _dedupe_option_texts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        text = str(value.get("text", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(value)
    return output
