from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..object_normalizer import infer_object_type_from_id, normalize_object_type
from ...utils.python_utils import current_python


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_image(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _run_env_command(workspace_dir: Path, *args: str) -> dict[str, Any]:
    command = [current_python(), "tongsim_env.py", *args]
    src_dir = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(src_dir) if not existing_pythonpath else os.pathsep.join([str(src_dir), existing_pythonpath])
    try:
        result = subprocess.run(
            command,
            cwd=workspace_dir,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "").strip()
        stderr = (exc.stderr or "").strip()
        details = [
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
    return json.loads(_task_json_path(workspace_dir).read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_prompt_log(path: Path, step_index: int, user_prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"===== STEP {step_index} =====\n")
        handle.write(user_prompt.rstrip())
        handle.write("\n\n")


def _label_for_index(index: int) -> str:
    if index < 0:
        raise ValueError("Option index must be non-negative")
    letters = []
    value = index
    while True:
        letters.append(chr(ord("A") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def _pretty_atom(value: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "unknown object"


_DISPLAY_NAME_ALIASES = {
    "aircondition": "air conditioner",
    "bathshelf": "bath shelf",
    "coffeetable": "coffee table",
    "remotecontrol": "remote control",
    "tablelamp": "table lamp",
    "ceilinglamp": "ceiling lamp",
    "trashcan": "trash can",
}


def naturalize_action_text(action_text: str) -> str:
    raw = str(action_text or "").strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", raw)
    if match:
        verb = _pretty_atom(match.group(1))
        args = [_pretty_atom(item) for item in match.group(2).split(",") if item.strip()]
        if verb == "place" and len(args) >= 2:
            return f"Place {args[0]} on {args[1]}"
        if verb in {"pick", "pickup", "pick up"} and args:
            return f"Pick up {args[0]}"
        if verb == "inspect" and args:
            return f"Inspect {args[0]}"
        if verb in {"look at", "look"} and args:
            return f"Look at {args[0]}"
        if verb in {"walk to", "reach"} and args:
            return f"Move toward {args[0]}"
        if verb == "wash" and args:
            return f"Wash {args[0]}"
        if args:
            return f"{verb.capitalize()} {' and '.join(args)}"
        return verb.capitalize()

    compact_match = re.match(r"^place_(.+?)_on_(.+)$", raw)
    if compact_match:
        return f"Place {_pretty_atom(compact_match.group(1))} on {_pretty_atom(compact_match.group(2))}"
    compact_match = re.match(r"^(?:pick_up|pick|acquire|inspect|look_at|walk_to|wash)_(.+)$", raw)
    if compact_match:
        verb = raw[: raw.rfind("_")]
        return f"{_pretty_atom(verb).capitalize()} {_pretty_atom(compact_match.group(1))}"
    return _pretty_atom(raw).capitalize()


@dataclass(frozen=True)
class NaturalLanguageOption:
    label: str
    text: str
    action_id: str | None = None
    action_level: str | None = None
    template_id: str | None = None
    action_type: str | None = None
    role_bindings: dict[str, str] | None = None
    required_roles: list[str] | None = None


@dataclass(frozen=True)
class NaturalLanguageObjectOption:
    label: str
    text: str
    object_id: str


def _object_display_name(object_id: str) -> str:
    inferred = infer_object_type_from_id(object_id)
    if inferred and inferred != "unknown":
        return _DISPLAY_NAME_ALIASES.get(inferred, _pretty_atom(inferred))
    return _pretty_atom(object_id)


def _role_allowed_object_ids(
    role: str,
    candidate: dict[str, Any],
    object_candidates: list[dict[str, Any]],
) -> list[str]:
    constraints = candidate.get("role_constraints", {})
    allowed_types = []
    if isinstance(constraints, dict) and isinstance(constraints.get(role), list):
        allowed_types = [normalize_object_type(str(item)) for item in constraints.get(role, [])]
    output: list[str] = []
    for item in object_candidates:
        object_id = str(item.get("object_id", "") or "").strip()
        if not object_id:
            continue
        if allowed_types and infer_object_type_from_id(object_id) not in allowed_types:
            continue
        output.append(object_id)
    return output


def _render_library_action_text(candidate: dict[str, Any]) -> str:
    action_type = str(candidate.get("action_type", "") or candidate.get("template_id", "action"))
    template_id = str(candidate.get("template_id", "") or action_type)
    if action_type == "look_at":
        return "Look at an object"
    if action_type == "point_at":
        return "Point at an object"
    if action_type == "walk_to":
        return "Move toward an object"
    if action_type == "pick_up":
        return "Pick up an object"
    if action_type == "open":
        return "Open an object"
    if action_type == "close":
        return "Close an object"
    if action_type == "switch_on":
        return "Switch on an object"
    if action_type == "switch_off":
        return "Switch off an object"
    if action_type == "plug_in":
        return "Plug in an object"
    if action_type == "unplug":
        return "Unplug an object"
    if action_type in {"wash_object", "wash"}:
        return "Wash an object"
    if template_id == "place_{object1}_on_{object2}":
        return "Place one object on another object"
    if template_id == "place_{object1}_in_{object2}":
        return "Place one object in another object"
    if action_type == "place":
        return "Place an object"
    if action_type == "drop":
        return "Drop an object"
    if action_type == "stand_up":
        return "Stand up"
    if action_type == "wash_hand":
        return "Wash hands"
    description = str(candidate.get("description", "") or "").strip()
    return description or _pretty_atom(action_type).capitalize()


def _render_library_option_text(candidate: dict[str, Any], role_bindings: dict[str, str]) -> str:
    action_type = str(candidate.get("action_type", "") or candidate.get("template_id", "action"))
    template_id = str(candidate.get("template_id", "") or action_type)
    display_values = {
        role: _object_display_name(object_id)
        for role, object_id in role_bindings.items()
    }

    if action_type == "look_at" and display_values.get("object"):
        return f"Look at {display_values['object']}"
    if action_type == "point_at" and display_values.get("object"):
        return f"Point at {display_values['object']}"
    if action_type == "walk_to" and display_values.get("object"):
        return f"Move toward {display_values['object']}"
    if action_type == "pick_up" and display_values.get("object"):
        return f"Pick up {display_values['object']}"
    if action_type == "open" and display_values.get("object"):
        return f"Open {display_values['object']}"
    if action_type == "close" and display_values.get("object"):
        return f"Close {display_values['object']}"
    if action_type == "switch_on" and display_values.get("object"):
        return f"Switch on {display_values['object']}"
    if action_type == "switch_off" and display_values.get("object"):
        return f"Switch off {display_values['object']}"
    if action_type == "plug_in" and display_values.get("object"):
        return f"Plug in {display_values['object']}"
    if action_type == "unplug" and display_values.get("object"):
        return f"Unplug {display_values['object']}"
    if action_type in {"wash_object", "wash"} and display_values.get("object"):
        return f"Wash {display_values['object']}"
    if template_id == "place_{object1}_on_{object2}" and display_values.get("object1") and display_values.get("object2"):
        return f"Place {display_values['object1']} on {display_values['object2']}"
    if template_id == "place_{object1}_in_{object2}" and display_values.get("object1") and display_values.get("object2"):
        return f"Place {display_values['object1']} in {display_values['object2']}"
    if action_type == "place" and display_values.get("object"):
        return f"Place {display_values['object']}"
    if action_type == "drop" and display_values.get("object"):
        return f"Drop {display_values['object']}"
    if action_type == "stand_up":
        return "Stand up"
    if action_type == "wash_hand":
        return "Wash hands"
    if display_values:
        return f"{_pretty_atom(action_type).capitalize()} {' and '.join(display_values[role] for role in sorted(display_values))}"
    description = str(candidate.get("description", "") or "").strip()
    return description or _pretty_atom(action_type).capitalize()


def _build_library_natural_language_options(observation: dict[str, Any]) -> list[NaturalLanguageOption]:
    output: list[NaturalLanguageOption] = []
    action_candidates = [
        item for item in observation.get("model_facing_action_candidates", [])
        if isinstance(item, dict)
    ]
    for candidate in action_candidates:
        output.append(
            NaturalLanguageOption(
                label=_label_for_index(len(output)),
                text=_render_library_action_text(candidate),
                action_level=str(candidate.get("action_level", "") or ""),
                template_id=str(candidate.get("template_id", "") or ""),
                action_type=str(candidate.get("action_type", "") or ""),
                role_bindings={},
                required_roles=[str(role) for role in candidate.get("required_roles", [])],
            )
        )
    return output


def build_natural_language_object_options(observation: dict[str, Any]) -> list[NaturalLanguageObjectOption]:
    output: list[NaturalLanguageObjectOption] = []
    for item in observation.get("model_facing_object_candidates", []):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("object_id", "") or "").strip()
        if not object_id:
            continue
        output.append(
            NaturalLanguageObjectOption(
                label=str(len(output) + 1),
                text=_object_display_name(object_id),
                object_id=object_id,
            )
        )
    return output


def build_natural_language_options(observation: dict[str, Any]) -> list[NaturalLanguageOption]:
    if observation.get("action_interface") == "library_factorized":
        return _build_library_natural_language_options(observation)

    options: list[NaturalLanguageOption] = []
    for index, candidate in enumerate(observation.get("candidate_actions", [])):
        action_id = str(candidate.get("action_id", "") or "").strip()
        if not action_id:
            continue
        display_text = naturalize_action_text(str(candidate.get("text", action_id)))
        options.append(
            NaturalLanguageOption(
                label=_label_for_index(index),
                text=display_text,
                action_id=action_id,
            )
        )
    return options


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are a visual decision-making assistant in an embodied task.",
            "You will receive the task goal, the current image, and listed choices for the next decision.",
            "Choose only from the listed choices based on the task goal, the current image, and the conversation so far.",
            "Do not invent extra choices.",
            "Explain your decision briefly in natural language.",
        ]
    )


def build_user_turn_prompt(
    *,
    task_instruction: str,
    options: list[NaturalLanguageOption],
    previous_feedback: str | None,
    step_index: int,
    max_steps: int,
) -> str:
    lines = [
        f"Task: {task_instruction}",
        f"Decision step: {step_index + 1} of at most {max_steps}.",
        "Current image: attached.",
    ]
    if previous_feedback:
        lines.append(f"Previous result: {previous_feedback}")
    lines.append("Available choices:")
    if options:
        for option in options:
            lines.append(f"{option.label}. {option.text}")
    else:
        lines.append("No choices are currently available.")
    lines.extend(
        [
            "",
            "Select the best next choice. Reply naturally, and end with `Choice: <letter>`.",
        ]
    )
    return "\n".join(lines)


def build_library_user_turn_prompt(
    *,
    task_instruction: str,
    action_options: list[NaturalLanguageOption],
    object_options: list[NaturalLanguageObjectOption],
    previous_feedback: str | None,
    step_index: int,
    max_steps: int,
) -> str:
    lines = [
        f"Task: {task_instruction}",
        f"Decision step: {step_index + 1} of at most {max_steps}.",
        "Current image: attached.",
    ]
    if previous_feedback:
        lines.append(f"Previous result: {previous_feedback}")
    lines.append("Action choices:")
    if action_options:
        for option in action_options:
            lines.append(f"{option.label}. {option.text}")
    else:
        lines.append("No action choices are currently available.")
    lines.append("Object choices:")
    if object_options:
        for option in object_options:
            lines.append(f"{option.label}. {option.text}")
    else:
        lines.append("No object choices are currently available.")
    lines.extend(
        [
            "",
            "Select one action. If that action needs an object, also select object numbers from the object choices.",
            "If the action needs two objects, use Object 1 for the object being moved or placed and Object 2 for the target object.",
            "Reply naturally, and end with one final line like `Action: A; Object: 1` or `Action: G; Object 1: 1; Object 2: 2`.",
            "For actions that need no object, end with `Action: A`.",
        ]
    )
    return "\n".join(lines)


def parse_natural_language_choice(raw_response: str, options: list[NaturalLanguageOption]) -> tuple[NaturalLanguageOption | None, str | None]:
    labels = {option.label: option for option in options}
    if not labels:
        return None, "No options were available."

    patterns = [
        r"(?:choice|selected choice|select|selected|answer|option|选择|选项|答案)\s*[:：]?\s*([A-Z]{1,3})\b",
        r"\b([A-Z]{1,3})\s*(?:is my choice|is the best choice)\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, raw_response, flags=re.IGNORECASE)
        for match in reversed(matches):
            label = str(match).upper()
            if label in labels:
                return labels[label], None

    stripped_lines = [line.strip().upper() for line in raw_response.splitlines() if line.strip()]
    for line in reversed(stripped_lines):
        if line in labels:
            return labels[line], None

    mentioned = [
        label
        for label in labels
        if re.search(rf"\b{re.escape(label)}\b", raw_response, flags=re.IGNORECASE)
    ]
    if len(mentioned) == 1:
        return labels[mentioned[0]], None
    if len(mentioned) > 1:
        return None, f"Ambiguous option labels mentioned: {', '.join(mentioned)}"
    return None, "Could not find a selected option label in the model response."


def _role_prompt_name(role: str) -> str:
    if role == "object":
        return "Object"
    if role in {"object1", "r1"}:
        return "Object 1"
    if role in {"object2", "r2"}:
        return "Object 2"
    return _pretty_atom(role).title()


def parse_library_natural_language_selection(
    raw_response: str,
    action_options: list[NaturalLanguageOption],
    object_options: list[NaturalLanguageObjectOption],
) -> tuple[NaturalLanguageOption | None, str | None]:
    action_labels = {option.label: option for option in action_options}
    object_labels = {option.label: option for option in object_options}
    if not action_labels:
        return None, "No action choices were available."

    action_label: str | None = None
    action_patterns = [
        r"(?:action|action choice|selected action)\s*[:锛歖?\s*([A-Z]{1,3})\b",
        r"(?:choice|selected choice|select|selected|answer|option|閫夋嫨|閫夐」|绛旀)\s*[:锛歖?\s*([A-Z]{1,3})\b",
    ]
    action_patterns = [
        r"(?:action|action choice|selected action)\s*:\s*([A-Z]{1,3})\b",
        r"(?:choice|selected choice|select|selected|answer|option)\s*:\s*([A-Z]{1,3})\b",
    ]
    for pattern in action_patterns:
        matches = re.findall(pattern, raw_response, flags=re.IGNORECASE)
        for match in reversed(matches):
            candidate = str(match).upper()
            if candidate in action_labels:
                action_label = candidate
                break
        if action_label is not None:
            break
    if action_label is None:
        mentioned = [
            label
            for label in action_labels
            if re.search(rf"\b{re.escape(label)}\b", raw_response, flags=re.IGNORECASE)
        ]
        if len(mentioned) == 1:
            action_label = mentioned[0]
    if action_label is None:
        return None, "Could not find a selected action label in the model response."

    selected = action_labels[action_label]
    required_roles = [str(role) for role in selected.required_roles or []]
    if not required_roles:
        selected = NaturalLanguageOption(
            label=selected.label,
            text=selected.text,
            action_level=selected.action_level,
            template_id=selected.template_id,
            action_type=selected.action_type,
            role_bindings={},
            required_roles=required_roles,
        )
        return selected, None

    if not object_labels:
        return None, "The selected action needs an object, but no object choices were available."

    role_bindings: dict[str, str] = {}
    final_line = raw_response.strip().splitlines()[-1] if raw_response.strip() else raw_response
    for role in required_roles:
        prompt_name = _role_prompt_name(role)
        if role == "object":
            pattern = r"(?:object|target)\s*[:锛歖?\s*(\d+)\b"
        else:
            pattern = rf"{re.escape(prompt_name)}\s*[:锛歖?\s*(\d+)\b"
        pattern = (
            r"(?:object|target)\s*:\s*(\d+)\b"
            if role == "object"
            else rf"{re.escape(prompt_name)}\s*:\s*(\d+)\b"
        )
        matches = re.findall(pattern, raw_response, flags=re.IGNORECASE)
        if matches:
            label = str(matches[-1])
            if label not in object_labels:
                return None, f"Selected object label is not in object choices: {label}"
            role_bindings[role] = object_labels[label].object_id

    if len(role_bindings) < len(required_roles):
        labels_in_final_line = [
            item
            for item in re.findall(r"\b\d+\b", final_line)
            if item in object_labels
        ]
        for role, label in zip(required_roles, labels_in_final_line):
            role_bindings.setdefault(role, object_labels[label].object_id)

    missing_roles = [role for role in required_roles if role not in role_bindings]
    if missing_roles:
        return None, f"Missing object choice for: {', '.join(_role_prompt_name(role) for role in missing_roles)}"

    selected = NaturalLanguageOption(
        label=selected.label,
        text=selected.text,
        action_level=selected.action_level,
        template_id=selected.template_id,
        action_type=selected.action_type,
        role_bindings=role_bindings,
        required_roles=required_roles,
    )
    return selected, None


class NaturalLanguageConversationBackend(Protocol):
    name: str

    def generate_reply(
        self,
        messages: list[dict[str, Any]],
        current_image_paths: list[Path],
    ) -> str:
        ...


class MockNaturalLanguageBackend:
    name = "mock-nl"

    def __init__(self, scripted_choices: list[str] | None = None) -> None:
        self.scripted_choices = scripted_choices or []
        self._index = 0

    def generate_reply(
        self,
        messages: list[dict[str, Any]],
        current_image_paths: list[Path],
    ) -> str:
        _ = messages, current_image_paths
        choice = self.scripted_choices[self._index] if self._index < len(self.scripted_choices) else "A"
        self._index += 1
        if ":" in choice or ";" in choice or "\n" in choice:
            return f"I choose this because it appears to be the best next step.\n{choice}"
        return f"I choose option {choice} because it appears to be the best next step.\nChoice: {choice}"


class OpenAICompatibleConversationBackend:
    name = "openai-compatible-nl"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _message_payload(
        self,
        messages: list[dict[str, Any]],
        current_image_paths: list[Path],
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        last_user_index = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        for index, message in enumerate(messages):
            role = str(message.get("role", "user"))
            text = str(message.get("content", ""))
            if index == last_user_index and current_image_paths:
                content: list[dict[str, Any]] = [{"type": "text", "text": text}]
                for image_path in current_image_paths:
                    content.append({"type": "image_url", "image_url": {"url": _encode_image(image_path)}})
                payload.append({"role": role, "content": content})
            else:
                payload.append({"role": role, "content": text})
        return payload

    def generate_reply(
        self,
        messages: list[dict[str, Any]],
        current_image_paths: list[Path],
    ) -> str:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("requests is required for the OpenAI-compatible natural-language backend.") from exc

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": self._message_payload(messages, current_image_paths),
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
        return json.dumps(content, ensure_ascii=False)


class LocalHFQwenConversationBackend:
    name = "local-hf-qwen-nl"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        device_map: str | None = None,
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        from .backends.local_hf_qwen import LocalHFQwenBackend

        self._backend = LocalHFQwenBackend(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _flatten_messages(self, messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}:\n{content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def generate_reply(
        self,
        messages: list[dict[str, Any]],
        current_image_paths: list[Path],
    ) -> str:
        self._backend._load()
        return self._backend._infer_raw_response(
            self._flatten_messages(messages),
            current_image_paths,
        )


def _result_feedback(result: dict[str, Any]) -> str:
    if bool(result.get("valid", False)):
        if bool(result.get("done", False)) and bool(result.get("reached_goal", False)):
            return "The previous choice was accepted and the task goal is now satisfied."
        return "The previous choice was accepted and the situation changed."
    diagnostic = result.get("diagnostic")
    if isinstance(diagnostic, dict) and diagnostic.get("reason"):
        return f"The previous choice was rejected: {diagnostic['reason']}"
    status = result.get("transition_status") or result.get("execution_status")
    if status:
        return f"The previous choice was rejected: {status}."
    return "The previous choice was rejected."


def _conversation_summary(
    *,
    backend: NaturalLanguageConversationBackend,
    task: dict[str, Any],
    entries: list[dict[str, Any]],
    final_state: dict[str, Any],
) -> dict[str, Any]:
    parse_error_steps = [entry["step_index"] for entry in entries if entry.get("parse_error")]
    invalid_steps = [entry["step_index"] for entry in entries if not entry.get("env_step_result", {}).get("valid", False)]
    return {
        "task_id": str(task.get("task_id", "unknown_task")),
        "model_backend": backend.name,
        "conversation_turn_count": len(entries) * 2,
        "total_steps": len(entries),
        "valid_decisions": sum(1 for entry in entries if entry.get("env_step_result", {}).get("valid", False)),
        "invalid_decisions": len(invalid_steps),
        "parse_error_count": len(parse_error_steps),
        "first_parse_error_step": parse_error_steps[0] if parse_error_steps else None,
        "first_invalid_step": invalid_steps[0] if invalid_steps else None,
        "final_done": bool(final_state.get("done", False)),
        "final_state_id": str(final_state.get("current_state_id", "")),
    }


def run_natural_language_conversation_loop(
    *,
    workspace_dir: Path,
    backend: NaturalLanguageConversationBackend,
    max_steps: int = 30,
    action_interface: str = "full_action",
    action_library_mode: str = "atomic_only",
    atomic_template_path: Path | None = None,
    subtask_template_path: Path | None = None,
    decision_log_path: Path | None = None,
    conversation_log_path: Path | None = None,
    prompt_log_path: Path | None = None,
) -> dict[str, Any]:
    workspace_dir = Path(workspace_dir)
    decision_log_path = decision_log_path or (workspace_dir / "nl_decision_log.jsonl")
    conversation_log_path = conversation_log_path or (workspace_dir / "nl_conversation.jsonl")
    prompt_log_path = prompt_log_path or (workspace_dir / "nl_prompt_log.txt")
    for path in (decision_log_path, conversation_log_path, prompt_log_path):
        if path.exists():
            path.unlink()

    task = _load_task(workspace_dir)
    messages: list[dict[str, Any]] = [{"role": "system", "content": build_system_prompt()}]
    _append_jsonl(conversation_log_path, {"role": "system", "content": messages[0]["content"], "timestamp": _timestamp()})
    decision_entries: list[dict[str, Any]] = []
    previous_feedback: str | None = None

    for step_index in range(max_steps):
        observe_args = ["observe", "--action-interface", action_interface]
        if action_interface == "library_factorized":
            observe_args.extend(["--action-library-mode", action_library_mode])
            if atomic_template_path is not None:
                observe_args.extend(["--atomic-template-path", str(atomic_template_path)])
            if subtask_template_path is not None:
                observe_args.extend(["--subtask-template-path", str(subtask_template_path)])
        observation = _run_env_command(workspace_dir, *observe_args)
        if observation.get("done"):
            break
        options = build_natural_language_options(observation)
        object_options = (
            build_natural_language_object_options(observation)
            if action_interface == "library_factorized"
            else []
        )
        image_paths = [Path(path) for path in observation.get("image_paths", [])]
        if action_interface == "library_factorized":
            user_prompt = build_library_user_turn_prompt(
                task_instruction=str(observation.get("task_instruction", task.get("description", ""))),
                action_options=options,
                object_options=object_options,
                previous_feedback=previous_feedback,
                step_index=step_index,
                max_steps=max_steps,
            )
        else:
            user_prompt = build_user_turn_prompt(
                task_instruction=str(observation.get("task_instruction", task.get("description", ""))),
                options=options,
                previous_feedback=previous_feedback,
                step_index=step_index,
                max_steps=max_steps,
            )
        user_message = {
            "role": "user",
            "content": user_prompt,
            "image_paths": [str(path) for path in image_paths],
        }
        messages.append(user_message)
        _append_jsonl(conversation_log_path, {**user_message, "timestamp": _timestamp()})
        _append_prompt_log(prompt_log_path, step_index, user_prompt)

        raw_response = backend.generate_reply(messages, image_paths)
        assistant_message = {"role": "assistant", "content": raw_response}
        messages.append(assistant_message)
        _append_jsonl(conversation_log_path, {**assistant_message, "timestamp": _timestamp()})

        if action_interface == "library_factorized":
            selected_option, parse_error = parse_library_natural_language_selection(raw_response, options, object_options)
            if selected_option is None:
                act_args = [
                    "act",
                    "--action-interface",
                    "library_factorized",
                    "--action-library-mode",
                    action_library_mode,
                    "--action_level",
                    "",
                    "--template_id",
                    "",
                    "--action_type",
                    "",
                    "--role_bindings",
                    "{}",
                    "--parse-error",
                    parse_error or "Could not parse selected option.",
                    "--raw-response",
                    raw_response,
                ]
            else:
                act_args = [
                    "act",
                    "--action-interface",
                    "library_factorized",
                    "--action-library-mode",
                    action_library_mode,
                    "--action_level",
                    selected_option.action_level or "",
                    "--template_id",
                    selected_option.template_id or "",
                    "--action_type",
                    selected_option.action_type or "",
                    "--role_bindings",
                    json.dumps(selected_option.role_bindings or {}, ensure_ascii=False),
                ]
            if atomic_template_path is not None:
                act_args.extend(["--atomic-template-path", str(atomic_template_path)])
            if subtask_template_path is not None:
                act_args.extend(["--subtask-template-path", str(subtask_template_path)])
            env_step_result = _run_env_command(workspace_dir, *act_args)
        else:
            selected_option, parse_error = parse_natural_language_choice(raw_response, options)
            action_id = selected_option.action_id if selected_option is not None else "__INVALID_MODEL_OUTPUT__"
            env_step_result = _run_env_command(workspace_dir, "act", "--action_id", action_id)
        previous_feedback = _result_feedback(env_step_result)

        entry = {
            "step_index": step_index,
            "task_id": str(task.get("task_id", workspace_dir.name)),
            "image_paths": [str(path) for path in image_paths],
            "natural_language_options": [
                {"label": option.label, "text": option.text}
                for option in options
            ],
            "natural_language_object_options": [
                {"label": option.label, "text": option.text}
                for option in object_options
            ],
            "action_interface": action_interface,
            "action_library_mode": action_library_mode if action_interface == "library_factorized" else None,
            "hidden_selected_action_id": selected_option.action_id if selected_option is not None else None,
            "hidden_selected_action_level": selected_option.action_level if selected_option is not None else None,
            "hidden_selected_template_id": selected_option.template_id if selected_option is not None else None,
            "hidden_selected_action_type": selected_option.action_type if selected_option is not None else None,
            "hidden_selected_role_bindings": selected_option.role_bindings if selected_option is not None else None,
            "selected_option_label": selected_option.label if selected_option is not None else None,
            "selected_option_text": selected_option.text if selected_option is not None else None,
            "raw_response": raw_response,
            "parse_error": parse_error,
            "env_step_result": env_step_result,
            "timestamp": _timestamp(),
        }
        decision_entries.append(entry)
        _append_jsonl(decision_log_path, entry)
        if env_step_result.get("done"):
            break

    final_state = _run_env_command(workspace_dir, "finish")
    summary = _conversation_summary(
        backend=backend,
        task=task,
        entries=decision_entries,
        final_state=final_state,
    )
    summary_path = decision_log_path.with_name("nl_decision_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "decision_log_path": str(decision_log_path),
        "conversation_log_path": str(conversation_log_path),
        "prompt_log_path": str(prompt_log_path),
        "decision_summary_path": str(summary_path),
        "decision_summary": summary,
        "final_state": final_state,
    }
