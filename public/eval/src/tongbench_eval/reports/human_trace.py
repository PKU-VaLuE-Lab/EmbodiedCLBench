from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def write_human_trace_markdown(
    *,
    output_dir: Path,
    task_id: str,
    title: str | None = None,
) -> Path | None:
    output_dir = Path(output_dir)
    task_output_dir = output_dir / "task_output"
    events_path = task_output_dir / "human_trace_events.jsonl"
    if not events_path.exists():
        return None
    events = _read_jsonl(events_path)
    if not events:
        return None

    assets_dir = output_dir / "human_trace_assets"
    lines: list[str] = [
        f"# {title or task_id} Human Trace",
        "",
        "This report shows the model-facing safe-choice interaction, the model selections, and the hidden graph action mapping used by the evaluator.",
        "",
    ]

    initial_prompt = _find_initial_model_input(output_dir)
    if initial_prompt:
        lines.extend([
            "## Initial Model Input",
            "",
            "```text",
            initial_prompt.rstrip(),
            "```",
            "",
        ])

    text_only_trace = _model_text_only_trace(output_dir)
    observe_events: list[dict[str, Any]] = []
    selection_events: list[dict[str, Any]] = []
    selections_by_version: dict[Any, list[dict[str, Any]]] = {}
    for event in events:
        event_type = str(event.get("event", "") or "")
        if event_type == "observe":
            observe_events.append(event)
        elif event_type == "select_option":
            selection_events.append(event)
            selections_by_version.setdefault(_event_version(event), []).append(event)

    selection_messages = _selection_messages_for_output(output_dir, len(selection_events))
    selection_index = 0
    rendered_versions: set[Any] = set()
    for observe_index, event in enumerate(observe_events, start=1):
        version = _event_version(event)
        rendered_versions.add(version)
        lines.extend(
            _render_observe_event(
                event,
                _version_label(version, observe_index),
                assets_dir,
                text_only_trace=text_only_trace,
            )
        )
        for select_event in selections_by_version.get(version, []):
            selection_index += 1
            model_message = selection_messages[selection_index - 1] if selection_index <= len(selection_messages) else None
            lines.extend(_render_select_event(select_event, selection_index, model_message))

    for select_event in selection_events:
        version = _event_version(select_event)
        if version in rendered_versions:
            continue
        selection_index += 1
        model_message = selection_messages[selection_index - 1] if selection_index <= len(selection_messages) else None
        lines.extend(_render_select_event(select_event, selection_index, model_message))

    path = output_dir / "human_trace.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _event_version(event: dict[str, Any]) -> Any:
    value = event.get("observation_version")
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value or "")


def _version_label(version: Any, fallback_index: int) -> str:
    text = str(version or "").strip()
    return text if text else str(fallback_index)


def _render_observe_event(
    event: dict[str, Any],
    index: str,
    assets_dir: Path,
    *,
    text_only_trace: bool = False,
) -> list[str]:
    lines = [
        f"## Observation {index}",
        "",
        f"- Observation version: `{event.get('observation_version')}`",
        f"- Done: `{bool(event.get('done', False))}`",
    ]

    observation_text = _model_observation_text(event, text_only_trace=text_only_trace)
    if observation_text:
        lines.extend(["", "**Model input: observation tool result text**", "", f"```text\n{observation_text}\n```"])

    image_markdown = _stage_images_for_markdown(event, assets_dir, text_only_trace=text_only_trace)
    if image_markdown:
        lines.extend(["", "**Image(s) shown to model**", "", *image_markdown])

    lines.extend(["", "**Choices and hidden mapping**", "", "| Option | Text | Hidden mapping |", "|---|---|---|"])
    hidden_choices = event.get("hidden_choices") if isinstance(event.get("hidden_choices"), dict) else {}
    for choice in event.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        label = str(choice.get("label", "") or "")
        text = str(choice.get("text", "") or "")
        hidden = hidden_choices.get(label, {}) if isinstance(hidden_choices, dict) else {}
        mapping = _inline_code(
            f"{hidden.get('kind', '')}: {hidden.get('action_id', '')}"
            if isinstance(hidden, dict)
            else ""
        )
        lines.append(f"| {label} | {_escape_table(text)} | {mapping} |")
    lines.append("")
    return lines


def _render_select_event(
    event: dict[str, Any],
    index: int,
    model_message: dict[str, Any] | None = None,
) -> list[str]:
    hidden = event.get("hidden_choice") if isinstance(event.get("hidden_choice"), dict) else {}
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    selected_choice = str(event.get("selected_choice", "") or "")
    lines = [
        f"## Selection {index}",
        "",
        f"- Observation version: `{event.get('observation_version')}`",
        f"- Model selected: `{selected_choice}`",
        f"- Choice text: {str(event.get('selected_choice_text', '') or '')}",
        f"- Hidden action: `{hidden.get('action_id', '')}`",
        f"- Hidden kind: `{hidden.get('kind', '')}`",
        f"- Accepted: `{bool(result.get('accepted', False))}`",
        f"- Done: `{bool(result.get('done', False))}`",
        f"- Reached goal: `{bool(result.get('reached_goal', False))}`",
    ]

    assistant_text = ""
    if isinstance(model_message, dict):
        assistant_text = str(model_message.get("assistant_text", "") or "").strip()
    if assistant_text:
        lines.extend(["", "**Model output: assistant text before tool call**", "", f"```text\n{assistant_text}\n```"])

    tool_call = {
        "tool": "select_option",
        "input": {"option_id": selected_choice},
    }
    if isinstance(model_message, dict) and model_message.get("tool_name"):
        tool_call["tool"] = str(model_message.get("tool_name"))
    lines.extend(["", "**Model output: tool call**", "", f"```json\n{_json_dump(tool_call)}\n```"])

    result_text = str(result.get("result_text", "") or "").strip()
    if not result_text:
        result_text = _fallback_result_text(result)
    if result_text:
        lines.extend(["", "**Tool result returned to model**", "", f"```text\n{result_text}\n```"])
    if isinstance(result.get("next_observation"), dict):
        lines.extend([
            "",
            "Note: this tool result also carried a refreshed observation. The same refreshed observation is rendered as the next Observation section below.",
        ])
    lines.append("")
    return lines


def _model_observation_text(event: dict[str, Any], *, text_only_trace: bool = False) -> str:
    text = str(event.get("observation_text", "") or "").strip()
    if text:
        return _scrub_text_only_observation_text(text) if text_only_trace else text
    lines: list[str] = []
    task_instruction = str(event.get("task_instruction", "") or "").strip()
    previous_feedback = str(event.get("previous_action_feedback", "") or "").strip()
    if task_instruction:
        lines.append(f"Task: {task_instruction}")
    has_image = bool(event.get("image_paths") or event.get("host_image_paths")) and not text_only_trace
    lines.append("Current image: attached." if has_image else "Current image: none.")
    if previous_feedback:
        lines.append(f"Previous result: {previous_feedback}")
    choices = [choice for choice in event.get("choices", []) or [] if isinstance(choice, dict)]
    if choices:
        lines.append("Choices:")
        for choice in choices:
            label = str(choice.get("label", "") or "").strip()
            choice_text = str(choice.get("text", "") or "").strip()
            if label and choice_text:
                lines.append(f"{label}. {choice_text}")
        lines.append("")
        lines.append("Select one option by its letter.")
    return "\n".join(lines).strip()


def _scrub_text_only_observation_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in str(text).splitlines():
        if raw_line.strip() == "Current image: attached.":
            lines.append("Current image: none.")
        else:
            lines.append(raw_line)
    return "\n".join(lines).strip()


def _fallback_result_text(result: dict[str, Any]) -> str:
    if not result:
        return ""
    lines = [
        f"Choice accepted: {'yes' if bool(result.get('accepted', False)) else 'no'}.",
        f"Task complete: {'yes' if bool(result.get('done', False)) else 'no'}.",
        f"Goal reached: {'yes' if bool(result.get('reached_goal', False)) else 'no'}.",
    ]
    feedback = str(result.get("action_feedback", "") or "").strip()
    if feedback:
        lines.append(f"Feedback: {feedback}")
    return "\n".join(lines)


def _stage_images_for_markdown(
    event: dict[str, Any],
    assets_dir: Path,
    *,
    text_only_trace: bool = False,
) -> list[str]:
    if text_only_trace:
        return []
    markdown: list[str] = []
    host_paths = [str(path) for path in event.get("host_image_paths", []) or [] if str(path).strip()]
    for index, raw_path in enumerate(host_paths, start=1):
        source = Path(raw_path)
        target = assets_dir / f"obs_{event.get('observation_version', 'x')}_{index}{source.suffix.lower() or '.jpg'}"
        if not source.exists():
            if target.exists():
                markdown.append(f"![observation {index}]({target.relative_to(assets_dir.parent).as_posix()})")
            else:
                markdown.append(f"- Missing local image: `{raw_path}`")
            continue
        assets_dir.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        markdown.append(f"![observation {index}]({target.relative_to(assets_dir.parent).as_posix()})")
    if markdown:
        return markdown
    bridge_paths = [str(path) for path in event.get("image_paths", []) or [] if str(path).strip()]
    return [f"- Model image path: `{path}`" for path in bridge_paths]


def _model_text_only_trace(output_dir: Path) -> bool:
    chat = _find_ancestor_file(output_dir, "chat.jsonl")
    if chat is None:
        return False
    try:
        text = chat.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "Current image: none." in text


def _find_initial_model_input(output_dir: Path) -> str:
    readme = _find_ancestor_file(output_dir, "README_TONGSIM_COMPOSITE.md")
    if readme is not None:
        text = readme.read_text(encoding="utf-8").strip()
        if text:
            return text
    chat = _find_ancestor_file(output_dir, "chat.jsonl")
    if chat is None:
        return ""
    for line in chat.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "message":
            continue
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        if message.get("role") != "user":
            continue
        text = _chat_content_text(message.get("content"))
        if text.strip():
            return text.strip()
    return ""


def _find_ancestor_file(output_dir: Path, filename: str) -> Path | None:
    current = Path(output_dir).resolve()
    for _ in range(10):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _selection_messages_for_output(output_dir: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    chat = _find_ancestor_file(output_dir, "chat.jsonl")
    if chat is None:
        return []
    all_messages = _read_selection_messages(chat)
    if not all_messages:
        return []
    offset = _selection_offset_from_previous_task_runs(output_dir)
    return all_messages[offset : offset + count]


def _selection_offset_from_previous_task_runs(output_dir: Path) -> int:
    output_dir = Path(output_dir).resolve()
    parent = output_dir.parent
    if parent.name != "task_runs" or not parent.exists():
        return 0
    offset = 0
    for sibling in sorted(item for item in parent.iterdir() if item.is_dir()):
        if sibling.resolve() == output_dir:
            break
        events_path = sibling / "task_output" / "human_trace_events.jsonl"
        if events_path.exists():
            offset += sum(1 for event in _read_jsonl(events_path) if event.get("event") == "select_option")
    return offset


def _read_selection_messages(chat_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in chat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "message":
            continue
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "tool_use":
                continue
            name = str(part.get("name", "") or "")
            if not name.endswith("select_option"):
                continue
            tool_input = part.get("input") if isinstance(part.get("input"), dict) else {}
            records.append(
                {
                    "assistant_text": "\n".join(text_parts).strip(),
                    "tool_name": name,
                    "tool_input": dict(tool_input),
                    "option_id": str(tool_input.get("option_id", "") or ""),
                }
            )
    return records


def _chat_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts)


def _escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _inline_code(value: str) -> str:
    return f"`{str(value).replace('`', '')}`"


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
