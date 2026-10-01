"""Build native checkpoints from independently saved learning sessions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from tongbench_eval.agents.base import AgentCheckpoint


def _task_outcome(checkpoint: AgentCheckpoint) -> dict[str, Any] | None:
    path = checkpoint.path / "task_outcome.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _session_files(checkpoint: AgentCheckpoint) -> list[Path]:
    native = checkpoint.path / "native"
    if checkpoint.backend == "hermesagent":
        return sorted((native / "sessions").glob("session_*.json"), key=lambda path: path.stat().st_mtime)
    if checkpoint.backend == "openclaw":
        return [native / "sessions" / "chat.jsonl"]
    if checkpoint.backend == "codex":
        return sorted((native / "sessions").rglob("*.jsonl"), key=lambda path: path.stat().st_mtime)
    if checkpoint.backend == "claudecode":
        return sorted(
            (path for path in (native / "projects").rglob("*.jsonl") if "subagents" not in path.parts),
            key=lambda path: path.stat().st_mtime,
        )
    raise ValueError(f"Unsupported checkpoint backend: {checkpoint.backend}")


def _selected_session_file(checkpoint: AgentCheckpoint) -> Path:
    files = _session_files(checkpoint)
    if not files:
        raise FileNotFoundError(f"Checkpoint contains no native session: {checkpoint.path}")
    if checkpoint.backend == "hermesagent":
        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("session_id") or "") == checkpoint.session_id:
                return path
    if checkpoint.backend == "claudecode":
        for path in files:
            if path.stem == checkpoint.session_id:
                return path
    return files[-1]


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                records.append(value)
    return records


def _rewrite_session_ids(value: Any, session_id: str) -> Any:
    if isinstance(value, list):
        return [_rewrite_session_ids(item, session_id) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten = {key: _rewrite_session_ids(item, session_id) for key, item in value.items()}
    if "sessionId" in rewritten:
        rewritten["sessionId"] = session_id
    return rewritten


def _is_session_header(record: dict[str, Any], backend: str) -> bool:
    record_type = str(record.get("type") or "")
    if backend == "openclaw":
        return record_type == "session"
    if backend == "codex":
        return record_type == "session_meta"
    return False


def _merge_jsonl(backend: str, checkpoints: list[AgentCheckpoint], destination: Path) -> str:
    merged: list[dict[str, Any]] = []
    first_session_id = checkpoints[0].session_id
    for index, checkpoint in enumerate(checkpoints):
        records = _jsonl_records(_selected_session_file(checkpoint))
        if not records:
            raise ValueError(f"Native session is empty: {checkpoint.path}")
        for record in records:
            if index > 0 and _is_session_header(record, backend):
                continue
            record = copy.deepcopy(record)
            if backend == "claudecode":
                record = _rewrite_session_ids(record, first_session_id)
            merged.append(record)

    if not merged:
        raise ValueError("Merged native session is empty")
    if backend == "codex":
        headers = [record for record in merged if record.get("type") == "session_meta"]
        if len(headers) != 1:
            raise ValueError(f"Merged Codex session must have one session_meta, got {len(headers)}")
        payload = headers[0].setdefault("payload", {})
        if isinstance(payload, dict):
            payload["id"] = first_session_id
    if backend == "openclaw":
        headers = [record for record in merged if record.get("type") == "session"]
        if len(headers) != 1:
            raise ValueError(f"Merged OpenClaw session must have one session header, got {len(headers)}")
        headers[0]["id"] = first_session_id

    if backend == "codex":
        relative = Path("native") / "sessions" / "merged" / f"rollout-{first_session_id}.jsonl"
    elif backend == "claudecode":
        relative = Path("native") / "projects" / "merged" / f"{first_session_id}.jsonl"
    else:
        relative = Path("native") / "sessions" / "chat.jsonl"
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return first_session_id


def _merge_hermes(checkpoints: list[AgentCheckpoint], destination: Path) -> str:
    payloads = []
    for checkpoint in checkpoints:
        path = _selected_session_file(checkpoint)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise ValueError(f"Invalid Hermes session payload: {path}")
        payloads.append(payload)
    merged = copy.deepcopy(payloads[0])
    messages: list[Any] = []
    for payload in payloads:
        messages.extend(copy.deepcopy(payload["messages"]))
    merged["messages"] = messages
    merged["message_count"] = len(messages)
    merged["session_id"] = checkpoints[0].session_id
    target = destination / "native" / "sessions" / f"session_{checkpoints[0].session_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checkpoints[0].session_id


def merge_checkpoints(
    checkpoints: list[AgentCheckpoint],
    destination: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentCheckpoint:
    """Merge ordered independent native sessions without another model call."""

    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    backend = checkpoints[0].backend
    if any(checkpoint.backend != backend for checkpoint in checkpoints):
        raise ValueError("Cannot merge checkpoints from different harnesses")
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if backend == "hermesagent":
        session_id = _merge_hermes(checkpoints, destination)
    else:
        session_id = _merge_jsonl(backend, checkpoints, destination)
    task_outcomes = [outcome for checkpoint in checkpoints if (outcome := _task_outcome(checkpoint))]
    if task_outcomes:
        (destination / "task_outcomes.json").write_text(
            json.dumps(task_outcomes, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return AgentCheckpoint.create(
        backend=backend,
        path=destination,
        session_id=session_id,
        metadata={
            **dict(metadata or {}),
            "checkpoint_variant": "independent_learning_archive",
            "source_checkpoint_payload_sha256": [checkpoint.payload_sha256 for checkpoint in checkpoints],
            "task_outcomes": task_outcomes,
        },
    )
