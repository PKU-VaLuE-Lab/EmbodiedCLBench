from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


CHECKPOINT_MANIFEST_NAME = "agent_checkpoint.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_payload_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    payload_root = root / "native"
    if not payload_root.is_dir():
        raise FileNotFoundError(f"Checkpoint native payload is missing: {payload_root}")
    records: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(item for item in payload_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(payload_root).as_posix()
        file_digest = _sha256_file(path)
        size = path.stat().st_size
        records.append({"path": relative_path, "size": size, "sha256": file_digest})
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(size).encode("ascii"))
        combined.update(b"\0")
        combined.update(file_digest.encode("ascii"))
        combined.update(b"\n")
    if not records:
        raise ValueError(f"Checkpoint native payload is empty: {payload_root}")
    return combined.hexdigest(), records


@dataclass(frozen=True)
class AgentCheckpoint:
    backend: str
    path: Path
    session_id: str
    payload_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        backend: str,
        path: Path,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> "AgentCheckpoint":
        checkpoint_path = path.resolve()
        payload_sha256, files = checkpoint_payload_digest(checkpoint_path)
        manifest = {
            "schema_version": "tongbench_agent_checkpoint_v1",
            "backend": backend,
            "session_id": session_id,
            "payload_sha256": payload_sha256,
            "files": files,
            "metadata": dict(metadata or {}),
        }
        (checkpoint_path / CHECKPOINT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return cls(
            backend=backend,
            path=checkpoint_path,
            session_id=session_id,
            payload_sha256=payload_sha256,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def load(cls, path: Path, *, expected_backend: str | None = None) -> "AgentCheckpoint":
        checkpoint_path = path.resolve()
        manifest_path = checkpoint_path / CHECKPOINT_MANIFEST_NAME
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend = str(payload.get("backend") or "")
        if expected_backend and backend != expected_backend:
            raise ValueError(
                f"Checkpoint backend mismatch: expected {expected_backend}, got {backend or '(empty)'}"
            )
        actual_digest, actual_files = checkpoint_payload_digest(checkpoint_path)
        expected_digest = str(payload.get("payload_sha256") or "")
        if actual_digest != expected_digest:
            raise ValueError(
                f"Checkpoint payload hash mismatch: expected {expected_digest}, got {actual_digest}"
            )
        if actual_files != payload.get("files"):
            raise ValueError("Checkpoint file manifest does not match the native payload.")
        return cls(
            backend=backend,
            path=checkpoint_path,
            session_id=str(payload.get("session_id") or ""),
            payload_sha256=actual_digest,
            metadata=dict(payload.get("metadata") or {}),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "path": str(self.path),
            "session_id": self.session_id,
            "payload_sha256": self.payload_sha256,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AgentPostTaskResult:
    prompt: str
    final_response: str = ""
    completed: bool = False
    api_calls: int = 0
    error: str | None = None

    @classmethod
    def from_payload(
        cls,
        prompt: str,
        payload: dict[str, Any] | None,
        *,
        error: str | None = None,
    ) -> "AgentPostTaskResult":
        data = payload if isinstance(payload, dict) else {}
        return cls(
            prompt=str(data.get("prompt") or prompt),
            final_response=str(data.get("final_response") or "").strip(),
            completed=bool(data.get("completed", not error)),
            api_calls=int(data.get("api_calls") or 0),
            error=str(data.get("error") or error or "").strip() or None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "final_response": self.final_response,
            "completed": self.completed,
            "api_calls": self.api_calls,
            "error": self.error,
        }


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: str
    task: dict[str, Any]
    workspace_path: str
    prompt: str
    timeout_seconds: int
    output_dir: Path
    model: str
    thinking: str | None = None
    models_config: dict[str, Any] | None = None
    lobster: dict[str, Any] | None = None
    runtime_options: dict[str, Any] | None = None
    post_task_prompt: str | None = None
    resume_checkpoint: AgentCheckpoint | None = None
    resume_without_prompt: bool = False
    tools_enabled: bool = True


@dataclass
class AgentExecution:
    elapsed_time: float
    error: str | None = None
    gateway_proc: subprocess.Popen[str] | None = None
    agent_proc: subprocess.Popen[str] | None = None
    post_task_result: AgentPostTaskResult | None = None
    final_response: str = ""


class BaseAgent(ABC):
    @property
    @abstractmethod
    def expects_gateway(self) -> bool:
        """Whether this backend starts a long-running gateway process."""

    @property
    @abstractmethod
    def transcript_container_path(self) -> str:
        """Path to chat transcript inside the runtime container."""

    def prepare_grading_transcript(self, task_id: str) -> str:
        """Prepare and return the transcript path used for grading."""
        _ = task_id
        return self.transcript_container_path

    @abstractmethod
    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        """Execute a task and return process handles, timing and error state."""

    @abstractmethod
    def export_checkpoint(
        self,
        task_id: str,
        checkpoint_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentCheckpoint:
        """Export a resumable native session before the runtime container is removed."""

    @abstractmethod
    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        """Collect token usage and cost for one task."""
