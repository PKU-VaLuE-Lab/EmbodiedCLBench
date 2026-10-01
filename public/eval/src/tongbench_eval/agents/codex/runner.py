from __future__ import annotations

import json
import hashlib
import logging
import os
import shlex
import socket
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tongbench_eval.agents.base import (
    AgentCheckpoint,
    AgentExecution,
    AgentPostTaskResult,
    AgentTaskSpec,
    BaseAgent,
)
from tongbench_eval.agents.codex.skill_text import extract_skill_text
from tongbench_eval.agents.codex.backend import (
    CODEX_PROMPT_PATH,
    load_skill_documents,
    prepare_codex_prompt,
)
from tongbench_eval.utils.docker_utils import (
    build_bind_mount_args,
    build_host_gateway_args,
    run_warmup,
    setup_skills,
    snapshot_workspace_state,
)
from tongbench_eval.utils.native_runtime import is_native_runtime, native_path
from tongbench_eval.utils.endpoint_utils import normalize_openrouter_base_url_for_openclaw

logger = logging.getLogger(__name__)

CODEX_HOME = "/root/.codex"
CODEX_SESSIONS_DIR = f"{CODEX_HOME}/sessions"
CODEX_CONFIG_PATH = f"{CODEX_HOME}/config.toml"
CODEX_SKILLS_DIR = f"{CODEX_HOME}/skills"
CODEX_PROVIDER_ENV_KEY = "CODEX_API_KEY"
CODEX_BASE_URL_ENV_KEY = "CODEX_BASE_URL"
CODEX_DASHSCOPE_ADAPTER_PATH = "/tmp/codex_dashscope_responses_adapter.py"
CODEX_DASHSCOPE_ADAPTER_LOG = "/tmp/codex_dashscope_responses_adapter.log"
CODEX_PROTOCOL_GATEWAY_PATH = "/tmp/tongbench_protocol_gateway.py"
CODEX_PROTOCOL_GATEWAY_LOG = "/tmp/tongbench_protocol_gateway.log"
CODEX_DASHSCOPE_ADAPTER_PORT = 18780
MCP_WHEEL_HOST_DIR = "/tmp/tongbench_mcp_wheel"
MCP_WHEEL_CONTAINER_DIR = "/opt/tongbench_mcp_wheel"
OPENCLAW_TRANSCRIPT_DIR = "/root/.openclaw/agents/main/sessions"
OPENCLAW_TRANSCRIPT_PATH = f"{OPENCLAW_TRANSCRIPT_DIR}/chat.jsonl"
DEFAULT_REASONING_EFFORT = "medium" #"high"
CODEX_LOG_NOISE_MARKERS = (
    "ReasoningRawContentDelta without active item",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_agent_log_event(output_dir: Path, event: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched = {"timestamp": _now_iso(), **event}
    with (output_dir / "agent.log").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def write_execution_status(output_dir: Path, **updates: Any) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "execution_status.json"
    status: dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    status.update(updates)
    status["updated_at"] = _now_iso()
    status_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return status


def initialize_host_run_artifacts(
    output_dir: Path,
    task_id: str,
    model: str,
    timeout_seconds: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agent.log").touch(exist_ok=True)
    write_execution_status(
        output_dir,
        task_id=task_id,
        model=model,
        timeout_seconds=timeout_seconds,
        status="created",
        started_at=_now_iso(),
        timed_out=False,
        exit_code=None,
        error=None,
    )
    append_agent_log_event(
        output_dir,
        {
            "type": "runner.status",
            "stage": "created",
            "message": "Host-side run artifacts initialized before container startup.",
        },
    )


def sanitize_agent_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    kept = [
        line for line in lines
        if not any(marker in line for marker in CODEX_LOG_NOISE_MARKERS)
    ]
    if len(kept) != len(lines):
        log_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


class CodexAgent(BaseAgent):
    def __init__(
        self,
        image: str | None = None,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "",
        reasoning_effort_default: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        resolved_image = image or os.environ.get("DOCKER_IMAGE_CODEX") or "wildclawbench-codex-ubuntu:v0.0"
        self.image: str = resolved_image
        self.openrouter_api_key = (
            openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        ).strip()
        self.openrouter_base_url = normalize_openrouter_base_url_for_openclaw(
            openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL", "")
        )
        self.reasoning_effort_default = reasoning_effort_default

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return CODEX_SESSIONS_DIR

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        elapsed_time = float(spec.timeout_seconds)
        start_time = time.perf_counter()
        task_id = spec.task_id
        runtime_api_key, runtime_base_url = self._resolve_runtime_provider(
            spec.model,
            spec.models_config,
        )
        initialize_host_run_artifacts(
            output_dir=spec.output_dir,
            task_id=task_id,
            model=spec.model,
            timeout_seconds=spec.timeout_seconds,
        )

        try:
            try:
                write_execution_status(spec.output_dir, status="starting_container")
                self._start_container(
                    task_id,
                    spec.workspace_path,
                    spec.task,
                    spec.lobster,
                    runtime_api_key=runtime_api_key,
                    runtime_base_url=runtime_base_url,
                )
                write_execution_status(spec.output_dir, status="container_started")
                write_execution_status(spec.output_dir, status="preparing_workspace")
                self._prepare_workspace(task_id, spec.workspace_path)
                self._ensure_python_mcp_runtime(task_id, spec.runtime_options)
                skills_text = spec.task.get("skills", "") if spec.task else ""
                skills_path = spec.task.get("skills_path", "") if spec.task else ""
                setup_skills(
                    task_id,
                    skills_text,
                    skills_path,
                    container_skills_root=CODEX_SKILLS_DIR,
                )
                skill_docs = load_skill_documents(
                    skills_text,
                    skills_path,
                    container_skill_root=CODEX_SKILLS_DIR,
                )
                run_warmup(
                    task_id,
                    spec.task.get("warmup", "") if spec.task else "",
                    detach_background=True,
                )
                codex_options = (
                    spec.runtime_options.get("codex", {})
                    if isinstance(spec.runtime_options, dict)
                    else {}
                )
                adapter_base_url = self._maybe_start_dashscope_responses_adapter(
                    task_id=task_id,
                    output_dir=spec.output_dir,
                    model=spec.model,
                    base_url=runtime_base_url,
                    api_key=runtime_api_key,
                    thinking=spec.thinking,
                    codex_options=codex_options,
                )
                codex_base_url = adapter_base_url or runtime_base_url
                self._write_codex_config(
                    task_id=task_id,
                    model=spec.model,
                    reasoning_effort=spec.thinking
                    or self._default_reasoning_effort_for_model(spec.model),
                    wire_api=self._default_wire_api_for_model(spec.model),
                    output_dir=spec.output_dir,
                    base_url=codex_base_url,
                    runtime_options=spec.runtime_options,
                )
                resume_checkpoint = None
                if spec.resume_checkpoint is not None:
                    resume_checkpoint = AgentCheckpoint.load(
                        spec.resume_checkpoint.path,
                        expected_backend="codex",
                    )
                    self._install_resume_checkpoint(task_id, resume_checkpoint)
                disable_image_helper = (
                    isinstance(codex_options, dict)
                    and bool(codex_options.get("disable_image_helper", False))
                )
                image_helper_enabled = (
                    not disable_image_helper
                    and self._should_enable_image_helper(spec.prompt, spec.workspace_path)
                )
                if image_helper_enabled:
                    self._install_image_helper(task_id, spec.model)
                snapshot_workspace_state(task_id)
                write_execution_status(spec.output_dir, status="codex_running")
                if resume_checkpoint is None:
                    task_prompt = self._build_task_prompt(
                        spec.prompt,
                        image_helper_enabled=image_helper_enabled,
                        skill_docs=skill_docs,
                    )
                    self._run_prompt(
                        task_id=task_id,
                        prompt=task_prompt,
                        timeout_seconds=spec.timeout_seconds,
                        output_dir=spec.output_dir,
                    )
                    final_response = ""
                else:
                    final_response = self._run_resumed_prompt(
                        task_id=task_id,
                        session_id=resume_checkpoint.session_id,
                        prompt=spec.prompt,
                        timeout_seconds=spec.timeout_seconds,
                        output_dir=spec.output_dir,
                        tools_enabled=bool(spec.tools_enabled),
                    )
                post_task_result = self._run_post_task_prompt(
                    task_id=task_id,
                    post_task_prompt=spec.post_task_prompt,
                    timeout_seconds=spec.timeout_seconds,
                    output_dir=spec.output_dir,
                )
                elapsed_time = time.perf_counter() - start_time
                write_execution_status(
                    spec.output_dir,
                    status="finished",
                    timed_out=False,
                    elapsed_time=round(elapsed_time, 2),
                    exit_code=0,
                )
                return AgentExecution(
                    elapsed_time=elapsed_time,
                    error=None,
                    gateway_proc=None,
                    agent_proc=None,
                    post_task_result=post_task_result,
                    final_response=final_response,
                )
            except subprocess.TimeoutExpired:
                logger.info("[%s] Codex timed out...", task_id)
                elapsed_time = float(spec.timeout_seconds)
                append_agent_log_event(
                    spec.output_dir,
                    {
                        "type": "runner.timeout",
                        "message": f"Codex timed out after {spec.timeout_seconds} seconds.",
                        "timeout_seconds": spec.timeout_seconds,
                        "elapsed_time": elapsed_time,
                    },
                )
                write_execution_status(
                    spec.output_dir,
                    status="timed_out",
                    timed_out=True,
                    elapsed_time=round(elapsed_time, 2),
                    error="Codex run timed out",
                )
                return AgentExecution(
                    elapsed_time=elapsed_time,
                    error="Codex run timed out",
                    gateway_proc=None,
                    agent_proc=None,
                )
            except Exception as exc:
                elapsed_time = time.perf_counter() - start_time
                logger.error("[%s] Codex execution error: %s", task_id, exc)
                append_agent_log_event(
                    spec.output_dir,
                    {
                        "type": "runner.error",
                        "stage": "codex_execution",
                        "message": str(exc),
                        "elapsed_time": round(elapsed_time, 2),
                    },
                )
                write_execution_status(
                    spec.output_dir,
                    status="error",
                    error=str(exc),
                    elapsed_time=round(elapsed_time, 2),
                )
                return AgentExecution(
                    elapsed_time=elapsed_time,
                    error=str(exc),
                    gateway_proc=None,
                    agent_proc=None,
                )
        finally:
            sanitize_agent_log(spec.output_dir / "agent.log")
            self._copy_dashscope_adapter_log(task_id, spec.output_dir)
            self._copy_protocol_gateway_log(task_id, spec.output_dir)
            try:
                self._install_openclaw_transcript_shim(task_id, spec.output_dir)
            except Exception as exc:
                logger.warning("[%s] OpenClaw transcript shim failed: %s", task_id, exc)

    def export_checkpoint(
        self,
        task_id: str,
        checkpoint_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentCheckpoint:
        checkpoint_path = checkpoint_dir.resolve()
        if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
        native_dir = checkpoint_path / "native" / "sessions"
        native_dir.mkdir(parents=True, exist_ok=True)
        copied = subprocess.run(
            ["docker", "cp", f"{task_id}:{CODEX_SESSIONS_DIR}/.", str(native_dir)],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Codex checkpoint export failed: {copied.stderr.strip()}")
        session_files = sorted(
            native_dir.rglob("*.jsonl"),
            key=lambda path: path.stat().st_mtime,
        )
        if not session_files:
            raise FileNotFoundError("Codex checkpoint contains no native rollout JSONL.")
        session_id = self._codex_session_id_from_rollout(session_files[-1])
        if not session_id:
            raise ValueError(f"Codex rollout has no session_meta id: {session_files[-1]}")
        return AgentCheckpoint.create(
            backend="codex",
            path=checkpoint_path,
            session_id=session_id,
            metadata=metadata,
        )

    @staticmethod
    def _codex_session_id_from_rollout(path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "session_meta":
                    continue
                payload = event.get("payload")
                if isinstance(payload, dict):
                    return str(payload.get("id") or "").strip()
        return ""

    @staticmethod
    def _install_resume_checkpoint(task_id: str, checkpoint: AgentCheckpoint) -> None:
        sessions_dir = checkpoint.path / "native" / "sessions"
        mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", CODEX_SESSIONS_DIR],
            capture_output=True,
            text=True,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(f"Codex checkpoint restore mkdir failed: {mkdir.stderr.strip()}")
        copied = subprocess.run(
            ["docker", "cp", f"{sessions_dir}/.", f"{task_id}:{CODEX_SESSIONS_DIR}/"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Codex checkpoint restore failed: {copied.stderr.strip()}")

    def collect_usage(
        self, task_id: str, output_dir: Path, elapsed_time: float
    ) -> dict[str, Any]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
            "elapsed_time": round(elapsed_time, 2),
        }
        output_dir.mkdir(parents=True, exist_ok=True)

        sessions_dest = output_dir / "codex_sessions"
        sessions_dest.mkdir(parents=True, exist_ok=True)
        self._copy_dir_from_container(task_id, f"{CODEX_SESSIONS_DIR}/.", sessions_dest)

        latest = self._find_latest_session(task_id)
        chat_dest = output_dir / "chat.jsonl"
        if latest:
            self._copy_file_from_container(
                task_id, f"{CODEX_SESSIONS_DIR}/{latest}", chat_dest
            )

        parsed = self._extract_usage_from_jsonl(chat_dest)
        if parsed["total_tokens"] == 0 and parsed["input_tokens"] == 0:
            parsed = self._extract_usage_from_session_dir(sessions_dest)

        if parsed["cost_usd"] == 0.0:
            parsed["cost_usd"] = round(self._estimate_cost(parsed), 6)

        usage.update(parsed)
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def _start_container(
        self,
        task_id: str,
        workspace_path: str,
        task: dict[str, Any],
        lobster: dict[str, Any] | None,
        runtime_api_key: str = "",
        runtime_base_url: str = "",
    ) -> None:
        workspace = Path(workspace_path).expanduser()
        if not workspace.is_dir():
            raise RuntimeError(
                f"Workspace path does not exist or is not a directory: {workspace}"
            )
        exec_path = workspace / "exec"
        if not exec_path.is_dir():
            raise RuntimeError(
                f"Workspace exec directory does not exist or is not a directory: {exec_path}"
            )

        proxy_http = os.environ.get("HTTP_PROXY_INNER", "").strip()
        proxy_https = os.environ.get("HTTPS_PROXY_INNER", "").strip()
        no_proxy_values = [
            value.strip()
            for value in os.environ.get("NO_PROXY_INNER", "").split(",")
            if value.strip()
        ]
        for local_host in ("127.0.0.1", "localhost"):
            if local_host not in no_proxy_values:
                no_proxy_values.append(local_host)
        no_proxy = ",".join(no_proxy_values)
        runtime_api_key = (runtime_api_key or "").strip()
        runtime_base_url = (runtime_base_url or self.openrouter_base_url).strip()
        openrouter_api_key = (
            runtime_api_key
            if self._is_openrouter_base_url(runtime_base_url)
            else self.openrouter_api_key
        )
        openrouter_base_url = (
            runtime_base_url
            if self._is_openrouter_base_url(runtime_base_url)
            else self.openrouter_base_url
        )
        native_bundle = os.environ.get("TONGBENCH_NATIVE_BUNDLE_ROOT", "").strip()
        cli_path = (
            f"{native_bundle}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            if is_native_runtime() and native_bundle
            else "/root/miniconda3/envs/eval/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        env_map: dict[str, str] = {
            "OPENROUTER_API_KEY": openrouter_api_key,
            "OPENROUTER_BASE_URL": openrouter_base_url,
            CODEX_PROVIDER_ENV_KEY: runtime_api_key,
            CODEX_BASE_URL_ENV_KEY: runtime_base_url,
            "OPENROUTER_IMAGE_MODEL": os.environ.get("OPENROUTER_IMAGE_MODEL", "").strip(),
            "WILDCLAW_IMAGE_MODEL": os.environ.get("WILDCLAW_IMAGE_MODEL", "").strip(),
            "BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", ""),
            "http_proxy": proxy_http,
            "https_proxy": proxy_https,
            "HTTP_PROXY": proxy_http,
            "HTTPS_PROXY": proxy_https,
            "no_proxy": no_proxy,
            "NO_PROXY": no_proxy,
            "PATH": cli_path,
        }

        env_args: list[str] = []
        for key, value in env_map.items():
            if value:
                env_args += ["-e", f"{key}={value}"]

        extra_env = task.get("env", "") if task else ""
        for line in extra_env.splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "").strip()
            env_args += ["-e", f"{key}={value}"]
            masked = (value[:4] + "***") if value else "(empty)"
            logger.info("[%s] Injecting env var: %s=%s", task_id, key, masked)

        for key in (lobster or {}).get("env", []) or []:
            value = os.environ.get(key, "").strip()
            if not value:
                logger.warning(
                    "[%s] Lobster env key %s not found, skipping", task_id, key
                )
                continue
            env_args += ["-e", f"{key}={value}"]
            logger.info("[%s] Injecting lobster env: %s=%s***", task_id, key, value[:4])

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            task_id,
            *build_host_gateway_args(),
            *env_args,
            "-v",
            f"{exec_path}:/workspace:ro",
            *(["-v", f"{MCP_WHEEL_HOST_DIR}:{MCP_WHEEL_CONTAINER_DIR}:ro"]
              if Path(MCP_WHEEL_HOST_DIR).is_dir() else []),
            *build_bind_mount_args(task.get("bind_mounts") if task else None),
            self.image,
            "/bin/bash",
            "-c",
            "tail -f /dev/null",
        ]
        logger.info("[%s] Starting Codex container (%s)", task_id, self.image)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Codex container startup failed:\n{r.stderr}")
        logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])

    def _prepare_workspace(self, task_id: str, workspace_path: str) -> None:
        r = subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "/bin/bash",
                "-c",
                (
                    "mkdir -p /tmp_workspace "
                    f"&& mkdir -p {CODEX_HOME} {CODEX_SESSIONS_DIR} "
                    "&& cp -r /workspace/. /tmp_workspace "
                    "&& chmod -R u+w /tmp_workspace"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Codex workspace copy failed:\n{r.stderr}")

        if is_native_runtime():
            # Codex reads the MCP command from the host-side config in native
            # mode. Verify that the files copied through the Docker shim are
            # present at the exact host paths referenced by that config.
            # This is limited to the Codex native runner and is a no-op in
            # Docker mode.
            native_workspace = Path(native_path(task_id, "/tmp_workspace"))
            source_workspace = Path(workspace_path) / "exec"
            native_workspace.mkdir(parents=True, exist_ok=True)
            for filename in ("tongsim_mcp_server.py",):
                source = source_workspace / filename
                destination = native_workspace / filename
                if not source.is_file():
                    raise RuntimeError(
                        f"Native Codex MCP source file is missing: {source}"
                    )
                # Always refresh the native copy. A task root can be reused
                # after a partial startup, and a stale file must not hide a
                # failed workspace copy.
                shutil.copy2(source, destination)
                if not destination.is_file():
                    raise RuntimeError(
                        f"Native Codex MCP file was not materialized: {destination}"
                    )
            session_files = (".tongsim_session.json", ".tongsim_sequence_session.json")
            copied_session = False
            for filename in session_files:
                source = source_workspace / filename
                if not source.is_file():
                    continue
                destination = native_workspace / filename
                shutil.copy2(source, destination)
                if not destination.is_file():
                    raise RuntimeError(
                        f"Native Codex MCP session file was not materialized: {destination}"
                    )
                copied_session = True
            if not copied_session:
                expected = ", ".join(str(source_workspace / name) for name in session_files)
                raise RuntimeError(
                    f"Native Codex MCP session file is missing; checked: {expected}"
                )
            logger.info(
                "[%s] Native Codex MCP files ready at %s",
                task_id,
                native_workspace,
            )

        tmp_path = Path(workspace_path) / "tmp"
        if tmp_path.exists():
            mkdir_tmp = subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"],
                capture_output=True,
                text=True,
            )
            if mkdir_tmp.returncode != 0:
                raise RuntimeError(f"Codex tmp mkdir failed:\n{mkdir_tmp.stderr}")

            copied = subprocess.run(
                ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"Codex tmp copy failed:\n{copied.stderr}")

    def _ensure_python_mcp_runtime(
        self,
        task_id: str,
        runtime_options: dict[str, Any] | None,
    ) -> None:
        mcp_servers = self._runtime_mcp_servers(runtime_options)
        if not mcp_servers:
            return
        if os.environ.get("CODEX_SKIP_MCP_PYTHON_INSTALL", "").strip() in {"1", "true", "yes"}:
            logger.info("[%s] Skipping Codex Python MCP dependency check by env override", task_id)
            return

        python_commands = {
            str(server.get("command") or "").strip()
            for server in mcp_servers.values()
            if isinstance(server, dict)
            and "python" in str(server.get("command") or "").lower()
        }
        for python_command in sorted(command for command in python_commands if command):
            if is_native_runtime():
                container_python = python_command
                shell_prefix = ""
            else:
                container_python = os.environ.get(
                    "TONGBENCH_CONTAINER_MCP_PYTHON_ABS",
                    "/root/miniconda3/envs/eval/bin/python3",
                )
                shell_prefix = "export PATH=/root/miniconda3/envs/eval/bin:$PATH; "
            probe = (
                f"{shell_prefix}{shlex.quote(container_python)} - <<'PY'\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "PY"
            )
            probe_result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c", probe],
                capture_output=True,
                text=True,
            )
            if probe_result.returncode == 0:
                continue
            if Path(MCP_WHEEL_HOST_DIR).is_dir():
                install = (
                    f"{shell_prefix}{shlex.quote(container_python)} -m pip install --quiet --no-index "
                    f"--find-links {MCP_WHEEL_CONTAINER_DIR} 'mcp==1.16.0'"
                )
            else:
                install = (
                    f"{shell_prefix}{shlex.quote(container_python)} -m pip install --quiet "
                    "'mcp==1.16.0'"
                )
            install_result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c", install],
                capture_output=True,
                text=True,
            )
            if install_result.returncode != 0:
                raise RuntimeError(
                    "Codex MCP server dependency is missing and automatic install failed. "
                    f"Command: {python_command} -m pip install 'mcp==1.16.0'\n"
                    f"{install_result.stderr or install_result.stdout}"
                )

    def _default_reasoning_effort_for_model(self, model: str) -> str | None:
        """Return an explicit reasoning override if one is configured.

        By default we let Codex CLI and the underlying model choose their
        native reasoning settings. The only automatic override we keep is the
        explicit ``CODEX_REASONING_EFFORT`` env knob, which is useful for
        controlled experiments or emergency rollouts.
        """
        override = os.environ.get("CODEX_REASONING_EFFORT", "").strip().lower()
        if override:
            return override
        return None

    def _default_wire_api_for_model(self, model: str) -> str | None:
        """Return an explicit wire API override.

        Codex CLI v0.121 rejects the legacy chat wire protocol. Keep the
        provider config explicit so OpenAI-compatible providers do not depend
        on CLI defaults drifting between releases.
        """
        _ = model
        override = os.environ.get("CODEX_WIRE_API", "").strip().lower()
        if override and override != "responses":
            logger.warning(
                "Ignoring unsupported CODEX_WIRE_API=%s; Codex CLI currently requires responses.",
                override,
            )
        return "responses"

    def _resolve_runtime_provider(
        self,
        model: str,
        models_config: dict | None,
    ) -> tuple[str, str]:
        api_key = self.openrouter_api_key
        base_url = self.openrouter_base_url
        config_key, config_base_url = self._resolve_provider_config(model, models_config)
        if config_key:
            api_key = config_key
        if config_base_url:
            base_url = config_base_url
        return api_key, base_url

    @staticmethod
    def _resolve_provider_config(
        model: str,
        models_config: dict | None,
    ) -> tuple[str, str]:
        if not models_config:
            return "", ""
        providers = models_config.get("providers", {})
        for _provider_name, provider in providers.items():
            if not isinstance(provider, dict):
                continue
            for entry in provider.get("models", []) or []:
                if isinstance(entry, dict) and entry.get("id") == model:
                    return provider.get("apiKey", ""), provider.get("baseUrl", "")
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return first.get("apiKey", ""), first.get("baseUrl", "")
        return "", ""

    @staticmethod
    def _is_openrouter_base_url(base_url: str) -> bool:
        return "openrouter.ai" in (base_url or "").lower()

    @staticmethod
    def _is_minimax_model(model: str) -> bool:
        bare_model = model.split("/", 1)[1] if model.startswith("openrouter/") else model
        return bare_model.lower().startswith("minimax/")

    @staticmethod
    def _is_dashscope_base_url(base_url: str) -> bool:
        return "dashscope.aliyuncs.com" in (base_url or "").lower()

    @staticmethod
    def _is_glm_model(model: str) -> bool:
        return str(model or "").strip().lower().startswith("zhipu/glm-")

    def _maybe_start_dashscope_responses_adapter(
        self,
        *,
        task_id: str,
        output_dir: Path,
        model: str,
        base_url: str,
        api_key: str,
        thinking: str | None,
        codex_options: Any,
    ) -> str | None:
        if not isinstance(codex_options, dict):
            return None
        if not bool(codex_options.get("dashscope_responses_adapter", False)):
            return None
        if not self._is_dashscope_base_url(base_url):
            return None

        use_protocol_gateway = self._is_glm_model(model)
        if use_protocol_gateway:
            self._install_protocol_gateway(task_id)
            adapter_path = CODEX_PROTOCOL_GATEWAY_PATH
            adapter_log = CODEX_PROTOCOL_GATEWAY_LOG
        else:
            self._install_dashscope_responses_adapter(task_id)
            adapter_path = CODEX_DASHSCOPE_ADAPTER_PATH
            adapter_log = CODEX_DASHSCOPE_ADAPTER_LOG
        adapter_port = self._allocate_dashscope_adapter_port(task_id)
        local_base_url = f"http://127.0.0.1:{adapter_port}/v1"
        env_items: list[str] = []
        if use_protocol_gateway:
            env_items.extend([
                f"TONGBENCH_GATEWAY_BASE_URL={shlex.quote(base_url)}",
                f"TONGBENCH_GATEWAY_API_KEY={shlex.quote(api_key)}",
                f"TONGBENCH_GATEWAY_MODEL={shlex.quote(model)}",
                f"TONGBENCH_GATEWAY_REASONING_EFFORT={shlex.quote(thinking or '')}",
                "TONGBENCH_GATEWAY_SINGLE_TONGSIM_CALL=1",
            ])
            if bool(codex_options.get("mcp_only_tools", False)):
                env_items.append("TONGBENCH_GATEWAY_MCP_ONLY_TOOLS=1")
        elif bool(codex_options.get("mcp_only_tools", False)):
            env_items.append("TONGSIM_CODEX_MCP_ONLY_TOOLS=1")
        env_prefix = ("env " + " ".join(env_items) + " ") if env_items else ""
        start_cmd = (
            f"nohup {env_prefix}python3 {shlex.quote(adapter_path)} "
            f"--host 127.0.0.1 --port {adapter_port} "
            f"> {shlex.quote(adapter_log)} 2>&1 & echo $!"
        )
        result = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", start_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Codex DashScope Responses adapter failed to start:\n"
                f"{result.stderr or result.stdout}"
            )

        health_url = f"{local_base_url}/health"
        health_cmd = (
            "python3 - <<'PY'\n"
            "import sys, urllib.request\n"
            f"url = {json.dumps(health_url)}\n"
            "try:\n"
            "    print(urllib.request.urlopen(url, timeout=1).read().decode('utf-8'))\n"
            "except Exception as exc:\n"
            "    print(exc, file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "PY"
        )
        last_error = ""
        for _attempt in range(20):
            health = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", health_cmd],
                capture_output=True,
                text=True,
            )
            if health.returncode == 0:
                append_agent_log_event(
                    output_dir,
                    {
                        "type": (
                            "codex.protocol_gateway.started"
                            if use_protocol_gateway
                            else "codex.dashscope_adapter.started"
                        ),
                        "base_url": local_base_url,
                        "target_base_url": base_url,
                        "target_model": model,
                        "reasoning_effort": thinking,
                        "mcp_only_tools": bool(codex_options.get("mcp_only_tools", False)),
                    },
                )
                return local_base_url
            last_error = health.stderr or health.stdout
            time.sleep(0.2)

        if use_protocol_gateway:
            self._copy_protocol_gateway_log(task_id, output_dir)
        else:
            self._copy_dashscope_adapter_log(task_id, output_dir)
        raise RuntimeError(
            "Codex DashScope Responses adapter did not become healthy:\n"
            f"{last_error}"
        )

    def _install_protocol_gateway(self, task_id: str) -> None:
        source = (Path(__file__).parent.parent / "protocol_gateway.py").resolve()
        if not source.is_file():
            raise RuntimeError(f"Missing protocol gateway source: {source}")
        copied = subprocess.run(
            ["docker", "cp", str(source), f"{task_id}:{CODEX_PROTOCOL_GATEWAY_PATH}"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(
                "Codex protocol gateway copy failed:\n"
                f"{copied.stderr or copied.stdout}"
            )

    @staticmethod
    def _allocate_dashscope_adapter_port(task_id: str) -> int:
        """Choose a per-task loopback port so parallel Codex runs do not collide."""

        # Use a task-specific range so parallel processes do not race for the
        # legacy shared port. The task id includes a per-run nonce, making a
        # collision between concurrent tasks unlikely; probing handles ports
        # already occupied by unrelated processes.
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        first_candidate = 20000 + (int.from_bytes(digest[:4], "big") % 30000)
        candidates = [
            20000 + ((first_candidate - 20000 + offset) % 30000)
            for offset in range(30000)
        ]
        for port in candidates:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError("Unable to allocate a loopback port for the Codex Responses adapter.")

    def _install_dashscope_responses_adapter(self, task_id: str) -> None:
        adapter = self._render_dashscope_responses_adapter()
        adapter_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(adapter)
                adapter_tmp = f.name

            copied = subprocess.run(
                ["docker", "cp", adapter_tmp, f"{task_id}:{CODEX_DASHSCOPE_ADAPTER_PATH}"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(
                    "Codex DashScope Responses adapter copy failed:\n"
                    f"{copied.stderr or copied.stdout}"
                )

            chmod = subprocess.run(
                ["docker", "exec", task_id, "chmod", "+x", CODEX_DASHSCOPE_ADAPTER_PATH],
                capture_output=True,
                text=True,
            )
            if chmod.returncode != 0:
                raise RuntimeError(
                    "Codex DashScope Responses adapter chmod failed:\n"
                    f"{chmod.stderr or chmod.stdout}"
                )
        finally:
            if adapter_tmp:
                Path(adapter_tmp).unlink(missing_ok=True)

    def _copy_dashscope_adapter_log(self, task_id: str, output_dir: Path) -> None:
        try:
            self._copy_file_from_container(
                task_id,
                CODEX_DASHSCOPE_ADAPTER_LOG,
                output_dir / "codex_dashscope_responses_adapter.log",
            )
        except Exception:
            pass

    def _copy_protocol_gateway_log(self, task_id: str, output_dir: Path) -> None:
        try:
            self._copy_file_from_container(
                task_id,
                CODEX_PROTOCOL_GATEWAY_LOG,
                output_dir / "protocol_gateway.log",
            )
        except Exception:
            pass

    @staticmethod
    def _render_dashscope_responses_adapter() -> str:
        return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
import json
import os
import sys
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

TONGSIM_MCP_TOOL_NAMES = {
    "observe",
    "select_option",
    "status",
    "observe_current_state",
    "choose_action",
    "check_task_status",
}


def _target_url(target_base_url: str, path: str) -> str:
    suffix = path or "/responses"
    if suffix.startswith("/v1/"):
        suffix = suffix[3:]
    elif suffix == "/v1":
        suffix = ""
    if not suffix:
        suffix = "/responses"
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    return target_base_url.rstrip("/") + suffix


def _mcp_only_tools_enabled() -> bool:
    return os.environ.get("TONGSIM_CODEX_MCP_ONLY_TOOLS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _adapter_debug_enabled() -> bool:
    return os.environ.get("TONGSIM_CODEX_ADAPTER_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    name = tool.get("name")
    if name:
        return str(name)
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return ""


def _is_tongsim_mcp_tool(tool: Any) -> bool:
    name = _tool_name(tool)
    if name.startswith("mcp__"):
        return True
    return name in TONGSIM_MCP_TOOL_NAMES


def _tool_call_name(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    name = item.get("name") or item.get("tool_name")
    if name:
        return str(name)
    function = item.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return ""


def _is_tongsim_tool_call_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = str(item.get("type") or "").lower().replace("-", "_")
    if item_type not in {"function_call", "tool_call"}:
        return False

    name = _tool_call_name(item)
    if name in TONGSIM_MCP_TOOL_NAMES:
        return True
    if name.startswith("mcp__"):
        return True

    namespace = str(item.get("namespace") or "")
    if namespace.startswith("mcp__") and name in TONGSIM_MCP_TOOL_NAMES:
        return True
    return False


def _item_id(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get("id") or item.get("item_id") or item.get("call_id") or item.get("callId")
    return str(value) if value is not None else ""


def _output_index(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    value = data.get("output_index")
    if value is None:
        value = data.get("index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _log_preview(value: Any, limit: int = 1000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    except Exception:
        text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _tool_call_arguments(item: Any) -> Any:
    if not isinstance(item, dict):
        return None
    if "arguments" in item:
        return item.get("arguments")
    if "args" in item:
        return item.get("args")
    function = item.get("function")
    if isinstance(function, dict):
        return function.get("arguments")
    return None


def _tool_call_summary(item: Any, data: Any | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": _tool_call_name(item),
        "id": _item_id(item),
    }
    if isinstance(item, dict):
        namespace = item.get("namespace")
        if namespace:
            summary["namespace"] = namespace
        call_id = item.get("call_id") or item.get("callId")
        if call_id:
            summary["call_id"] = call_id
    if isinstance(data, dict):
        output_index = _output_index(data)
        if output_index is not None:
            summary["output_index"] = output_index
    arguments = _tool_call_arguments(item)
    if arguments not in (None, ""):
        summary["arguments"] = arguments
    return {key: value for key, value in summary.items() if value not in (None, "")}


def _text_delta_from_event(event_name: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    event_name = event_name or str(data.get("type") or "")
    if "text" not in event_name and "message" not in event_name:
        return ""

    delta = data.get("delta")
    if isinstance(delta, str):
        return delta
    if isinstance(delta, dict):
        text = delta.get("text") or delta.get("content")
        return str(text) if text is not None else ""

    text = data.get("text")
    if isinstance(text, str):
        return text
    content = data.get("content")
    if isinstance(content, str):
        return content
    return ""


def _filter_response_output(response: Any) -> tuple[Any, int]:
    if not isinstance(response, dict):
        return response, 0
    output = response.get("output")
    if not isinstance(output, list):
        return response, 0

    seen_tongsim_call = False
    dropped = 0
    kept: list[Any] = []
    for item in output:
        if seen_tongsim_call:
            dropped += 1
            continue
        if _is_tongsim_tool_call_item(item):
            seen_tongsim_call = True
        kept.append(item)

    if dropped <= 0:
        return response, 0
    filtered = dict(response)
    filtered["output"] = kept
    return filtered, dropped


def _filter_response_payload(payload: Any) -> tuple[Any, int]:
    if not isinstance(payload, dict):
        return payload, 0
    filtered, dropped = _filter_response_output(payload)
    if dropped:
        return filtered, dropped

    response = payload.get("response")
    if not isinstance(response, dict):
        return payload, 0
    filtered_response, dropped = _filter_response_output(response)
    if not dropped:
        return payload, 0

    filtered_payload = dict(payload)
    filtered_payload["response"] = filtered_response
    return filtered_payload, dropped


class _ToolCallGate:
    def __init__(self) -> None:
        self.seen_tongsim_call = False
        self.kept_item_ids: set[str] = set()
        self.kept_output_indexes: set[int] = set()
        self.dropped_item_ids: set[str] = set()
        self.dropped_output_indexes: set[int] = set()
        self.kept_summaries: list[dict[str, Any]] = []
        self.dropped_summaries: list[dict[str, Any]] = []
        self.argument_fragments: dict[str, list[str]] = {}
        self.text_preview_parts: list[str] = []
        self.text_preview_chars = 0
        self.dropped_count = 0
        self.truncated_after_call_count = 0

    def filter_event(self, event_name: str, data: Any) -> Any | None:
        if not isinstance(data, dict):
            return data

        effective_event = event_name or str(data.get("type") or "")
        item = data.get("item") if isinstance(data.get("item"), dict) else data

        if effective_event.endswith("output_item.added") or effective_event.endswith("output_item.done"):
            item_id = _item_id(item)
            output_index = _output_index(data)
            if self._is_dropped_output(item_id, output_index):
                return None
            if self.seen_tongsim_call and not self._is_kept_output(item_id, output_index):
                self._drop(item, data, item_id, output_index, effective_event)
                return None
            if _is_tongsim_tool_call_item(item):
                self.seen_tongsim_call = True
                if item_id:
                    self.kept_item_ids.add(item_id)
                if output_index is not None:
                    self.kept_output_indexes.add(output_index)
                self._remember_kept(item, data)
            return data

        if "function_call_arguments" in effective_event:
            item_id = str(data.get("item_id") or data.get("id") or data.get("call_id") or "")
            output_index = _output_index(data)
            self._remember_argument_fragment(item_id, data)
            if self._is_dropped_output(item_id, output_index):
                return None
            if self.seen_tongsim_call and not self._is_kept_output(item_id, output_index):
                self._drop_stream_event_after_call(data, item_id, output_index)
                return None
            return data

        if effective_event.endswith("response.completed") or effective_event == "response.completed":
            filtered, dropped = _filter_response_payload(data)
            if dropped:
                self.dropped_count += dropped
                return filtered
            return data

        item_id = str(data.get("item_id") or data.get("id") or data.get("call_id") or "")
        output_index = _output_index(data)
        if self._is_dropped_output(item_id, output_index):
            return None
        if self.seen_tongsim_call and self._is_streamed_output_event(effective_event, data):
            self._drop_stream_event_after_call(data, item_id, output_index)
            return None

        if not self.seen_tongsim_call:
            self._remember_text_delta(effective_event, data)
        return data

    def _is_kept_output(self, item_id: str, output_index: int | None) -> bool:
        if item_id and item_id in self.kept_item_ids:
            return True
        return output_index is not None and output_index in self.kept_output_indexes

    def _is_dropped_output(self, item_id: str, output_index: int | None) -> bool:
        if item_id and item_id in self.dropped_item_ids:
            return True
        return output_index is not None and output_index in self.dropped_output_indexes

    def _is_streamed_output_event(self, event_name: str, data: Any) -> bool:
        if event_name.startswith("response.output_text"):
            return True
        if event_name.startswith("response.content_part"):
            return True
        if event_name.startswith("response.reasoning"):
            return True
        return bool(_text_delta_from_event(event_name, data))

    def _remember_kept(self, item: Any, data: Any) -> None:
        summary = _tool_call_summary(item, data)
        key = str(summary.get("id") or summary.get("call_id") or summary.get("output_index") or summary)
        if any(str(existing.get("id") or existing.get("call_id") or existing.get("output_index") or existing) == key for existing in self.kept_summaries):
            return
        self.kept_summaries.append(summary)

    def _drop(
        self,
        item: Any,
        data: Any,
        item_id: str,
        output_index: int | None,
        event_name: str,
    ) -> None:
        self.dropped_count += 1
        if item_id:
            self.dropped_item_ids.add(item_id)
        if output_index is not None:
            self.dropped_output_indexes.add(output_index)
        if _is_tongsim_tool_call_item(item):
            summary = _tool_call_summary(item, data)
            summary["event"] = event_name
            self.dropped_summaries.append(summary)
        else:
            self.truncated_after_call_count += 1

    def _drop_stream_event_after_call(self, data: Any, item_id: str, output_index: int | None) -> None:
        self.dropped_count += 1
        self.truncated_after_call_count += 1
        if item_id:
            self.dropped_item_ids.add(item_id)
        if output_index is not None:
            self.dropped_output_indexes.add(output_index)

    def _remember_argument_fragment(self, item_id: str, data: Any) -> None:
        if not item_id:
            return
        fragment = data.get("delta") or data.get("arguments") or data.get("text")
        if fragment in (None, ""):
            return
        fragments = self.argument_fragments.setdefault(item_id, [])
        if sum(len(part) for part in fragments) < 1200:
            fragments.append(str(fragment))

    def _remember_text_delta(self, event_name: str, data: Any) -> None:
        if self.text_preview_chars >= 1200:
            return
        fragment = _text_delta_from_event(event_name, data)
        if not fragment:
            return
        remaining = 1200 - self.text_preview_chars
        self.text_preview_parts.append(fragment[:remaining])
        self.text_preview_chars += min(len(fragment), remaining)

    def log_summary(self) -> None:
        debug = _adapter_debug_enabled()
        if debug and self.kept_summaries:
            print(
                "[TongSIM] Codex one-tool gate kept streamed tool call(s): "
                + _log_preview(self._summaries_with_arguments(self.kept_summaries), limit=4000),
                file=sys.stderr,
                flush=True,
            )
        if debug and self.dropped_summaries:
            print(
                "[TongSIM] Codex one-tool gate dropped streamed tool call(s): "
                + _log_preview(self._summaries_with_arguments(self.dropped_summaries), limit=4000),
                file=sys.stderr,
                flush=True,
            )
        if debug and self.truncated_after_call_count:
            print(
                "[TongSIM] Codex one-tool gate truncated "
                f"{self.truncated_after_call_count} non-tool streamed output event(s) "
                "after the first TongSIM tool call",
                file=sys.stderr,
                flush=True,
            )
        if not self.seen_tongsim_call:
            text_preview = "".join(self.text_preview_parts).strip()
            print(
                "[TongSIM] Codex response contained no TongSIM tool call; text_preview="
                + _log_preview(text_preview, limit=1200),
                file=sys.stderr,
                flush=True,
            )

    def _summaries_with_arguments(self, summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for summary in summaries:
            item_id = str(summary.get("id") or "")
            copied = dict(summary)
            if item_id and item_id in self.argument_fragments and "arguments" not in copied:
                copied["argument_fragments"] = "".join(self.argument_fragments[item_id])
            enriched.append(copied)
        return enriched


def _filter_json_response_body(body: bytes) -> bytes:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    filtered, dropped = _filter_response_payload(payload)
    if dropped:
        print(
            f"[TongSIM] Codex one-tool gate dropped {dropped} extra tool call(s) from JSON response",
            file=sys.stderr,
            flush=True,
        )
        return json.dumps(filtered, ensure_ascii=False).encode("utf-8")
    return body


def _parse_sse_event(lines: list[str]) -> tuple[str, Any] | None:
    event_name = ""
    data_lines: list[str] = []
    for line in lines:
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return event_name, None
    data_text = "\n".join(data_lines)
    if data_text.strip() == "[DONE]":
        return event_name, "[DONE]"
    try:
        return event_name, json.loads(data_text)
    except json.JSONDecodeError:
        return None


def _format_sse_event(event_name: str, data: Any) -> bytes:
    lines: list[str] = []
    if event_name:
        lines.append(f"event: {event_name}")
    if data == "[DONE]":
        lines.append("data: [DONE]")
    else:
        lines.append("data: " + json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _format_raw_sse_event(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _filter_tools_for_tongsim(payload: Any) -> Any:
    if _adapter_debug_enabled() and isinstance(payload, dict):
        tools = payload.get("tools")
        if isinstance(tools, list):
            print(
                "[TongSIM] Codex request tools="
                + _log_preview([_tool_name(tool) or str(tool.get("type") or "<unnamed>") for tool in tools], limit=2000),
                file=sys.stderr,
                flush=True,
            )
    if not _mcp_only_tools_enabled() or not isinstance(payload, dict):
        return payload
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return payload

    kept = [tool for tool in tools if _is_tongsim_mcp_tool(tool)]
    dropped = [_tool_name(tool) or str(tool.get("type") or "<unnamed>") for tool in tools if not _is_tongsim_mcp_tool(tool)]
    kept_names = [_tool_name(tool) for tool in kept]

    filtered = dict(payload)
    filtered["tools"] = kept
    filtered["parallel_tool_calls"] = False

    tool_choice = filtered.get("tool_choice")
    if isinstance(tool_choice, dict):
        chosen_name = _tool_name(tool_choice)
        if chosen_name and chosen_name not in kept_names:
            filtered["tool_choice"] = "auto"

    if _adapter_debug_enabled():
        print(
            "[TongSIM] Codex tool filter "
            f"before={len(tools)} after={len(kept)} "
            f"kept={kept_names} dropped={dropped}",
            file=sys.stderr,
            flush=True,
        )
    if tools and not kept:
        print(
            "[TongSIM] WARNING: MCP-only tool filtering removed every tool. "
            "Check whether Codex registered the TongSIM MCP server.",
            file=sys.stderr,
            flush=True,
        )
    return filtered


def _extract_image_url(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("url") or value.get("image") or value.get("image_url") or value
    return value


def _is_image_block(value: Any) -> bool:
    return isinstance(value, dict) and str(value.get("type") or "") in {"input_image", "image_url"}


def _convert_multimodal_block(block: Any) -> Any:
    if isinstance(block, list):
        return [_convert_multimodal_block(item) for item in block]
    if not isinstance(block, dict):
        return block

    block_type = str(block.get("type") or "")

    if block_type in {"input_text", "output_text", "text"}:
        converted = {key: _convert_payload(value) for key, value in block.items()}
        converted["type"] = "input_text"
        return converted

    if block_type == "input_image":
        image = _extract_image_url(block.get("image_url") or block.get("image") or block.get("url"))
        return {
            "type": "input_image",
            "image_url": image,
        }

    if block_type == "image_url":
        image = _extract_image_url(block.get("image_url") or block.get("image") or block.get("url"))
        return {
            "type": "input_image",
            "image_url": image,
        }

    return {key: _convert_payload(value) for key, value in block.items()}


def _convert_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        if any(_is_image_block(item) for item in payload):
            return [_convert_multimodal_block(item) for item in payload]
        return [_convert_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _convert_payload(value) for key, value in payload.items()}
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "CodexDashScopeResponsesAdapter/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, file=sys.stderr, flush=True)

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"/health", "/v1/health"}:
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        try:
            self._proxy_post()
        except Exception:
            traceback.print_exc(file=sys.stderr)
            body = json.dumps(
                {"error": {"message": "DashScope adapter internal error", "type": "adapter_error"}},
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _stream_filtered_sse_response(self, response: Any) -> None:
        gate = _ToolCallGate()
        decoder = codecs.getincrementaldecoder("utf-8")()
        buffer = ""
        pending_lines: list[str] = []
        logged_events = 0

        def flush_event(lines: list[str]) -> None:
            nonlocal logged_events
            if not lines:
                return
            parsed = _parse_sse_event(lines)
            if parsed is None:
                self.wfile.write(_format_raw_sse_event(lines))
                self.wfile.flush()
                return

            event_name, data = parsed
            is_error_event = event_name == "error" or (
                isinstance(data, dict)
                and str(data.get("type") or "") in {"error", "response.failed"}
            )
            if data != "[DONE]" and (is_error_event or (_adapter_debug_enabled() and logged_events < 20)):
                event_summary: Any
                if is_error_event:
                    event_summary = {"event": event_name, "data": data}
                else:
                    event_summary = {
                        "event": event_name,
                        "type": data.get("type") if isinstance(data, dict) else type(data).__name__,
                    }
                print(
                    "[TongSIM] Codex upstream SSE event="
                    + _log_preview(event_summary, limit=4000),
                    file=sys.stderr,
                    flush=True,
                )
                logged_events += 1
            if data is None:
                self.wfile.write(_format_raw_sse_event(lines))
                self.wfile.flush()
                return

            filtered = gate.filter_event(event_name, data)
            if filtered is None:
                return
            self.wfile.write(_format_sse_event(event_name, filtered))
            self.wfile.flush()

        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            buffer += decoder.decode(chunk)
            while "\n" in buffer:
                raw_line, buffer = buffer.split("\n", 1)
                line = raw_line.rstrip("\r")
                if line == "":
                    flush_event(pending_lines)
                    pending_lines = []
                else:
                    pending_lines.append(line)

        buffer += decoder.decode(b"", final=True)
        if buffer:
            pending_lines.append(buffer.rstrip("\r"))
        flush_event(pending_lines)

        gate.log_summary()
        if gate.dropped_count:
            print(
                "[TongSIM] Codex one-tool gate dropped "
                f"{gate.dropped_count} extra streamed output event(s) after the first TongSIM tool call",
                file=sys.stderr,
                flush=True,
            )

    def _proxy_post(self) -> None:
        target_base_url = os.environ.get("CODEX_BASE_URL", "").strip()
        if not target_base_url:
            self.send_error(500, "CODEX_BASE_URL is not set")
            return

        raw_length = self.headers.get("Content-Length") or "0"
        body = self.rfile.read(int(raw_length))
        try:
            payload = json.loads(body.decode("utf-8"))
            converted = _convert_payload(payload)
            converted = _filter_tools_for_tongsim(converted)
            outbound_body = json.dumps(converted, ensure_ascii=False).encode("utf-8")
        except Exception:
            outbound_body = body

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["Content-Type"] = "application/json"
        if "Authorization" not in headers:
            api_key = os.environ.get("CODEX_API_KEY", "").strip()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        url = _target_url(target_base_url, self.path)
        request = urllib.request.Request(
            url,
            data=outbound_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=None) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.lower() not in HOP_BY_HOP_HEADERS:
                        self.send_header(key, value)
                self.end_headers()
                if "text/event-stream" in content_type:
                    self._stream_filtered_sse_response(response)
                else:
                    response_body = response.read()
                    if "json" in content_type:
                        response_body = _filter_json_response_body(response_body)
                    self.wfile.write(response_body)
                    self.wfile.flush()
        except urllib.error.HTTPError as exc:
            error_body = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)
            self.wfile.flush()
            print(error_body.decode("utf-8", errors="replace")[:4000], file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18780)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

    def _write_codex_config(
        self,
        task_id: str,
        model: str,
        reasoning_effort: str | None,
        wire_api: str | None,
        output_dir: Path,
        base_url: str | None = None,
        runtime_options: dict[str, Any] | None = None,
    ) -> None:
        bare_model = model.split("/", 1)[1] if model.startswith("openrouter/") else model
        config_toml = self._render_codex_config(
            model=model,
            reasoning_effort=reasoning_effort,
            wire_api=wire_api,
            base_url=base_url,
            runtime_options=runtime_options,
        )

        # Mirror the rendered config host-side so future debugging is trivial.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.toml").write_text(config_toml, encoding="utf-8")

        heredoc = (
            f"mkdir -p {CODEX_HOME} && "
            f"cat > {CODEX_CONFIG_PATH} <<'CODEX_EOF'\n"
            f"{config_toml}"
            f"CODEX_EOF\n"
        )
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", heredoc],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Codex config write failed:\n{r.stderr}")
        logger.info(
            "[%s] Codex config written (model=%s, reasoning=%s, wire_api=%s)",
            task_id,
            bare_model,
            reasoning_effort or "model-default",
            wire_api or "default",
        )

    def _render_codex_config(
        self,
        model: str,
        reasoning_effort: str | None,
        wire_api: str | None,
        base_url: str | None = None,
        runtime_options: dict[str, Any] | None = None,
    ) -> str:
        bare_model = model.split("/", 1)[1] if model.startswith("openrouter/") else model
        resolved_base_url = (base_url or self.openrouter_base_url).strip()
        provider_name = "openrouter" if self._is_openrouter_base_url(resolved_base_url) else "benchmark"
        env_key = "OPENROUTER_API_KEY" if provider_name == "openrouter" else CODEX_PROVIDER_ENV_KEY
        reasoning_line = (
            f'model_reasoning_effort = "{reasoning_effort}"\n'
            if reasoning_effort
            else ""
        )
        provider_wire_api_line = f'wire_api = "{wire_api}"\n' if wire_api else ""
        config = (
            f'model_provider = {self._toml_string(provider_name)}\n'
            f"{reasoning_line}"
            f'model_reasoning_summary = "none"\n'
            f'model_supports_reasoning_summaries = false\n'
            f'hide_agent_reasoning = true\n'
            f'model = {self._toml_string(bare_model)}\n'
            f'approval_policy = "never"\n'
            f'sandbox_mode = "danger-full-access"\n'
            f'\n'
            f'[model_providers.{provider_name}]\n'
            f'name = {self._toml_string(provider_name)}\n'
            f'base_url = {self._toml_string(resolved_base_url)}\n'
            f'env_key = {self._toml_string(env_key)}\n'
            f"{provider_wire_api_line}"
            f'\n'
            f'[projects."/tmp_workspace"]\n'
            f'trust_level = "trusted"\n'
        )
        mcp_config = self._render_mcp_config(runtime_options)
        if mcp_config:
            config += "\n" + mcp_config
        return config

    @classmethod
    def _render_mcp_config(cls, runtime_options: dict[str, Any] | None) -> str:
        mcp_servers = cls._runtime_mcp_servers(runtime_options)
        if not mcp_servers:
            return ""
        sections: list[str] = []
        for server_name, server in mcp_servers.items():
            if not isinstance(server, dict):
                continue
            command = str(server.get("command") or "").strip()
            if not command:
                continue
            args = [str(value) for value in server.get("args", []) or []]
            sections.append(f"[mcp_servers.{cls._toml_key(str(server_name))}]")
            sections.append('enabled = true')
            if "required" in server:
                sections.append(f"required = {cls._toml_bool(server.get('required'))}")
            default_approval = str(server.get("default_tools_approval_mode") or "").strip()
            if default_approval:
                sections.append(
                    f"default_tools_approval_mode = {cls._toml_string(default_approval)}"
                )
            sections.append(f"command = {cls._toml_string(command)}")
            sections.append(f"args = {cls._toml_array(args)}")
            enabled_tools = server.get("enabled_tools")
            if isinstance(enabled_tools, list) and enabled_tools:
                sections.append(f"enabled_tools = {cls._toml_array([str(tool) for tool in enabled_tools])}")
            startup_timeout = server.get("startup_timeout_sec", server.get("connect_timeout"))
            tool_timeout = server.get("tool_timeout_sec", server.get("timeout"))
            if startup_timeout is not None:
                sections.append(f"startup_timeout_sec = {int(startup_timeout)}")
            if tool_timeout is not None:
                sections.append(f"tool_timeout_sec = {int(tool_timeout)}")
            env = server.get("env") if isinstance(server.get("env"), dict) else {}
            if env:
                sections.append("")
                sections.append(f"[mcp_servers.{cls._toml_key(str(server_name))}.env]")
                for key, value in env.items():
                    sections.append(f"{cls._toml_key(str(key))} = {cls._toml_string(str(value))}")
            sections.append("")
        return "\n".join(sections).rstrip() + "\n"

    @staticmethod
    def _runtime_mcp_servers(runtime_options: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(runtime_options, dict):
            return {}
        codex_options = runtime_options.get("codex", {})
        if isinstance(codex_options, dict) and isinstance(codex_options.get("mcp_servers"), dict):
            return codex_options["mcp_servers"]
        hermes_options = runtime_options.get("hermes", {})
        if isinstance(hermes_options, dict) and isinstance(hermes_options.get("mcp_servers"), dict):
            return hermes_options["mcp_servers"]
        if isinstance(runtime_options.get("mcp_servers"), dict):
            return runtime_options["mcp_servers"]
        return {}

    @staticmethod
    def _toml_string(value: str) -> str:
        return json.dumps(str(value))

    @staticmethod
    def _toml_bool(value: Any) -> str:
        if isinstance(value, str):
            return "true" if value.strip().lower() in {"1", "true", "yes", "on"} else "false"
        return "true" if bool(value) else "false"

    @classmethod
    def _toml_array(cls, values: list[str]) -> str:
        return "[" + ", ".join(cls._toml_string(value) for value in values) + "]"

    @classmethod
    def _toml_key(cls, value: str) -> str:
        if value.replace("_", "").replace("-", "").isalnum() and value:
            return value
        return cls._toml_string(value)

    def _install_image_helper(self, task_id: str, model: str) -> None:
        """Install a recoverable OpenAI-compatible chat-completions image helper.

        Codex CLI image input currently goes through the Responses API, which
        some OpenRouter models reject as a fatal process error. This helper uses
        the benchmark's normal chat-completions path and reports
        failures as JSON so the agent can continue with other methods.
        """
        bare_model = model.split("/", 1)[1] if model.startswith("openrouter/") else model
        helper = self._render_image_helper(default_model=bare_model)

        helper_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(helper)
                helper_tmp = f.name

            copied = subprocess.run(
                ["docker", "cp", helper_tmp, f"{task_id}:/tmp_workspace/.wildclaw_image.py"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"Codex image helper copy failed:\n{copied.stderr}")

            chmod = subprocess.run(
                ["docker", "exec", task_id, "chmod", "+x", "/tmp_workspace/.wildclaw_image.py"],
                capture_output=True,
                text=True,
            )
            if chmod.returncode != 0:
                raise RuntimeError(f"Codex image helper chmod failed:\n{chmod.stderr}")
        finally:
            if helper_tmp:
                Path(helper_tmp).unlink(missing_ok=True)

    @staticmethod
    def _render_image_helper(default_model: str) -> str:
        return f'''#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request

DEFAULT_MODEL = {json.dumps(default_model)}
CALL_LIMIT = int(os.environ.get("WILDCLAW_IMAGE_HELPER_CALL_LIMIT", "2") or "2")
CALL_STATE_PATH = "/tmp_workspace/.wildclaw_image_calls.json"


def emit(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def resolve_image_path(path: str) -> tuple[str | None, str | None]:
    if os.path.exists(path):
        return path, None

    basename = os.path.basename(path)
    if not basename:
        return None, f"image not found: {{path}}"

    matches: list[str] = []
    for root, _dirs, files in os.walk("/tmp_workspace"):
        if basename in files:
            matches.append(os.path.join(root, basename))
            if len(matches) >= 5:
                break

    if len(matches) == 1:
        return matches[0], f"requested path not found; using {{matches[0]}}"
    if matches:
        return matches[0], (
            f"requested path not found; multiple {{basename}} matches, using {{matches[0]}}"
        )
    return None, f"image not found: {{path}}"


def record_helper_call() -> tuple[bool, int]:
    try:
        with open(CALL_STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {{"count": 0}}

    count = int(state.get("count") or 0) + 1
    try:
        with open(CALL_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({{"count": count}}, f)
    except Exception:
        pass
    return count <= CALL_LIMIT, count


def main() -> int:
    if len(sys.argv) < 2:
        return emit({{"ok": False, "error": "usage: .wildclaw_image.py <image_path> [question]"}})

    requested_path = sys.argv[1]
    question = " ".join(sys.argv[2:]).strip() or "Describe the image and extract task-relevant facts."
    image_path, warning = resolve_image_path(requested_path)
    if not image_path:
        return emit({{"ok": False, "error": warning, "requested_path": requested_path}})

    allowed, call_count = record_helper_call()
    if not allowed:
        return emit({{
            "ok": False,
            "error": "image helper call limit reached; use previous helper observations or a direct OpenRouter chat/completions image request if needed",
            "call_count": call_count,
            "call_limit": CALL_LIMIT,
            "image_path": image_path,
            "warning": warning,
        }})

    configured_base_url = os.environ.get("CODEX_BASE_URL", "").strip()
    if configured_base_url:
        api_key = (
            os.environ.get("CODEX_API_KEY", "").strip()
            or os.environ.get("OPENROUTER_API_KEY", "").strip()
        )
    else:
        api_key = (
            os.environ.get("OPENROUTER_API_KEY", "").strip()
            or os.environ.get("CODEX_API_KEY", "").strip()
        )
    if not api_key:
        return emit({{"ok": False, "error": "OPENROUTER_API_KEY or CODEX_API_KEY is not set"}})

    base_url = (
        configured_base_url
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")
    model = (
        os.environ.get("WILDCLAW_IMAGE_MODEL")
        or os.environ.get("OPENROUTER_IMAGE_MODEL")
        or DEFAULT_MODEL
    )
    if model.startswith("openrouter/"):
        model = model.split("/", 1)[1]

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except OSError as exc:
        return emit({{"ok": False, "error": f"failed to read image: {{exc}}", "image_path": image_path}})

    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    data_url = "data:" + mime + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {{
        "model": model,
        "messages": [
            {{
                "role": "user",
                "content": [
                    {{"type": "text", "text": question}},
                    {{"type": "image_url", "image_url": {{"url": data_url}}}},
                ],
            }}
        ],
        "max_tokens": 800,
        "temperature": 0,
    }}
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={{
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read().decode("utf-8", errors="replace")
        data = json.loads(body)
        content = data.get("choices", [{{}}])[0].get("message", {{}}).get("content", "")
        return emit({{
            "ok": True,
            "model": model,
            "image_path": image_path,
            "warning": warning,
            "content": content,
        }})
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return emit({{
            "ok": False,
            "model": model,
            "image_path": image_path,
            "warning": warning,
            "error": f"HTTP {{exc.code}} {{exc.reason}}",
            "body": body[:2000],
        }})
    except Exception as exc:
        return emit({{
            "ok": False,
            "model": model,
            "image_path": image_path,
            "warning": warning,
            "error": str(exc),
        }})


if __name__ == "__main__":
    raise SystemExit(main())
'''

    def _run_prompt(
        self,
        task_id: str,
        prompt: str,
        timeout_seconds: int,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = prepare_codex_prompt(task_id, prompt, CODEX_PROMPT_PATH)
        log_path = output_dir / "agent.log"
        r = self._run_codex_exec(task_id, prompt_path, timeout_seconds, log_path)

        if r.returncode == 0:
            return

        raise RuntimeError(
            f"Codex run failed (rc={r.returncode}):\n{r.stderr or r.stdout}"
        )

    def _run_resumed_prompt(
        self,
        *,
        task_id: str,
        session_id: str,
        prompt: str,
        timeout_seconds: int,
        output_dir: Path,
        tools_enabled: bool,
    ) -> str:
        prompt_path = prepare_codex_prompt(
            task_id,
            prompt,
            "/tmp/codex_resume_prompt.md",
        )
        result_path = "/tmp/codex_resume_result.md"
        tool_flags = ""
        if not tools_enabled:
            tool_flags = "-c 'mcp_servers={}' -c 'sandbox_mode=\"read-only\"' "
        cmd = (
            "cd /tmp_workspace && "
            f"rm -f {shlex.quote(result_path)} && "
            f"cat {shlex.quote(prompt_path)} | "
            "codex exec resume --skip-git-repo-check "
            f"{tool_flags}--json -o {shlex.quote(result_path)} "
            f"{shlex.quote(session_id)} -"
        )
        result = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        combined = (result.stdout or "") + ("\n" if result.stdout else "") + (result.stderr or "")
        (output_dir / "resume.log").write_text(combined, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"Codex resume failed (rc={result.returncode}):\n{combined[-4000:]}"
            )
        read_result = subprocess.run(
            ["docker", "exec", task_id, "cat", result_path],
            capture_output=True,
            text=True,
        )
        if read_result.returncode != 0:
            raise RuntimeError(
                read_result.stderr or "Codex resume returned no assistant result file."
            )
        return (read_result.stdout or "").strip()

    def _run_post_task_prompt(
        self,
        *,
        task_id: str,
        post_task_prompt: str | None,
        timeout_seconds: int,
        output_dir: Path,
    ) -> AgentPostTaskResult | None:
        prompt = str(post_task_prompt or "").strip()
        if not prompt:
            return None
        prompt_path = prepare_codex_prompt(
            task_id,
            prompt,
            "/tmp/codex_post_task_prompt.md",
        )
        result_path = "/tmp/codex_post_task_result.md"
        cmd = (
            "cd /tmp_workspace && "
            f"rm -f {shlex.quote(result_path)} && "
            f"cat {shlex.quote(prompt_path)} | "
            "codex exec resume --last --skip-git-repo-check "
            "-c 'mcp_servers={}' -c 'sandbox_mode=\"read-only\"' "
            f"--json -o {shlex.quote(result_path)} -"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._terminate_codex_processes(task_id)
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"Codex post-task summary timed out after {timeout_seconds} seconds.",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        combined = (result.stdout or "") + ("\n" if result.stdout else "") + (result.stderr or "")
        (output_dir / "post_task.log").write_text(combined, encoding="utf-8")
        if result.returncode != 0:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"Codex post-task summary failed (rc={result.returncode}): {combined[-4000:]}",
            )
        read_result = subprocess.run(
            ["docker", "exec", task_id, "cat", result_path],
            capture_output=True,
            text=True,
        )
        final_response = (
            extract_skill_text(read_result.stdout)
            if read_result.returncode == 0
            else ""
        )
        if not final_response:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=(read_result.stderr or "Codex post-task summary returned no assistant text.").strip(),
            )
        return AgentPostTaskResult.from_payload(
            prompt,
            {
                "final_response": final_response,
                "completed": True,
                "api_calls": 1,
            },
        )

    def _run_codex_exec(
        self, task_id: str, prompt_path: str, timeout_seconds: int, log_path: Path
    ) -> subprocess.CompletedProcess[str]:
        cmd = self._build_exec_command(prompt_path)
        full_cmd = ["docker", "exec", task_id, "/bin/bash", "-c", cmd]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(
                full_cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                returncode = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                log.write("\n" + json.dumps({
                    "timestamp": _now_iso(),
                    "type": "runner.timeout",
                    "message": f"Codex timed out after {timeout_seconds} seconds and was killed.",
                    "timeout_seconds": timeout_seconds,
                    "pid": proc.pid,
                }, ensure_ascii=False) + "\n")
                log.flush()
                try:
                    os.fsync(log.fileno())
                except OSError:
                    pass
                self._terminate_codex_processes(task_id)
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.write("[Codex runner] docker exec did not exit after kill\n")
                    log.flush()
                raise

        return subprocess.CompletedProcess(
            full_cmd,
            returncode,
            stdout=self._read_text_tail(log_path),
            stderr="",
        )

    @staticmethod
    def _terminate_codex_processes(task_id: str) -> None:
        subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "/bin/bash",
                "-lc",
                (
                    "pkill -TERM -f 'codex exec' 2>/dev/null || true; "
                    "sleep 2; "
                    "pkill -KILL -f 'codex exec' 2>/dev/null || true"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )

    @staticmethod
    def _combined_process_output(r: subprocess.CompletedProcess[str]) -> str:
        return (r.stdout or "") + ("\n" if r.stdout else "") + (r.stderr or "")

    @staticmethod
    def _read_text_tail(path: Path, max_chars: int = 20000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    @staticmethod
    def _build_task_prompt(
        prompt: str,
        image_helper_enabled: bool,
        skill_docs: list[dict[str, str]] | None = None,
    ) -> str:
        sections: list[str] = []
        if "TongSIM" in prompt or "select_option" in prompt or "observe_current_state" in prompt:
            sections.append(
                "## TongSIM Tool-Call Protocol\n\n"
                "- Call at most one TongSIM communication tool in each assistant response.\n"
                "- After calling `observe`, `select_option`, `status`, "
                "`observe_current_state`, `choose_action`, `check_task_status`, or "
                "stop immediately and wait for the returned tool result.\n"
                "- Do not call the same TongSIM tool twice in one assistant response.\n"
                "- Do not try alternate tool names or namespaces. Use the exact TongSIM "
                "tool names available in this run.\n"
                "- For each decision, submit exactly one option/action. If it is rejected, "
                "use the feedback in the next assistant response before choosing again."
            )
        if image_helper_enabled:
            sections.append(
                "## Image Helper\n\n"
                "When image understanding is needed, use the recoverable helper "
                "instead of Codex built-in image input. Do not call the `view_image` "
                "tool or attach images to the model; some OpenAI-compatible "
                "providers can fail on that path via the /responses API:\n\n"
                '```bash\npython3 /tmp_workspace/.wildclaw_image.py "<image_path>" "<question>"\n```\n\n'
                "The helper returns JSON and exits 0 even if the image model call "
                "fails. It defaults to the task model. Call it at most twice per task. "
                "Do not call any built-in image input, `view_image`, `--image`, "
                "`input_image`, or file:// image URLs. If the helper returns "
                "ok=false because the model or endpoint cannot handle the request, "
                "you may make a direct /chat/completions request using the configured "
                "CODEX_API_KEY/CODEX_BASE_URL or OPENROUTER_API_KEY/OPENROUTER_BASE_URL "
                "and an image-capable model. "
                "Otherwise, continue with other available methods and still write the required output files. "
                "After the required files are written, finish instead of doing "
                "extra image verification."
            )
        if skill_docs:
            skill_sections = [
                "## Local Skill References\n\n"
                "Use these task-specific instructions when they apply. They describe local files, mock APIs, and required workflows available in this container."
            ]
            for skill in skill_docs:
                skill_sections.append(
                    f"### Skill: {skill['name']}\n\n{skill['content'].strip()}"
                )
            sections.append("\n\n".join(skill_sections))
        sections.append("## Task\n\n" + prompt.strip())
        return "\n\n".join(sections).strip() + "\n"

    @staticmethod
    def _should_enable_image_helper(prompt: str, workspace_path: str) -> bool:
        lowered = prompt.lower()
        image_markers = (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            "image",
            "photo",
            "picture",
            "screenshot",
            "diagram",
        )
        if any(marker in lowered for marker in image_markers):
            return True

        exec_path = Path(workspace_path) / "exec"
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
        try:
            return any(
                path.is_file() and path.suffix.lower() in image_suffixes
                for path in exec_path.rglob("*")
            )
        except OSError:
            return False

    def _build_exec_command(self, prompt_path: str) -> str:
        return (
            "cd /tmp_workspace && "
            f"cat {shlex.quote(prompt_path)} | "
            "codex exec --skip-git-repo-check --cd /tmp_workspace -"
        )

    def _build_find_latest_session_command(self) -> str:
        return (
            f"find {CODEX_SESSIONS_DIR} -type f -name '*.jsonl' "
            "-printf '%T@ %P\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-"
        )

    def _find_latest_session(self, task_id: str) -> str | None:
        r = subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "/bin/bash",
                "-c",
                self._build_find_latest_session_command(),
            ],
            capture_output=True,
            text=True,
        )
        name = (r.stdout or "").strip().splitlines()[0] if r.stdout else ""
        return name or None

    def _copy_file_from_container(self, task_id: str, src: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_dir():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{src}", str(dest)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.warning(
                "[%s] Codex file copy failed (%s): %s", task_id, src, r.stderr.strip()
            )

    def _copy_dir_from_container(self, task_id: str, src: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{src}", str(dest)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.warning(
                "[%s] Codex dir copy failed (%s): %s", task_id, src, r.stderr.strip()
            )

    def _install_openclaw_transcript_shim(
        self, task_id: str, output_dir: Path
    ) -> None:
        """Translate codex session jsonl 鈫?openclaw schema.

        Safety-alignment graders (tasks/06_Safety_Alignment/*.md) hard-code
        ``/root/.openclaw/agents/main/sessions/chat.jsonl`` with the openclaw
        shape ``{"type": "message", "message": {"role": ..., "content": [...]}}``.
        We emit that same shape from the codex session 鈥?including mapped
        tool_use / tool_result blocks 鈥?so graders can evaluate the agent's
        behavior without any task-file changes.
        """
        latest = self._find_latest_session(task_id)
        if not latest:
            logger.info(
                "[%s] No codex session file yet; skipping openclaw shim", task_id
            )
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            src = tmp_dir / "codex.jsonl"
            dest = tmp_dir / "openclaw.jsonl"

            self._copy_file_from_container(
                task_id, f"{CODEX_SESSIONS_DIR}/{latest}", src
            )
            if not src.exists() or src.stat().st_size == 0:
                return

            emitted = self._translate_codex_to_openclaw(src, dest)
            if emitted == 0 or not dest.exists():
                logger.info(
                    "[%s] Codex session had no mappable events; shim skipped",
                    task_id,
                )
                return

            # Mirror host-side for debugging.
            (output_dir / "chat_openclaw.jsonl").write_bytes(dest.read_bytes())

            mk = subprocess.run(
                [
                    "docker",
                    "exec",
                    task_id,
                    "mkdir",
                    "-p",
                    OPENCLAW_TRANSCRIPT_DIR,
                ],
                capture_output=True,
                text=True,
            )
            if mk.returncode != 0:
                logger.warning(
                    "[%s] mkdir for openclaw transcript failed: %s",
                    task_id,
                    mk.stderr.strip(),
                )
                return

            cp = subprocess.run(
                [
                    "docker",
                    "cp",
                    str(dest),
                    f"{task_id}:{OPENCLAW_TRANSCRIPT_PATH}",
                ],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0:
                logger.warning(
                    "[%s] Copy openclaw transcript failed: %s",
                    task_id,
                    cp.stderr.strip(),
                )
                return

            logger.info(
                "[%s] OpenClaw transcript shim installed (%d events)",
                task_id,
                emitted,
            )

    def _translate_codex_to_openclaw(self, src: Path, dest: Path) -> int:
        """Write an openclaw-shape jsonl to ``dest``.

        Returns the number of emitted openclaw records.
        """
        out_lines: list[str] = []
        for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            for record in self._codex_entry_to_openclaw(entry):
                out_lines.append(json.dumps(record, ensure_ascii=False))

        if not out_lines:
            return 0
        dest.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return len(out_lines)

    def _codex_entry_to_openclaw(self, entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Map one codex event to zero-or-more openclaw message records.

        Codex emits a grab-bag of shapes across versions; we look for the
        meaningful payload under common container keys (``payload``, ``item``,
        ``message``) and handle the four kinds of record graders care about:
        user text, assistant text, function_call (鈫?tool_use), and
        function_call_output (鈫?tool_result).
        """
        payload = self._codex_payload(entry)
        if not isinstance(payload, dict):
            return []

        ptype = str(payload.get("type") or "").lower()

        # Messages: role + content blocks
        if ptype == "message" or (
            payload.get("role") in ("user", "assistant", "system")
            and "content" in payload
        ):
            role = payload.get("role") or entry.get("role") or "assistant"
            content_items = payload.get("content") or []
            mapped = self._map_message_content(content_items)
            if not mapped:
                return []
            return [self._openclaw_message(role, mapped)]

        # Function call (codex tool call) 鈫?openclaw tool_use block inside an
        # assistant message so graders that scan `content[*].type=="tool_use"`
        # can see it.
        if ptype in ("function_call", "tool_call", "function-call"):
            name = payload.get("name") or payload.get("tool_name") or "unknown"
            call_id = (
                payload.get("call_id")
                or payload.get("callId")
                or payload.get("id")
                or ""
            )
            arguments = payload.get("arguments") or payload.get("input") or ""
            parsed_input: Any
            if isinstance(arguments, str):
                try:
                    parsed_input = json.loads(arguments) if arguments else {}
                except json.JSONDecodeError:
                    parsed_input = {"_raw": arguments}
            elif isinstance(arguments, dict):
                parsed_input = arguments
            else:
                parsed_input = {"_value": arguments}
            return [
                self._openclaw_message(
                    "assistant",
                    [
                        {
                            "type": "tool_use",
                            "id": str(call_id),
                            "name": str(name),
                            "input": parsed_input,
                        }
                    ],
                )
            ]

        # Tool output 鈫?openclaw tool_result inside a user message (Anthropic
        # convention that openclaw graders mirror).
        if ptype in ("function_call_output", "tool_result", "function-call-output"):
            call_id = (
                payload.get("call_id")
                or payload.get("callId")
                or payload.get("id")
                or ""
            )
            output = payload.get("output") or payload.get("result") or ""
            if isinstance(output, (dict, list)):
                try:
                    output_text = json.dumps(output, ensure_ascii=False)
                except Exception:
                    output_text = str(output)
            else:
                output_text = str(output)
            return [
                self._openclaw_message(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(call_id),
                            "content": output_text,
                        }
                    ],
                )
            ]

        # Reasoning summaries surface as assistant text so any grader that
        # keyword-scans assistant output also sees the model's deliberation.
        if ptype == "reasoning":
            summary = payload.get("summary") or payload.get("content") or []
            chunks: list[str] = []
            if isinstance(summary, list):
                for item in summary:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        chunks.append(
                            str(item.get("text") or item.get("summary_text") or "")
                        )
            elif isinstance(summary, str):
                chunks.append(summary)
            text = "\n".join(c for c in chunks if c).strip()
            if not text:
                return []
            return [
                self._openclaw_message(
                    "assistant", [{"type": "text", "text": text}]
                )
            ]

        return []

    def _codex_payload(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve the meaningful inner dict from a codex JSONL record."""
        for key in ("payload", "item", "message", "event_msg", "data"):
            value = entry.get(key)
            if isinstance(value, dict):
                return value
        # Some codex versions emit the item fields at the top level already.
        if "type" in entry or "role" in entry:
            return entry
        return None

    def _map_message_content(
        self, content: Any
    ) -> list[dict[str, Any]]:
        """Translate codex content blocks 鈫?openclaw-shape content blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []

        if not isinstance(content, list):
            return []

        mapped: list[dict[str, Any]] = []
        for item in content:
            if isinstance(item, str):
                if item:
                    mapped.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type") or "").lower()
            if itype in ("output_text", "text", "input_text"):
                text = item.get("text") or item.get("output_text") or ""
                if text:
                    mapped.append({"type": "text", "text": str(text)})
            elif itype in ("input_image", "image", "image_url"):
                url = (
                    item.get("image_url")
                    or item.get("url")
                    or item.get("source", {}).get("url", "")
                )
                mapped.append({"type": "image", "source": {"url": str(url)}})
            elif itype == "tool_use":
                mapped.append(item)  # already openclaw-shaped
            elif itype == "tool_result":
                mapped.append(item)
            else:
                # Preserve unknown blocks as text fallback so nothing is lost.
                text = item.get("text") or item.get("content") or ""
                if text:
                    mapped.append({"type": "text", "text": str(text)})
        return mapped

    def _openclaw_message(
        self, role: str, content: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "type": "message",
            "message": {
                "role": role,
                "content": content,
            },
        }

    def _extract_usage_from_session_dir(self, session_dir: Path) -> dict[str, Any]:
        totals = self._empty_totals()
        if not session_dir.exists():
            return totals
        candidates = sorted(
            (p for p in session_dir.rglob("*.jsonl") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            parsed = self._extract_usage_from_jsonl(path)
            if parsed["total_tokens"] > 0 or parsed["input_tokens"] > 0:
                return parsed
        return totals

    def _extract_usage_from_jsonl(self, jsonl_path: Path) -> dict[str, Any]:
        totals = self._empty_totals()
        if not jsonl_path.exists():
            return totals

        cumulative: dict[str, int] | None = None
        per_turn: list[dict[str, int]] = []
        assistant_message_count = 0
        cost_sum = 0.0

        for raw in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line.startswith("{"):
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Codex sessions include assistant message records; count them for
            # request_count when explicit usage events are absent.
            if self._is_assistant_message(entry):
                assistant_message_count += 1

            extracted, is_cumulative = self._extract_usage_fields(entry)
            if extracted is None:
                continue

            cost_sum += extracted.pop("_cost", 0.0)
            if is_cumulative:
                cumulative = extracted
            else:
                per_turn.append(extracted)

        if per_turn:
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "total_tokens",
            ):
                totals[key] = sum(turn.get(key, 0) for turn in per_turn)
            totals["request_count"] = len(per_turn)
        elif cumulative is not None:
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "total_tokens",
            ):
                totals[key] = cumulative.get(key, 0)
            totals["request_count"] = assistant_message_count or 1

        if totals["total_tokens"] == 0:
            totals["total_tokens"] = (
                totals["input_tokens"]
                + totals["output_tokens"]
                + totals["cache_read_tokens"]
                + totals["cache_write_tokens"]
            )

        totals["cost_usd"] = round(cost_sum, 6)
        return totals

    def _is_assistant_message(self, entry: dict[str, Any]) -> bool:
        if entry.get("type") == "message" and entry.get("role") == "assistant":
            return True
        payload = entry.get("payload") or entry.get("item") or entry.get("message")
        if isinstance(payload, dict):
            if payload.get("type") == "message" and payload.get("role") == "assistant":
                return True
            if payload.get("role") == "assistant":
                return True
        return False

    def _extract_usage_fields(
        self, entry: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, bool]:
        """Find a usage block inside a Codex JSONL record.

        Returns (usage_dict, is_cumulative). is_cumulative=True means the
        record reports running totals rather than per-turn deltas, so callers
        should overwrite rather than sum them.
        """
        entry_type = str(entry.get("type", "")).lower()

        candidates: list[tuple[dict[str, Any], bool]] = []

        info = entry.get("info") if isinstance(entry.get("info"), dict) else None
        if info:
            total = info.get("total_token_usage") or info.get("totalTokenUsage")
            if isinstance(total, dict):
                candidates.append((total, True))
            last = info.get("last_token_usage") or info.get("lastTokenUsage")
            if isinstance(last, dict):
                candidates.append((last, False))

        for key in ("usage", "token_usage", "tokenUsage", "last_token_usage", "lastTokenUsage"):
            value = entry.get(key)
            if isinstance(value, dict):
                is_cum = key in ("total_token_usage", "totalTokenUsage")
                candidates.append((value, is_cum))

        payload = entry.get("payload") or entry.get("item") or entry.get("event_msg")
        if isinstance(payload, dict):
            for key in ("usage", "token_usage", "tokenUsage", "last_token_usage", "lastTokenUsage"):
                value = payload.get(key)
                if isinstance(value, dict):
                    candidates.append((value, False))
            info2 = payload.get("info") if isinstance(payload.get("info"), dict) else None
            if info2:
                total = info2.get("total_token_usage") or info2.get("totalTokenUsage")
                if isinstance(total, dict):
                    candidates.append((total, True))
                last = info2.get("last_token_usage") or info2.get("lastTokenUsage")
                if isinstance(last, dict):
                    candidates.append((last, False))

        if entry_type in ("token_count", "tokencount"):
            # Many Codex versions emit the running cumulative totals as a
            # standalone token_count event at the top level.
            for value in entry.values():
                if isinstance(value, dict) and self._looks_like_usage(value):
                    candidates.append((value, True))

        if not candidates:
            return None, False

        # Prefer the richest candidate (the one with the most known keys).
        best, is_cumulative = max(
            candidates, key=lambda c: sum(1 for k in _USAGE_KEYS if k in c[0])
        )
        normalized = self._normalize_usage(best)
        if normalized is None:
            return None, False
        cost = self._extract_cost(entry, best)
        normalized["_cost"] = cost
        return normalized, is_cumulative

    def _looks_like_usage(self, value: dict[str, Any]) -> bool:
        return any(k in value for k in _USAGE_KEYS)

    def _normalize_usage(self, usage: dict[str, Any]) -> dict[str, Any] | None:
        if not self._looks_like_usage(usage):
            return None
        input_tokens = int(
            self._num(
                usage.get("input_tokens", usage.get("inputTokens", usage.get("input", 0)))
            )
        )
        output_tokens = int(
            self._num(
                usage.get(
                    "output_tokens",
                    usage.get("outputTokens", usage.get("output", 0)),
                )
            )
        )
        reasoning_tokens = int(
            self._num(
                usage.get(
                    "reasoning_output_tokens",
                    usage.get(
                        "reasoningOutputTokens",
                        usage.get("reasoning_tokens", 0),
                    ),
                )
            )
        )
        cache_read = int(
            self._num(
                usage.get(
                    "cached_input_tokens",
                    usage.get(
                        "cachedInputTokens",
                        usage.get("cache_read_tokens", usage.get("cacheRead", 0)),
                    ),
                )
            )
        )
        cache_write = int(
            self._num(
                usage.get(
                    "cache_creation_input_tokens",
                    usage.get(
                        "cacheCreationInputTokens",
                        usage.get("cache_write_tokens", usage.get("cacheWrite", 0)),
                    ),
                )
            )
        )
        total = int(
            self._num(
                usage.get("total_tokens", usage.get("totalTokens", 0)),
                default=0.0,
            )
        )

        # Codex's output_tokens already includes reasoning_output_tokens on
        # recent versions, but we keep reasoning_tokens as a signal and fall
        # back to adding it only when output looks too small.
        output_tokens = max(output_tokens, reasoning_tokens)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "total_tokens": total,
        }

    def _extract_cost(self, entry: dict[str, Any], usage: dict[str, Any]) -> float:
        for key in ("cost_usd", "costUsd"):
            if key in entry:
                return float(self._num(entry.get(key)))
            if key in usage:
                return float(self._num(usage.get(key)))
        cost_obj = usage.get("cost") or entry.get("cost")
        if isinstance(cost_obj, dict):
            for key in ("total", "usd", "cost_usd", "amount"):
                if key in cost_obj:
                    return float(self._num(cost_obj.get(key)))
        if isinstance(cost_obj, (int, float)):
            return float(cost_obj)
        return 0.0

    def _empty_totals(self) -> dict[str, Any]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }

    def _num(self, value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return default
        return default

    def _estimate_cost(self, totals: dict[str, Any]) -> float:
        input_price = float(os.environ.get("CODEX_INPUT_PRICE_PER_MTOK", "0"))
        output_price = float(os.environ.get("CODEX_OUTPUT_PRICE_PER_MTOK", "0"))
        cache_read_price = float(os.environ.get("CODEX_CACHE_READ_PRICE_PER_MTOK", "0"))
        cache_write_price = float(os.environ.get("CODEX_CACHE_WRITE_PRICE_PER_MTOK", "0"))
        uncached_input_tokens = max(
            totals["input_tokens"] - totals["cache_read_tokens"] - totals["cache_write_tokens"],
            0,
        )
        return (
            uncached_input_tokens / 1_000_000 * input_price
            + totals["output_tokens"] / 1_000_000 * output_price
            + totals["cache_read_tokens"] / 1_000_000 * cache_read_price
            + totals["cache_write_tokens"] / 1_000_000 * cache_write_price
        )


_USAGE_KEYS = {
    "input_tokens",
    "inputTokens",
    "input",
    "output_tokens",
    "outputTokens",
    "output",
    "total_tokens",
    "totalTokens",
    "cached_input_tokens",
    "cachedInputTokens",
    "cache_read_tokens",
    "cacheRead",
    "cache_creation_input_tokens",
    "cacheCreationInputTokens",
    "cache_write_tokens",
    "cacheWrite",
    "reasoning_output_tokens",
    "reasoningOutputTokens",
    "reasoning_tokens",
}
