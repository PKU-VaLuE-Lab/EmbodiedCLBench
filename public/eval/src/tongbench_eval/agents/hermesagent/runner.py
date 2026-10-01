from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from tongbench_eval.agents.base import (
    AgentCheckpoint,
    AgentExecution,
    AgentPostTaskResult,
    AgentTaskSpec,
    BaseAgent,
)
from tongbench_eval.agents.hermesagent.skill_text import extract_skill_text
from tongbench_eval.utils.docker_utils import (
    build_bind_mount_args,
    build_host_gateway_args,
    run_warmup,
    setup_skills,
    inject_lobster_workspace,
    TMP_WORKSPACE,
)
from tongbench_eval.utils.grading import extract_usage_from_jsonl

load_dotenv()

logger = logging.getLogger(__name__)

HERMES_IMAGE = os.environ.get("HERMES_DOCKER_IMAGE", "wildclawbench-hermes-agent:v0.5")
HERMES_HOME = "/root/.hermes"
HERMES_INSTALL_DIR = "/opt/hermes"
HERMES_VENV_PYTHON = "/opt/hermes/.venv/bin/python3"

OPENCLAW_COMPAT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"
BENCH_RUNNER_HOST_PATH = Path(__file__).with_name("bench_runner.py")
BENCH_CONFIG_CONTAINER_PATH = "/tmp/hermes_bench_config.json"
POST_TASK_RESULT_CONTAINER_PATH = "/tmp/hermes_post_task_result.json"
PRIMARY_RESULT_CONTAINER_PATH = "/tmp/hermes_primary_result.json"
RESUME_SESSION_CONTAINER_PATH = "/tmp/hermes_resume_session.json"
COMPAT_TRANSCRIPT_HOST_PATH = Path(__file__).with_name("compat_transcript.py")
HERMES_SOURCE_PATCH_DIR = Path(__file__).resolve().parents[2] / "frameworks" / "hermes_impl"
HERMES_SOURCE_PATCHES = (
    ("run_agent.py", f"{HERMES_INSTALL_DIR}/run_agent.py"),
    ("mcp_tool.py", f"{HERMES_INSTALL_DIR}/tools/mcp_tool.py"),
    ("task_history_compact.py", f"{HERMES_INSTALL_DIR}/task_history_compact.py"),
)


class HermesAgentAgent(BaseAgent):
    def __init__(
        self,
        image: str | None = None,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        brave_api_key: str = "",
    ) -> None:
        self.image = image or HERMES_IMAGE
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = openrouter_base_url
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return OPENCLAW_COMPAT_TRANSCRIPT_PATH

    def prepare_grading_transcript(self, task_id: str) -> str:
        self._write_compat_transcript(task_id)
        return self.transcript_container_path

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        elapsed_time = float(spec.timeout_seconds)
        agent_proc = None
        termination_reason = "startup"

        try:
            api_key, base_url = self._resolve_runtime_provider(spec.model, spec.models_config)

            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            self._start_container(
                spec.task_id,
                exec_path,
                api_key=api_key,
                base_url=base_url,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
                bind_mounts=spec.task.get("bind_mounts"),
            )
            self._patch_hermes_sources(spec.task_id)
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            self._prepare_workspace(spec.task_id)
            setup_skills(
                spec.task_id,
                spec.task.get("skills", ""),
                spec.task.get("skills_path", ""),
                container_skills_root=f"{HERMES_HOME}/skills",
            )
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            self._configure_hermes(spec.task_id, api_key, base_url, spec.runtime_options)

            resume_session_path = None
            if spec.resume_checkpoint is not None:
                checkpoint = AgentCheckpoint.load(
                    spec.resume_checkpoint.path,
                    expected_backend="hermesagent",
                )
                resume_session_path = self._install_resume_checkpoint(spec.task_id, checkpoint)

            reasoning_config = self._map_thinking(spec.thinking)
            self._write_bench_runner(
                spec.task_id, spec.prompt, spec.model,
                api_key, base_url, reasoning_config, spec.runtime_options,
                post_task_prompt=spec.post_task_prompt,
                resume_session_path=resume_session_path,
                resume_without_prompt=bool(spec.resume_without_prompt),
                tools_enabled=bool(spec.tools_enabled),
            )

            start_time = time.perf_counter()

            agent_proc = self._run_bench_runner_background(
                task_id=spec.task_id,
                log_path=spec.output_dir / "agent.log",
            )

            logger.info("[%s] Waiting for hermes-agent to finish...", spec.task_id)
            deadline = start_time + float(spec.timeout_seconds)
            while True:
                if agent_proc.poll() is not None:
                    termination_reason = "process_exit"
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(
                        "[%s] hermes-agent finished, elapsed: %.2f seconds",
                        spec.task_id,
                        elapsed_time,
                    )
                    break

                if self._should_stop_for_runtime(spec.runtime_options):
                    termination_reason = "runtime_done"
                    logger.info("[%s] TongSIM runtime reached done=true; stopping hermes-agent immediately.", spec.task_id)
                    elapsed_time = time.perf_counter() - start_time
                    agent_proc.terminate()
                    try:
                        agent_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        agent_proc.kill()
                        agent_proc.wait()
                    break

                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    termination_reason = "timeout"
                    logger.info("[%s] hermes-agent timed out...", spec.task_id)
                    elapsed_time = float(spec.timeout_seconds)
                    agent_proc.kill()
                    agent_proc.wait()
                    break

                try:
                    agent_proc.wait(timeout=min(0.5, remaining))
                except subprocess.TimeoutExpired:
                    continue
            self._close_runner_streams(agent_proc)

            logger.info("[%s] hermes-agent exit code: %s", spec.task_id, agent_proc.returncode)
            self._cleanup_bench_config(spec.task_id)
            primary_result = self._read_primary_result(spec.task_id, spec.prompt)
            post_task_result = self._read_post_task_result(spec.task_id, spec.post_task_prompt)

            execution_error = None
            if termination_reason == "timeout":
                execution_error = f"Hermes agent timed out after {spec.timeout_seconds:g} seconds"
            elif termination_reason == "process_exit" and agent_proc.returncode != 0:
                execution_error = f"Hermes agent exited with code {agent_proc.returncode}"
            elif primary_result.error:
                execution_error = primary_result.error

            return AgentExecution(
                elapsed_time=elapsed_time,
                error=execution_error,
                gateway_proc=None,
                agent_proc=agent_proc,
                post_task_result=post_task_result,
                final_response=primary_result.final_response,
            )
        except Exception as exc:
            if agent_proc is not None:
                self._close_runner_streams(agent_proc)
            self._cleanup_bench_config(spec.task_id)
            logger.error("[%s] hermes-agent execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=None,
                agent_proc=agent_proc,
            )

    def export_checkpoint(
        self,
        task_id: str,
        checkpoint_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentCheckpoint:
        checkpoint_path = checkpoint_dir.resolve()
        native_dir = checkpoint_path / "native" / "sessions"
        if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
        native_dir.mkdir(parents=True, exist_ok=True)
        copied = subprocess.run(
            ["docker", "cp", f"{task_id}:{HERMES_HOME}/sessions/.", str(native_dir)],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Hermes checkpoint export failed: {copied.stderr.strip()}")
        session_files = sorted(native_dir.glob("session_*.json"), key=lambda path: path.stat().st_mtime)
        if not session_files:
            raise FileNotFoundError("Hermes checkpoint contains no native session JSON.")
        session_payload = json.loads(session_files[-1].read_text(encoding="utf-8"))
        session_id = str(session_payload.get("session_id") or session_files[-1].stem)
        return AgentCheckpoint.create(
            backend="hermesagent",
            path=checkpoint_path,
            session_id=session_id,
            metadata=metadata,
        )

    @staticmethod
    def _install_resume_checkpoint(task_id: str, checkpoint: AgentCheckpoint) -> str:
        session_files = sorted((checkpoint.path / "native" / "sessions").glob("session_*.json"))
        if not session_files:
            raise FileNotFoundError(f"Hermes checkpoint has no session JSON: {checkpoint.path}")
        selected = session_files[-1]
        for candidate in session_files:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if str(payload.get("session_id") or "") == checkpoint.session_id:
                selected = candidate
                break
        if os.environ.get("TONGBENCH_RUNTIME", "docker").strip().lower() == "native":
            # Native mode has no real container namespace.  Copy directly to
            # the task's host-mapped /tmp so the benchmark runner and the
            # compatibility CLI cannot disagree about which root to use.
            from tongbench_eval.utils.native_runtime import native_path

            native_destination = Path(
                native_path(task_id, RESUME_SESSION_CONTAINER_PATH)
            )
            native_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(selected, native_destination)
            if not native_destination.is_file():
                raise FileNotFoundError(
                    f"Hermes native checkpoint restore did not create: {native_destination}"
                )
            return str(native_destination)

        copied = subprocess.run(
            ["docker", "cp", str(selected), f"{task_id}:{RESUME_SESSION_CONTAINER_PATH}"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"Hermes checkpoint restore failed: {copied.stderr.strip()}")
        return RESUME_SESSION_CONTAINER_PATH

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        transcript_host = output_dir / "chat.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        r_cp = subprocess.run(
            ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(transcript_host)],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode == 0 and transcript_host.exists():
            usage = extract_usage_from_jsonl(transcript_host)
        else:
            logger.warning("[%s] Transcript copy failed: %s", task_id, r_cp.stderr.strip())
            usage = self._extract_usage_from_session_logs(task_id)

        if self._usage_has_no_tokens(usage):
            log_usage = self._extract_usage_from_agent_log(output_dir / "agent.log")
            if not self._usage_has_no_tokens(log_usage):
                usage = log_usage

        self._copy_session_log(task_id, output_dir)
        subprocess.run(
            [
                "docker",
                "cp",
                f"{task_id}:{POST_TASK_RESULT_CONTAINER_PATH}",
                str(output_dir / "post_task_result.json"),
            ],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "docker",
                "cp",
                f"{task_id}:{PRIMARY_RESULT_CONTAINER_PATH}",
                str(output_dir / "primary_result.json"),
            ],
            capture_output=True,
            text=True,
        )

        # Prefer the usage summary written by the benchmark-owned runner.
        # The values originate from the provider response usage accumulated by
        # Hermes; transcript and log fallbacks remain for older runs.
        primary_host = output_dir / "primary_result.json"
        primary_payload: dict[str, Any] = {}
        if primary_host.exists():
            try:
                loaded = json.loads(primary_host.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    primary_payload = loaded
            except (OSError, json.JSONDecodeError):
                primary_payload = {}
        primary_usage = primary_payload.get("usage")
        if isinstance(primary_usage, dict) and primary_usage:
            usage = self._normalize_primary_usage(primary_usage)
            usage["context_mode"] = primary_payload.get("context_mode", "full")

        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    @staticmethod
    def _normalize_primary_usage(raw: dict[str, Any]) -> dict[str, Any]:
        def integer(key: str) -> int:
            try:
                return int(raw.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        prompt_tokens = integer("prompt_tokens") or integer("input_tokens")
        output_tokens = integer("output_tokens") or integer("completion_tokens")
        total_tokens = integer("total_tokens") or prompt_tokens + output_tokens
        return {
            "input_tokens": prompt_tokens,
            "uncached_input_tokens": integer("uncached_input_tokens"),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_tokens": integer("cache_read_tokens"),
            "cache_write_tokens": integer("cache_write_tokens"),
            "reasoning_tokens": integer("reasoning_tokens"),
            "request_count": integer("request_count"),
            "usage_source": "provider_response_summary",
            "usage_complete": total_tokens == prompt_tokens + output_tokens,
        }

    def _should_stop_for_runtime(self, runtime_options: dict[str, Any] | None) -> bool:
        if not isinstance(runtime_options, dict):
            return False
        if not bool(runtime_options.get("stop_when_finalized", False)):
            return False
        raw_path = str(runtime_options.get("runtime_meta_path", "") or "").strip()
        if not raw_path:
            return False
        meta_path = Path(raw_path)
        if not meta_path.exists():
            return False
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return bool(payload.get("finalized", False) or payload.get("last_done", False))

    # ------------------------------------------------------------------
    # Provider / thinking helpers
    # ------------------------------------------------------------------

    def _resolve_runtime_provider(self, model: str, models_config: dict | None) -> tuple[str, str]:
        api_key = self.openrouter_api_key
        base_url = self.openrouter_base_url
        config_key, config_base_url = self._resolve_provider_config(model, models_config)
        if config_key:
            api_key = config_key
        if config_base_url:
            base_url = config_base_url
        return api_key, base_url

    @staticmethod
    def _resolve_provider_config(model: str, models_config: dict | None) -> tuple[str, str]:
        """Extract api_key and base_url from *models_config* for *model*.

        Returns (api_key, base_url) — either or both may be empty strings
        if the config does not contain a matching provider.
        """
        if not models_config:
            return "", ""
        providers = models_config.get("providers", {})
        # Try exact model-id match first.
        for _prov_name, prov in providers.items():
            if not isinstance(prov, dict):
                continue
            for m in prov.get("models", []):
                if isinstance(m, dict) and m.get("id") == model:
                    return prov.get("apiKey", ""), prov.get("baseUrl", "")
        # No exact match — fall back to the first (usually only) provider.
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return first.get("apiKey", ""), first.get("baseUrl", "")
        return "", ""

    @staticmethod
    def _map_thinking(thinking: str | None) -> dict | None:
        """Map the benchmark ``thinking`` value to a Hermes *reasoning_config* dict."""
        if thinking is None:
            return None
        t = thinking.strip().lower()
        if t in ("off", "none", "disabled", "false"):
            return {"enabled": False}
        if t in ("on", "enabled", "medium", "true"):
            return {"enabled": True, "effort": "medium"}
        if t == "high":
            return {"enabled": True, "effort": "high"}
        if t in ("low", "minimal"):
            return {"enabled": True, "effort": "low"}
        return {"enabled": True, "effort": t}

    # ------------------------------------------------------------------
    # Container setup helpers
    # ------------------------------------------------------------------

    def _start_container(
        self,
        task_id: str,
        workspace_path: str,
        api_key: str,
        base_url: str,
        extra_env: str = "",
        tmp_path: str = "",
        lobster_env: list[str] | None = None,
        bind_mounts: list[dict[str, Any]] | None = None,
    ) -> None:
        proxy_http = os.environ.get("HTTP_PROXY_INNER", "")
        proxy_https = os.environ.get("HTTPS_PROXY_INNER", "")
        env_args = [
            "-e", f"http_proxy={proxy_http}",
            "-e", f"https_proxy={proxy_https}",
            "-e", f"HTTP_PROXY={proxy_http}",
            "-e", f"HTTPS_PROXY={proxy_https}",
            "-e", f"BRAVE_API_KEY={self.brave_api_key}",
            "-e", f"OPENROUTER_API_KEY={api_key}",
            "-e", f"OPENROUTER_BASE_URL={base_url}",
            "-e", f"no_proxy={'' if not proxy_http else os.environ.get('NO_PROXY_INNER', '')}",
        ]
        for line in extra_env.splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "")
            env_args += ["-e", f"{key}={value}"]

        for key in (lobster_env or []):
            value = os.environ.get(key, "")
            if not value:
                continue
            env_args += ["-e", f"{key}={value}"]

        cmd = [
            "docker", "run", "-d",
            "--name", task_id,
            *build_host_gateway_args(),
            *env_args,
            "-v", f"{workspace_path}:/app:ro",
            *build_bind_mount_args(bind_mounts),
            self.image,
            "/bin/bash", "-c", "tail -f /dev/null",
        ]
        logger.info("[%s] Starting hermes-agent container (image=%s)", task_id, self.image)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"hermes-agent container startup failed:\n{r.stderr}")
        logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])

        if tmp_path and os.path.exists(tmp_path):
            subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"],
                capture_output=True,
            )
            cp_r = subprocess.run(
                ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"],
                capture_output=True, text=True,
            )
            if cp_r.returncode != 0:
                logger.error("[%s] Temp file copy failed: %s", task_id, cp_r.stderr)

    def _prepare_workspace(self, task_id: str) -> None:
        r = subprocess.run(
            [
                "docker", "exec", task_id, "/bin/bash", "-c",
                f"mkdir -p {TMP_WORKSPACE} && cp -r /app/. {TMP_WORKSPACE} && chmod -R u+w {TMP_WORKSPACE}",
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"hermes-agent workspace copy failed:\n{r.stderr}")

    def _patch_hermes_sources(self, task_id: str) -> None:
        """Overlay local Hermes source patches into the running container."""
        if not HERMES_SOURCE_PATCH_DIR.exists():
            logger.debug("[%s] Hermes source patch dir not found: %s", task_id, HERMES_SOURCE_PATCH_DIR)
            return

        r_mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", f"{HERMES_INSTALL_DIR}/tools"],
            capture_output=True,
            text=True,
        )
        if r_mkdir.returncode != 0:
            raise RuntimeError(f"hermes-agent patch mkdir failed:\n{r_mkdir.stderr}")

        copied_any = False
        for source_name, container_path in HERMES_SOURCE_PATCHES:
            source_path = HERMES_SOURCE_PATCH_DIR / source_name
            if not source_path.exists():
                logger.debug("[%s] Hermes source patch not found: %s", task_id, source_path)
                continue
            copied = subprocess.run(
                ["docker", "cp", str(source_path), f"{task_id}:{container_path}"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"hermes-agent source patch failed ({container_path}):\n{copied.stderr}")
            copied_any = True
            logger.info("[%s] Patched Hermes source: %s", task_id, container_path)

        if not copied_any:
            logger.debug("[%s] No Hermes source patches were applied", task_id)

    def _configure_hermes(
        self,
        task_id: str,
        api_key: str = "",
        base_url: str = "",
        runtime_options: dict[str, Any] | None = None,
    ) -> None:
        """Configure hermes-agent inside the container with one consistent provider config."""
        hermes_yaml = self._build_hermes_yaml(runtime_options)
        hermes_env = (
            f"OPENROUTER_API_KEY={api_key}\n"
            f"OPENROUTER_BASE_URL={base_url}\n"
            f"BRAVE_API_KEY={self.brave_api_key}\n"
        )

        with tempfile.TemporaryDirectory(prefix="hermes_config_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            yaml_host = tmp_root / "hermes.yaml"
            env_host = tmp_root / ".env"
            yaml_host.write_text(hermes_yaml, encoding="utf-8")
            env_host.write_text(hermes_env, encoding="utf-8")

            r_mkdir = subprocess.run(
                [
                    "docker",
                    "exec",
                    task_id,
                    "/bin/bash",
                    "-c",
                    f"mkdir -p {HERMES_HOME} && mkdir -p $(dirname {OPENCLAW_COMPAT_TRANSCRIPT_PATH})",
                ],
                capture_output=True,
                text=True,
            )
            if r_mkdir.returncode != 0:
                raise RuntimeError(f"hermes-agent config mkdir failed:\n{r_mkdir.stderr}")

            for src, dst in (
                (yaml_host, f"{HERMES_HOME}/hermes.yaml"),
                (yaml_host, f"{HERMES_HOME}/config.yaml"),
                (env_host, f"{HERMES_HOME}/.env"),
            ):
                copied = subprocess.run(
                    ["docker", "cp", str(src), f"{task_id}:{dst}"],
                    capture_output=True,
                    text=True,
                )
                if copied.returncode != 0:
                    raise RuntimeError(f"hermes-agent config copy failed ({dst}):\n{copied.stderr}")

            r_link = subprocess.run(
                ["docker", "exec", task_id, "ln", "-sfn", TMP_WORKSPACE, f"{HERMES_HOME}/workspace"],
                capture_output=True,
                text=True,
            )
            if r_link.returncode != 0:
                raise RuntimeError(f"hermes-agent workspace link failed:\n{r_link.stderr}")

        logger.info("[%s] hermes-agent configured", task_id)

    @staticmethod
    def _yaml_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if value is None:
            return "null"
        return json.dumps(str(value), ensure_ascii=False)

    @classmethod
    def _append_yaml_mapping(cls, lines: list[str], value: dict[str, Any], indent: int) -> None:
        prefix = " " * indent
        for key, item in value.items():
            if isinstance(item, dict):
                if item:
                    lines.append(f"{prefix}{key}:")
                    cls._append_yaml_mapping(lines, item, indent + 2)
                else:
                    lines.append(f"{prefix}{key}: {{}}")
            elif isinstance(item, list):
                lines.append(f"{prefix}{key}:")
                for element in item:
                    lines.append(f"{prefix}  - {cls._yaml_scalar(element)}")
            else:
                lines.append(f"{prefix}{key}: {cls._yaml_scalar(item)}")

    @classmethod
    def _build_hermes_yaml(cls, runtime_options: dict[str, Any] | None = None) -> str:
        lines = [
            "tools:",
            "  profile: coding",
            "  web:",
            "    search:",
            "      enabled: true",
            "      provider: brave",
        ]
        hermes_options = runtime_options.get("hermes", {}) if isinstance(runtime_options, dict) else {}
        mcp_servers = hermes_options.get("mcp_servers") if isinstance(hermes_options, dict) else None
        if isinstance(mcp_servers, dict) and mcp_servers:
            lines.append("mcp_servers:")
            cls._append_yaml_mapping(lines, mcp_servers, 2)
        return "\n".join(lines) + "\n"

    def _write_bench_runner(
        self,
        task_id: str,
        prompt: str,
        model: str,
        api_key: str,
        base_url: str,
        reasoning_config: dict | None,
        runtime_options: dict[str, Any] | None = None,
        post_task_prompt: str | None = None,
        resume_session_path: str | None = None,
        resume_without_prompt: bool = False,
        tools_enabled: bool = True,
    ) -> None:
        """Write the bench runner config into the container."""
        hermes_options = runtime_options.get("hermes", {}) if isinstance(runtime_options, dict) else {}
        max_iterations = 90
        if isinstance(hermes_options, dict) and hermes_options.get("max_iterations") is not None:
            try:
                max_iterations = int(hermes_options.get("max_iterations"))
            except (TypeError, ValueError):
                max_iterations = 90
        config_payload = {
            "config": {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "max_iterations": max_iterations,
                "reasoning_config": reasoning_config,
                "enabled_toolsets": hermes_options.get("enabled_toolsets") if isinstance(hermes_options, dict) else None,
                "disabled_toolsets": hermes_options.get("disabled_toolsets") if isinstance(hermes_options, dict) else None,
                "request_overrides": hermes_options.get("request_overrides") if isinstance(hermes_options, dict) else None,
                "context_mode": hermes_options.get("context_mode", "full") if isinstance(hermes_options, dict) else "full",
                "disable_context_compression": bool(
                    hermes_options.get("disable_context_compression", False)
                ) if isinstance(hermes_options, dict) else False,
            },
            "prompt": prompt,
            "resume_session_path": resume_session_path,
            "resume_without_prompt": bool(resume_without_prompt),
            "tools_enabled": bool(tools_enabled),
            "post_task_prompt": post_task_prompt or (
                hermes_options.get("post_task_prompt")
                if isinstance(hermes_options, dict)
                else None
            ),
        }

        config_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8",
            ) as f:
                json.dump(config_payload, f, ensure_ascii=False)
                config_tmp = f.name

            r = subprocess.run(
                ["docker", "cp", config_tmp, f"{task_id}:{BENCH_CONFIG_CONTAINER_PATH}"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"Failed to copy {BENCH_CONFIG_CONTAINER_PATH} into container:\n{r.stderr}")
        finally:
            for p in (config_tmp,):
                if p:
                    Path(p).unlink(missing_ok=True)

    @staticmethod
    def _read_primary_result(task_id: str, prompt: str) -> AgentPostTaskResult:
        result = subprocess.run(
            ["docker", "exec", task_id, "cat", PRIMARY_RESULT_CONTAINER_PATH],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=(result.stderr or "Hermes primary result was not written.").strip(),
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"Invalid Hermes primary result JSON: {exc}",
            )
        return AgentPostTaskResult.from_payload(prompt, payload)

    @staticmethod
    def _read_post_task_result(
        task_id: str,
        post_task_prompt: str | None,
    ) -> AgentPostTaskResult | None:
        prompt = str(post_task_prompt or "").strip()
        if not prompt:
            return None
        result = subprocess.run(
            ["docker", "exec", task_id, "cat", POST_TASK_RESULT_CONTAINER_PATH],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=(result.stderr or "Hermes post-task result was not written.").strip(),
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"Invalid Hermes post-task JSON: {exc}",
            )
        if isinstance(payload, dict) and "final_response" in payload:
            payload = dict(payload)
            payload["final_response"] = extract_skill_text(payload.get("final_response"))
        return AgentPostTaskResult.from_payload(prompt, payload)

    def _run_bench_runner_background(self, task_id: str, log_path: Path) -> subprocess.Popen[str]:
        if not BENCH_RUNNER_HOST_PATH.exists():
            raise RuntimeError(f"Hermes bench runner script not found: {BENCH_RUNNER_HOST_PATH}")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        script_file = BENCH_RUNNER_HOST_PATH.open("r", encoding="utf-8")
        proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                task_id,
                "/bin/bash",
                "-c",
                f"cd {TMP_WORKSPACE} && {HERMES_VENV_PYTHON} -",
            ],
            stdin=script_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        proc._log_file = log_file  # type: ignore[attr-defined]
        proc._script_file = script_file  # type: ignore[attr-defined]
        logger.info("[%s] Started Hermes bench runner PID=%s -> %s", task_id, proc.pid, log_path)
        return proc

    @staticmethod
    def _close_runner_streams(proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        for attr in ("_script_file", "_log_file"):
            stream = getattr(proc, attr, None)
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _cleanup_bench_config(task_id: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "rm", "-f", BENCH_CONFIG_CONTAINER_PATH],
            capture_output=True,
            text=True,
        )

    # ------------------------------------------------------------------
    # Transcript conversion (all sessions merged)
    # ------------------------------------------------------------------

    def _write_compat_transcript(self, task_id: str) -> None:
        """Convert Hermes session logs to OpenClaw-compatible JSONL for grading."""
        if not COMPAT_TRANSCRIPT_HOST_PATH.exists():
            logger.warning(
                "[%s] Compat transcript script not found: %s",
                task_id,
                COMPAT_TRANSCRIPT_HOST_PATH,
            )
            return

        with COMPAT_TRANSCRIPT_HOST_PATH.open("r", encoding="utf-8") as script_file:
            r = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    task_id,
                    HERMES_VENV_PYTHON,
                    "-",
                ],
                stdin=script_file,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        if r.returncode != 0:
            logger.warning("[%s] Compat transcript write failed: %s", task_id, r.stderr)
        else:
            logger.info("[%s] Compat transcript written to %s", task_id, OPENCLAW_COMPAT_TRANSCRIPT_PATH)

    # ------------------------------------------------------------------
    # Usage extraction (all sessions merged)
    # ------------------------------------------------------------------

    def _extract_usage_from_session_logs(self, task_id: str) -> dict[str, Any]:
        """Fallback: extract usage from copied Hermes session JSON files."""
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }

        with tempfile.TemporaryDirectory(prefix="hermes_usage_") as tmp_dir:
            sessions_host = Path(tmp_dir) / "sessions"
            sessions_host.mkdir(parents=True, exist_ok=True)
            copied = subprocess.run(
                ["docker", "cp", f"{task_id}:{HERMES_HOME}/sessions/.", str(sessions_host)],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                return usage

            total_requests = 0
            for session_file in sorted(sessions_host.glob("session_*.json"), key=lambda p: p.stat().st_mtime):
                try:
                    payload = json.loads(session_file.read_text(encoding="utf-8"))
                except Exception:
                    continue
                messages = payload.get("messages", [])
                if not isinstance(messages, list):
                    continue
                total_requests += sum(
                    1 for item in messages if isinstance(item, dict) and item.get("role") == "assistant"
                )
            usage["request_count"] = total_requests

        return usage

    def _usage_has_no_tokens(self, usage: dict[str, Any]) -> bool:
        return (
            usage.get("input_tokens", 0) == 0
            and usage.get("output_tokens", 0) == 0
            and usage.get("total_tokens", 0) == 0
        )

    def _extract_usage_from_agent_log(self, log_path: Path) -> dict[str, Any]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not log_path.exists():
            return usage

        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "API Response received" not in line or (
                "CompletionUsage(" not in line and "ResponseUsage(" not in line
            ):
                continue
            usage["request_count"] += 1
            usage["input_tokens"] += (
                self._extract_int_from_log(line, "prompt_tokens")
                or self._extract_int_from_log(line, "input_tokens")
            )
            usage["output_tokens"] += (
                self._extract_int_from_log(line, "completion_tokens")
                or self._extract_int_from_log(line, "output_tokens")
            )
            usage["total_tokens"] += self._extract_int_from_log(line, "total_tokens")
            usage["cache_read_tokens"] += self._extract_int_from_log(line, "cached_tokens")
            usage["cache_write_tokens"] += self._extract_int_from_log(line, "cache_write_tokens")
            usage["cost_usd"] += self._extract_float_from_log(line, "cost")

        usage["cost_usd"] = round(usage["cost_usd"], 6)
        return usage

    def _extract_int_from_log(self, line: str, field: str) -> int:
        match = re.search(rf"\b{re.escape(field)}=(\d+)", line)
        return int(match.group(1)) if match else 0

    def _extract_float_from_log(self, line: str, field: str) -> float:
        match = re.search(rf"\b{re.escape(field)}=([0-9.eE+-]+)", line)
        return float(match.group(1)) if match else 0.0

    def _copy_session_log(self, task_id: str, output_dir: Path) -> None:
        """Copy *all* hermes session logs from the container to the output directory."""
        hermes_log_dest = output_dir / "hermes_session"
        hermes_log_dest.mkdir(parents=True, exist_ok=True)

        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{HERMES_HOME}/sessions/.", str(hermes_log_dest)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            logger.info("[%s] Hermes session logs copied to %s", task_id, hermes_log_dest)
        else:
            logger.warning("[%s] Hermes session log copy failed: %s", task_id, r.stderr.strip())
