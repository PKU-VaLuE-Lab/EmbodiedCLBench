from __future__ import annotations

import json
import hashlib
import logging
import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from dotenv import load_dotenv

from tongbench_eval.agents.base import (
    AgentCheckpoint,
    AgentExecution,
    AgentPostTaskResult,
    AgentTaskSpec,
    BaseAgent,
)
from tongbench_eval.agents.claudecode.skill_text import extract_skill_text
from tongbench_eval.agents.claudecode.transcript import convert_claudecode_chat_to_openclaw_jsonl
from tongbench_eval.utils.docker_utils import (
    build_bind_mount_args,
    build_host_gateway_args,
    run_warmup,
    setup_skills,
    snapshot_workspace_state,
)
from tongbench_eval.utils.endpoint_utils import normalize_openrouter_base_url_for_claudecode
from tongbench_eval.utils.native_runtime import is_native_runtime

load_dotenv()

logger = logging.getLogger(__name__)
CLAUDECODE_SKILLS_DIR = "/root/.claude/skills"
CLAUDECODE_COMPAT_TRANSCRIPT_PATH = "/tmp/claudecode/openclaw_chat.jsonl"
CLAUDECODE_GATEWAY_PATH = "/tmp/tongbench_protocol_gateway.py"
CLAUDECODE_GATEWAY_LOG = "/tmp/tongbench_protocol_gateway.log"
MCP_WHEEL_HOST_DIR = "/tmp/tongbench_mcp_wheel"
MCP_WHEEL_CONTAINER_DIR = "/opt/tongbench_mcp_wheel"
OPENCLAW_COMPAT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"
CLAUDECODE_DEFAULT_DISALLOWED_TOOLS = [
    "Agent",
    "AskUserQuestion",
    "Bash",
    "Edit",
    "EnterPlanMode",
    "EnterWorktree",
    "ExitPlanMode",
    "ExitWorktree",
    "Glob",
    "Grep",
    "NotebookEdit",
    "Read",
    "Skill",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
]


class ClaudeCodeAgent(BaseAgent):
    def __init__(
        self,
        image: str | None = None,
        anthropic_api_key: str = "",
        anthropic_base_url: str = "",
        openrouter_base_url: str = "",
    ) -> None:
        self.image = (
            image
            or os.environ.get("DOCKER_IMAGE_CLAUDECODE")
            or os.environ.get("CLAUDECODE_DOCKER_IMAGE")
            or "wildclawbench-claudecode-ubuntu:v0.2"
        )
        explicit_api_key = anthropic_api_key.strip()
        self.api_key = explicit_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = normalize_openrouter_base_url_for_claudecode(
            openrouter_base_url or os.environ.get("OPENROUTER_BASE_URL", "")
        )
        explicit_base_url = anthropic_base_url.strip()
        self.api_base_url = explicit_base_url.rstrip("/") if explicit_base_url else self.openrouter_base_url

    @property
    def expects_gateway(self) -> bool:
        return False

    @property
    def transcript_container_path(self) -> str:
        return "/claude_code/log/chat.json"

    def prepare_grading_transcript(self, task_id: str) -> str:
        if not self._container_exists(task_id):
            return self.transcript_container_path
        with tempfile.TemporaryDirectory(prefix="claudecode_transcript_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            chat_host = tmp_root / "chat.json"
            compat_host = tmp_root / "chat.jsonl"

            r_cp = subprocess.run(
                ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(chat_host)],
                capture_output=True,
                text=True,
            )
            if r_cp.returncode != 0:
                logger.warning(
                    "[%s] Failed to copy ClaudeCode transcript for grading: %s",
                    task_id,
                    r_cp.stderr.strip(),
                )
                return self.transcript_container_path

            converted_count = convert_claudecode_chat_to_openclaw_jsonl(chat_host, compat_host)
            container_parent = str(PurePosixPath(CLAUDECODE_COMPAT_TRANSCRIPT_PATH).parent)
            r_mkdir = subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", container_parent],
                capture_output=True,
                text=True,
            )
            if r_mkdir.returncode != 0:
                logger.warning(
                    "[%s] Failed to create ClaudeCode compat transcript dir (%s): %s",
                    task_id,
                    container_parent,
                    r_mkdir.stderr.strip(),
                )
                return self.transcript_container_path

            r_push = subprocess.run(
                ["docker", "cp", str(compat_host), f"{task_id}:{CLAUDECODE_COMPAT_TRANSCRIPT_PATH}"],
                capture_output=True,
                text=True,
            )
            if r_push.returncode != 0:
                logger.warning(
                    "[%s] Failed to copy ClaudeCode compat transcript into container: %s",
                    task_id,
                    r_push.stderr.strip(),
                )
                return self.transcript_container_path

            openclaw_parent = str(PurePosixPath(OPENCLAW_COMPAT_TRANSCRIPT_PATH).parent)
            r_openclaw_mkdir = subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", openclaw_parent],
                capture_output=True,
                text=True,
            )
            if r_openclaw_mkdir.returncode == 0:
                r_openclaw_push = subprocess.run(
                    ["docker", "cp", str(compat_host), f"{task_id}:{OPENCLAW_COMPAT_TRANSCRIPT_PATH}"],
                    capture_output=True,
                    text=True,
                )
                if r_openclaw_push.returncode != 0:
                    logger.warning(
                        "[%s] Failed to copy ClaudeCode compat transcript to OpenClaw path: %s",
                        task_id,
                        r_openclaw_push.stderr.strip(),
                    )
            else:
                logger.warning(
                    "[%s] Failed to create OpenClaw compat transcript dir (%s): %s",
                    task_id,
                    openclaw_parent,
                    r_openclaw_mkdir.stderr.strip(),
                )

            logger.info(
                "[%s] ClaudeCode transcript normalized for grading (%d messages): %s",
                task_id,
                converted_count,
                CLAUDECODE_COMPAT_TRANSCRIPT_PATH,
            )
            return CLAUDECODE_COMPAT_TRANSCRIPT_PATH

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        elapsed_time = float(spec.timeout_seconds)
        start_time = time.perf_counter()
        task_id = spec.task_id
        gateway_base_url = None

        try:
            api_key, api_base_url = self._resolve_runtime_provider(spec.model, spec.models_config)
            self._validate_runtime_provider(spec.model, api_key, api_base_url)
            self._start_container(
                task_id,
                spec.workspace_path,
                spec.task,
                api_key=api_key,
                api_base_url=api_base_url,
            )
            gateway_base_url = self._maybe_start_dashscope_gateway(
                task_id=task_id,
                model=spec.model,
                api_base_url=api_base_url,
                api_key=api_key,
                thinking=spec.thinking,
            )
            self._prepare_workspace(task_id)
            self._copy_tmp_files(task_id, spec.workspace_path)
            self._patch_tongsim_mcp_server_annotations(task_id)
            self._ensure_python_mcp_runtime(task_id, spec.runtime_options)
            setup_skills(
                task_id,
                spec.task.get("skills", ""),
                spec.task.get("skills_path", ""),
                container_skills_root=CLAUDECODE_SKILLS_DIR,
            )
            run_warmup(task_id, spec.task.get("warmup", ""))
            snapshot_workspace_state(task_id)
            self._configure_mcp_servers(task_id, spec.runtime_options, spec.output_dir)
            resume_checkpoint = None
            if spec.resume_checkpoint is not None:
                resume_checkpoint = AgentCheckpoint.load(
                    spec.resume_checkpoint.path,
                    expected_backend="claudecode",
                )
                self._install_resume_checkpoint(task_id, resume_checkpoint)
            final_response = self._run_prompt(
                task_id,
                spec.prompt,
                spec.model,
                spec.timeout_seconds,
                spec.output_dir,
                spec.runtime_options,
                resume_session_id=(resume_checkpoint.session_id if resume_checkpoint else None),
                resume_session_file=(
                    self._native_resume_session_file(resume_checkpoint)
                    if resume_checkpoint is not None and is_native_runtime()
                    else None
                ),
                tools_enabled=bool(spec.tools_enabled),
                api_base_url_override=gateway_base_url,
            )
            post_task_result = self._run_post_task_prompt(
                task_id,
                spec.post_task_prompt,
                spec.model,
                spec.timeout_seconds,
                spec.output_dir,
                api_base_url_override=gateway_base_url,
            )
            elapsed_time = time.perf_counter() - start_time
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=None,
                agent_proc=None,
                post_task_result=post_task_result,
                final_response=final_response,
            )
        except subprocess.TimeoutExpired as exc:
            logger.info("[%s] ClaudeCode timed out...", task_id)
            try:
                spec.output_dir.mkdir(parents=True, exist_ok=True)
                (spec.output_dir / "timeout_stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
                (spec.output_dir / "timeout_stderr.txt").write_text(exc.stderr or "", encoding="utf-8")
                subprocess.run(["docker", "cp", f"{task_id}:/tmp_workspace/claude_debug.log", str(spec.output_dir / "claude_debug.log")], capture_output=True, text=True)
            except Exception:
                pass
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error="ClaudeCode run timed out",
                gateway_proc=None,
                agent_proc=None,
            )
        except Exception as exc:
            elapsed_time = time.perf_counter() - start_time
            logger.error("[%s] ClaudeCode execution error: %s", task_id, exc)
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=str(exc),
                gateway_proc=None,
                agent_proc=None,
            )
        finally:
            if gateway_base_url:
                self._copy_file_from_container(
                    task_id, CLAUDECODE_GATEWAY_LOG, spec.output_dir / "protocol_gateway.log"
                )

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
        native_dir = checkpoint_path / "native" / "projects"
        native_dir.mkdir(parents=True, exist_ok=True)
        copied = subprocess.run(
            ["docker", "cp", f"{task_id}:/root/.claude/projects/.", str(native_dir)],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"ClaudeCode checkpoint export failed: {copied.stderr.strip()}")
        session_files = [
            path
            for path in native_dir.rglob("*.jsonl")
            if "subagents" not in path.parts
        ]
        session_files.sort(key=lambda path: path.stat().st_mtime)
        if not session_files:
            raise FileNotFoundError("ClaudeCode checkpoint contains no native project session JSONL.")
        session_id = session_files[-1].stem
        return AgentCheckpoint.create(
            backend="claudecode",
            path=checkpoint_path,
            session_id=session_id,
            metadata=metadata,
        )

    @staticmethod
    def _install_resume_checkpoint(task_id: str, checkpoint: AgentCheckpoint) -> None:
        projects_dir = checkpoint.path / "native" / "projects"
        mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", "/root/.claude/projects"],
            capture_output=True,
            text=True,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(f"ClaudeCode checkpoint restore mkdir failed: {mkdir.stderr.strip()}")
        copied = subprocess.run(
            ["docker", "cp", f"{projects_dir}/.", f"{task_id}:/root/.claude/projects/"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"ClaudeCode checkpoint restore failed: {copied.stderr.strip()}")

    @staticmethod
    def _native_resume_session_file(checkpoint: AgentCheckpoint) -> str:
        """Return the native checkpoint JSONL path for file-based resume.

        Native tasks have a unique host-side working directory per episode.
        Claude Code derives its session directory from that directory, so a
        session ID alone cannot locate a checkpoint copied from another task.
        Passing the transcript path bypasses that project-directory lookup.
        """

        candidates = [
            path
            for path in (checkpoint.path / "native" / "projects").rglob("*.jsonl")
            if "subagents" not in path.parts
        ]
        for candidate in candidates:
            if candidate.stem == checkpoint.session_id:
                return str(candidate.resolve())
        if len(candidates) == 1:
            return str(candidates[0].resolve())
        raise FileNotFoundError(
            f"ClaudeCode native checkpoint has no session JSONL for {checkpoint.session_id}: "
            f"{checkpoint.path}"
        )

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
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
        if not self._container_exists(task_id):
            return usage

        log_dest = output_dir / "claude_code_log"
        log_dest.mkdir(parents=True, exist_ok=True)
        self._copy_file_from_container(task_id, "/claude_code/log/usage.json", log_dest / "usage.json")
        self._copy_file_from_container(task_id, "/claude_code/log/chat.json", log_dest / "chat.json")
        self._copy_dir_from_container(task_id, "/claude_code/log/.", log_dest)
        self._sync_agent_log_from_claude_logs(task_id, output_dir, log_dest)

        parsed = self._extract_usage_from_chat_json(log_dest / "chat.json")
        if parsed["request_count"] == 0:
            parsed = self._extract_usage_from_usage_json(log_dest / "usage.json")
        if parsed["request_count"] == 0:
            parsed["request_count"] = self._extract_request_count_from_chat_json(log_dest / "chat.json")
        if parsed["request_count"] == 0:
            fallback = self._extract_usage_from_logs(log_dest)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "total_tokens",
                "cost_usd",
                "request_count",
            ):
                parsed_value = parsed.get(key, 0)
                fallback_value = fallback.get(key, 0)
                if (parsed_value is None or parsed_value <= 0) and fallback_value > 0:
                    parsed[key] = fallback_value

        usage.update(parsed)
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def _extract_usage_from_chat_json(self, chat_path: Path) -> dict[str, Any]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not chat_path.exists():
            return totals

        try:
            content = chat_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return totals

        payloads: list[Any] = []
        try:
            parsed = json.loads(content)
            payloads = parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            for line in content.splitlines():
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        for payload in payloads:
            self._accumulate_costed_usage(payload, totals)

        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_tokens"]
            + totals["cache_write_tokens"]
        )
        if totals["request_count"] > 0 and totals["cost_usd"] == 0.0:
            totals["cost_usd"] = self._estimate_cost(totals)
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        return totals

    def _accumulate_costed_usage(self, payload: Any, totals: dict[str, Any]) -> None:
        if isinstance(payload, list):
            for item in payload:
                self._accumulate_costed_usage(item, totals)
            return
        if not isinstance(payload, dict):
            return

        if (
            "input_tokens" in payload
            and "output_tokens" in payload
            and ("cost_details" in payload or "cost" in payload)
        ):
            input_tokens = int(payload.get("input_tokens") or 0)
            output_tokens = int(payload.get("output_tokens") or 0)
            cache_read_tokens = int(payload.get("cache_read_input_tokens") or 0)
            cache_write_tokens = int(payload.get("cache_creation_input_tokens") or 0)
            cost_details = payload.get("cost_details")
            cost = self._num(
                cost_details.get("upstream_inference_cost") if isinstance(cost_details, dict) else payload.get("cost"),
                default=0.0,
            )

            if (
                input_tokens == 0
                and output_tokens == 0
                and cache_read_tokens == 0
                and cache_write_tokens == 0
                and cost == 0
            ):
                return

            totals["input_tokens"] += input_tokens
            totals["output_tokens"] += output_tokens
            totals["cache_read_tokens"] += cache_read_tokens
            totals["cache_write_tokens"] += cache_write_tokens
            totals["cost_usd"] += cost
            totals["request_count"] += 1
            return

        for value in payload.values():
            self._accumulate_costed_usage(value, totals)

    def _sync_agent_log_from_claude_logs(self, task_id: str, output_dir: Path, log_dest: Path) -> None:
        candidates = (
            log_dest / "agent.log",
            log_dest / "chat.json",
            log_dest / "chat.jsonl",
        )
        for src in candidates:
            if not src.exists() or not src.is_file():
                continue
            try:
                content = src.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning("[%s] Failed to read ClaudeCode log source %s: %s", task_id, src, exc)
                continue
            if not content.strip():
                continue
            try:
                (output_dir / "agent.log").write_text(content, encoding="utf-8")
            except Exception as exc:
                logger.warning("[%s] Failed to write agent.log from %s: %s", task_id, src, exc)
                return
            logger.info("[%s] agent.log synced from %s", task_id, src)
            return

    def _resolve_runtime_provider(self, model: str, models_config: dict | None) -> tuple[str, str]:
        api_key = self.api_key
        base_url = self.api_base_url
        config_key, config_base_url = self._resolve_provider_config(model, models_config)
        if config_key:
            api_key = config_key
        if config_base_url:
            base_url = config_base_url.rstrip("/")
        return api_key, base_url

    @staticmethod
    def _resolve_provider_config(model: str, models_config: dict | None) -> tuple[str, str]:
        if not models_config:
            return "", ""
        providers = models_config.get("providers", {})
        for _provider_name, provider in providers.items():
            if not isinstance(provider, dict):
                continue
            for item in provider.get("models", []) or []:
                if isinstance(item, dict) and item.get("id") == model:
                    return (
                        ClaudeCodeAgent._provider_value(provider, "apiKey", "api_key"),
                        ClaudeCodeAgent._provider_value(provider, "baseUrl", "base_url"),
                    )
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return (
                    ClaudeCodeAgent._provider_value(first, "apiKey", "api_key"),
                    ClaudeCodeAgent._provider_value(first, "baseUrl", "base_url"),
                )
        return "", ""

    @staticmethod
    def _provider_value(provider: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = provider.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _validate_runtime_provider(model: str, api_key: str, base_url: str) -> None:
        if not api_key:
            raise RuntimeError(
                "ClaudeCode API key is empty. Provide --api-key, or set a non-empty apiKey in "
                "--models-config. If the config uses ${DASHSCOPE_API_KEY}, set that environment "
                "variable before loading the config."
            )
        normalized_base = str(base_url or "").rstrip("/")
        if "dashscope.aliyuncs.com/compatible-mode" in normalized_base.lower() and not str(model).lower().startswith("zhipu/"):
            raise RuntimeError(
                "ClaudeCode cannot call DashScope OpenAI-compatible endpoints directly. "
                "Claude Code uses the Anthropic Messages API, while "
                f"{normalized_base} is an OpenAI-compatible /chat/completions API. "
                "Use an Anthropic/OpenRouter-compatible ClaudeCode model such as "
                "openrouter/anthropic/claude-sonnet-4.6, or run this Qwen/DashScope model "
                "through hermesagent/codex instead. "
                f"Rejected model/base_url: {model} @ {normalized_base}"
            )

    @staticmethod
    def _gateway_port(task_id: str) -> int:
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        first = 21000 + int.from_bytes(digest[:4], "big") % 20000
        for offset in range(20000):
            port = 21000 + ((first - 21000 + offset) % 20000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
        raise RuntimeError("Unable to allocate a ClaudeCode DashScope gateway port")

    def _maybe_start_dashscope_gateway(
        self,
        *,
        task_id: str,
        model: str,
        api_base_url: str,
        api_key: str,
        thinking: str | None = None,
    ) -> str | None:
        if "dashscope.aliyuncs.com/compatible-mode" not in str(api_base_url).lower():
            return None
        if not str(model).lower().startswith("zhipu/"):
            return None
        source = (Path(__file__).parent.parent / "protocol_gateway.py").resolve()
        if not source.is_file():
            raise RuntimeError(f"Missing protocol gateway source: {source}")
        copied = subprocess.run(
            ["docker", "cp", str(source), f"{task_id}:{CLAUDECODE_GATEWAY_PATH}"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"ClaudeCode DashScope gateway copy failed: {copied.stderr or copied.stdout}")
        port = self._gateway_port(task_id)
        env_prefix = " ".join(
            [
                "env",
                f"TONGBENCH_GATEWAY_BASE_URL={shlex.quote(api_base_url)}",
                f"TONGBENCH_GATEWAY_API_KEY={shlex.quote(api_key)}",
                f"TONGBENCH_GATEWAY_MODEL={shlex.quote(model)}",
                f"TONGBENCH_GATEWAY_REASONING_EFFORT={shlex.quote(thinking or '')}",
            ]
        )
        start_cmd = (
            f"nohup {env_prefix} python3 {CLAUDECODE_GATEWAY_PATH} --host 127.0.0.1 --port {port} "
            f"> {CLAUDECODE_GATEWAY_LOG} 2>&1 </dev/null & echo $!"
        )
        started = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", start_cmd],
            capture_output=True,
            text=True,
        )
        if started.returncode != 0:
            raise RuntimeError(f"ClaudeCode DashScope gateway failed to start: {started.stderr or started.stdout}")
        health_url = f"http://127.0.0.1:{port}/health"
        for _ in range(20):
            health = subprocess.run(
                [
                    "docker",
                    "exec",
                    task_id,
                    "/bin/bash",
                    "-lc",
                    f"python3 -c 'import urllib.request; urllib.request.urlopen({json.dumps(health_url)}, timeout=1)'",
                ],
                capture_output=True,
                text=True,
            )
            if health.returncode == 0:
                return f"http://127.0.0.1:{port}"
            time.sleep(0.2)
        raise RuntimeError("ClaudeCode DashScope gateway did not become healthy")

    @classmethod
    def _render_mcp_config(cls, runtime_options: dict[str, Any] | None) -> dict[str, Any]:
        mcp_servers = cls._runtime_mcp_servers(runtime_options)
        if not mcp_servers:
            return {}
        rendered: dict[str, Any] = {"mcpServers": {}}
        for server_name, server in mcp_servers.items():
            if not isinstance(server, dict):
                continue
            command = str(server.get("command") or "").strip()
            if not command:
                continue
            server_config: dict[str, Any] = {
                "type": str(server.get("type") or "stdio"),
                "command": command,
                "args": [str(value) for value in server.get("args", []) or []],
                "alwaysLoad": bool(server.get("alwaysLoad", True)),
            }
            timeout_value = server.get("timeout")
            if timeout_value is not None:
                server_config["timeout"] = cls._seconds_to_milliseconds(timeout_value)
            env = server.get("env") if isinstance(server.get("env"), dict) else {}
            if env:
                server_config["env"] = {str(key): str(value) for key, value in env.items()}
            rendered["mcpServers"][str(server_name)] = server_config
        return rendered if rendered["mcpServers"] else {}

    @classmethod
    def _allowed_mcp_tools(cls, runtime_options: dict[str, Any] | None) -> list[str]:
        claudecode_options = cls._runtime_claudecode_options(runtime_options)
        configured = claudecode_options.get("allowed_tools")
        if isinstance(configured, list) and configured:
            return [str(item) for item in configured if str(item).strip()]
        return [f"mcp__{server_name}__*" for server_name in cls._runtime_mcp_servers(runtime_options)]

    @classmethod
    def _disallowed_tools(cls, runtime_options: dict[str, Any] | None) -> list[str]:
        claudecode_options = cls._runtime_claudecode_options(runtime_options)
        configured = claudecode_options.get("disallowed_tools")
        if isinstance(configured, list):
            return [str(item) for item in configured if str(item).strip()]
        if claudecode_options.get("disable_builtin_tools", True) is False:
            return []
        if cls._runtime_mcp_servers(runtime_options):
            return list(CLAUDECODE_DEFAULT_DISALLOWED_TOOLS)
        return []

    @staticmethod
    def _container_mcp_config_path(runtime_options: dict[str, Any] | None) -> str:
        claudecode_options = ClaudeCodeAgent._runtime_claudecode_options(runtime_options)
        configured = str(claudecode_options.get("mcp_config_path") or "").strip()
        return configured or "/tmp_workspace/.tongsim_claudecode_mcp.json"

    @staticmethod
    def _runtime_claudecode_options(runtime_options: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(runtime_options, dict):
            return {}
        claudecode_options = runtime_options.get("claudecode", {})
        return claudecode_options if isinstance(claudecode_options, dict) else {}

    @staticmethod
    def _runtime_mcp_servers(runtime_options: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(runtime_options, dict):
            return {}
        claudecode_options = runtime_options.get("claudecode", {})
        if isinstance(claudecode_options, dict) and isinstance(claudecode_options.get("mcp_servers"), dict):
            return claudecode_options["mcp_servers"]
        hermes_options = runtime_options.get("hermes", {})
        if isinstance(hermes_options, dict) and isinstance(hermes_options.get("mcp_servers"), dict):
            return hermes_options["mcp_servers"]
        if isinstance(runtime_options.get("mcp_servers"), dict):
            return runtime_options["mcp_servers"]
        return {}

    @staticmethod
    def _seconds_to_milliseconds(value: object) -> int:
        return int(float(value) * 1000)

    def _copy_file_from_container(self, task_id: str, src: str, dest: Path) -> None:
        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{src}", str(dest)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] ClaudeCode file copy failed (%s): %s", task_id, src, r.stderr.strip())

    def _copy_dir_from_container(self, task_id: str, src: str, dest: Path) -> None:
        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{src}", str(dest)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] ClaudeCode log dir copy failed: %s", task_id, r.stderr.strip())

    @staticmethod
    def _container_exists(task_id: str) -> bool:
        r = subprocess.run(
            ["docker", "inspect", task_id],
            capture_output=True,
            text=True,
        )
        return r.returncode == 0

    def _start_container(
        self,
        task_id: str,
        workspace_path: str,
        task: dict[str, Any] | None = None,
        *,
        api_key: str | None = None,
        api_base_url: str | None = None,
    ) -> None:
        proxy_http = os.environ.get("HTTP_PROXY_INNER", "")
        proxy_https = os.environ.get("HTTPS_PROXY_INNER", "")
        resolved_api_key = api_key if api_key is not None else self.api_key
        resolved_api_base_url = api_base_url if api_base_url is not None else self.api_base_url
        resolved_openrouter_base_url = (
            normalize_openrouter_base_url_for_claudecode(resolved_api_base_url)
            if resolved_api_base_url
            else self.openrouter_base_url
        )
        native_bundle = os.environ.get("TONGBENCH_NATIVE_BUNDLE_ROOT", "").strip()
        cli_path = (
            f"{native_bundle}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            if is_native_runtime() and native_bundle
            else "/root/miniconda3/envs/eval/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        env_map = {
            "ANTHROPIC_API_KEY": resolved_api_key,
            "ANTHROPIC_BASE_URL": resolved_api_base_url,
            # Third-party Anthropic-compatible gateways may expose a model name
            # that Claude Code does not recognize by default.
            "ANTHROPIC_CUSTOM_MODEL_OPTION": os.environ.get("ANTHROPIC_CUSTOM_MODEL_OPTION", ""),
            "OPENROUTER_API_KEY": resolved_api_key,
            "OPENROUTER_BASE_URL": resolved_openrouter_base_url,
            "DISABLE_PROMPT_CACHING": os.environ.get("DISABLE_PROMPT_CACHING", "1"),
            "DISABLE_INTERLEAVED_THINKING": os.environ.get("DISABLE_INTERLEAVED_THINKING", "1"),
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": os.environ.get("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1"),
            "ENABLE_TOOL_SEARCH": os.environ.get("ENABLE_TOOL_SEARCH", "true"),
            "IS_SANDBOX": os.environ.get("IS_SANDBOX", "1"),
            "CLAUDE_CODE_FULL_LOG_PATH": os.environ.get("CLAUDE_CODE_FULL_LOG_PATH", "./log"),
            "http_proxy": proxy_http,
            "https_proxy": proxy_https,
            "HTTP_PROXY": proxy_http,
            "HTTPS_PROXY": proxy_https,
            "PATH": cli_path,
        }
        env_args: list[str] = []
        for key, value in env_map.items():
            if value:
                env_args += ["-e", f"{key}={value}"]

        exec_path = os.path.join(workspace_path, "exec")
        os.makedirs(exec_path, exist_ok=True)
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
            *build_bind_mount_args((task or {}).get("bind_mounts")),
            "--entrypoint",
            "/bin/bash",
            self.image,
            "-lc",
            "tail -f /dev/null",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ClaudeCode container startup failed:\n{r.stderr}")

    def _patch_claudecode_runtime(self, task_id: str) -> None:
        patch_cmd = r"""python3 -u - <<'PY'
import re
import subprocess
from pathlib import Path

path = Path("/claude_code/src/tasks/LocalAgentTask/LocalAgentTask.tsx")
text = path.read_text(encoding="utf-8")
old = "  const usage = message.message.usage;\n  // Keep latest input (it's cumulative in the API), sum outputs\n"
new = '''  const usage = message.message.usage ?? {
    input_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    output_tokens: 0,
  };
  // Keep latest input (it's cumulative in the API), sum outputs
'''
if old in text and new not in text:
    path.write_text(text.replace(old, new), encoding="utf-8")

query_engine = Path("/claude_code/src/QueryEngine.ts")
query_text = query_engine.read_text(encoding="utf-8")
patched_query_text = query_text
engine_marker = "const engine = new QueryEngine({"
inject_code = r'''const __tongsimQueryMcpPath = "/tmp_workspace/.tongsim_claudecode_mcp.json";
let __tongsimMcpClients = mcpClients;
let __tongsimTools = tools;
const __tongsimUniqByName = (__tongsimItems) => {
  const __tongsimSeen = new Set();
  return __tongsimItems.filter((__tongsimItem) => {
    const __tongsimName = String(__tongsimItem?.name ?? "");
    if (!__tongsimName || __tongsimSeen.has(__tongsimName)) {
      return false;
    }
    __tongsimSeen.add(__tongsimName);
    return true;
  });
};
try {
  process.env.MCP_CONNECTION_NONBLOCKING = "0";
  process.env.MCP_CONNECT_TIMEOUT_MS ??= "60000";
  const __tongsimFs = await import("fs");
  if (__tongsimFs.existsSync(__tongsimQueryMcpPath)) {
    const { connectToServer, fetchToolsForClient } = await import("./services/mcp/client.js");
    const __tongsimMcpConfig = JSON.parse(__tongsimFs.readFileSync(__tongsimQueryMcpPath, "utf8"));
    const __tongsimMcpServers = __tongsimMcpConfig.mcpServers ?? {};
    const __tongsimAgentMcpClients = [];
    const __tongsimMcpTools = [];
    for (const [__tongsimName, __tongsimServer] of Object.entries(__tongsimMcpServers)) {
      if (!__tongsimServer || typeof __tongsimServer !== "object") {
        continue;
      }
      const __tongsimConfig = {
        ...__tongsimServer,
        type: __tongsimServer.type ?? "stdio",
        alwaysLoad: true,
        name: __tongsimName,
        scope: "dynamic",
      };
      const __tongsimClient = await connectToServer(__tongsimName, __tongsimConfig);
      __tongsimAgentMcpClients.push(__tongsimClient);
      if (__tongsimClient.type === "connected") {
        const __tongsimToolsForClient = await fetchToolsForClient(__tongsimClient);
        __tongsimMcpTools.push(...__tongsimToolsForClient);
        console.error(`[TongSIM] connected MCP server ${__tongsimName} with ${__tongsimToolsForClient.length} tool(s)`);
      } else {
        console.error(`[TongSIM] MCP server ${__tongsimName} not connected: ${__tongsimClient.type}`);\n        console.error("[TongSIM] MCP failed client detail:", __tongsimClient);
      }
    }
    if (Object.keys(__tongsimMcpServers).length > 0 && __tongsimMcpTools.length === 0) {
      throw new Error(`[TongSIM] MCP config loaded but no MCP tools were registered from: ${Object.keys(__tongsimMcpServers).join(", ")}`);
    }
    __tongsimMcpClients = __tongsimUniqByName([...mcpClients, ...__tongsimAgentMcpClients]);
    __tongsimTools = __tongsimMcpTools.length > 0
      ? __tongsimUniqByName([...tools, ...__tongsimMcpTools])
      : tools;
    console.error(`[TongSIM] QueryEngine MCP tools: ${__tongsimMcpTools.map((tool) => tool.name).join(", ")}`);
  }
} catch (__tongsimMcpError) {
  console.error(`[TongSIM] MCP QueryEngine initialization failed: ${__tongsimMcpError?.stack ?? __tongsimMcpError}`);
  throw __tongsimMcpError;
}
'''
engine_index = patched_query_text.find(engine_marker)
if engine_index < 0:
    raise RuntimeError("ClaudeCode MCP patch failed: QueryEngine constructor marker not found")
if "__tongsimQueryMcpPath" not in patched_query_text:
    patched_query_text = patched_query_text[:engine_index] + inject_code + patched_query_text[engine_index:]
    engine_index += len(inject_code)
engine_args_marker = "const engine = new QueryEngine({\n    cwd,\n    tools,\n    commands,\n    mcpClients,"
engine_args_replacement = (
    "const engine = new QueryEngine({\n"
    "    cwd,\n"
    "    tools: __tongsimTools,\n"
    "    commands,\n"
    "    mcpClients: __tongsimMcpClients,"
)
if "tools: __tongsimTools," not in patched_query_text:
    engine_args_index = patched_query_text.find(engine_args_marker, engine_index, engine_index + 3000)
    if engine_args_index < 0:
        raise RuntimeError(
            "ClaudeCode MCP patch failed: QueryEngine args marker not found\n"
            + patched_query_text[engine_index:engine_index + 3000]
        )
    patched_query_text = (
        patched_query_text[:engine_args_index]
        + engine_args_replacement
        + patched_query_text[engine_args_index + len(engine_args_marker):]
    )
if "__tongsimQueryMcpPath" not in patched_query_text:
    raise RuntimeError("ClaudeCode MCP patch failed: MCP initializer not injected")
if "tools: __tongsimTools," not in patched_query_text:
    raise RuntimeError("ClaudeCode MCP patch failed: QueryEngine tools not patched")
if "mcpClients: __tongsimMcpClients," not in patched_query_text:
    raise RuntimeError("ClaudeCode MCP patch failed: QueryEngine clients not patched")
if "fetchToolsForClient" not in patched_query_text or "connectToServer" not in patched_query_text:
    raise RuntimeError("ClaudeCode MCP patch failed: MCP client helpers not referenced")
if patched_query_text != query_text:
    query_engine.write_text(patched_query_text, encoding="utf-8")

print("[TongSIM] skipped broken MCP image content adapter patch; use direct content-block return path instead")

def _disable_oversized_tool_result_storage(text: str, needle: str) -> tuple[str, bool]:
    def _matching_index(open_index: int, open_char: str, close_char: str) -> int:
        depth = 0
        for index in range(open_index, len(text)):
            char = text[index]
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    return index
        return -1

    def _overflow_guard_bounds(needle_index: int) -> tuple[int, int]:
        if_matches = list(re.finditer(r"\bif\s*\(", text[:needle_index]))
        for match in reversed(if_matches):
            if_index = match.start()
            paren_index = text.find("(", if_index)
            end_index = _matching_index(paren_index, "(", ")")
            if end_index < 0:
                continue

            block_start = text.find("{", end_index, needle_index)
            if block_start >= 0:
                block_end = _matching_index(block_start, "{", "}")
                if block_end < 0 or block_end >= needle_index:
                    return if_index, end_index
        return -1, -1

    changed = False
    search_from = 0
    while True:
        needle_index = text.find(needle, search_from)
        if needle_index < 0:
            break
        if_index, end_index = _overflow_guard_bounds(needle_index)
        if if_index < 0:
            context = text[max(0, needle_index - 500):needle_index + 500].replace("\n", "\\n")
            print("[TongSIM] ClaudeCode image result patch skipped marker without enclosing if:", context)
            search_from = needle_index + len(needle)
            continue

        if "TongSIM direct image tool results" in text[if_index:end_index + 1]:
            search_from = needle_index + len(needle)
            continue

        text = (
            text[:if_index]
            + "if (false /* TongSIM direct image tool results: bypass large-result file storage */)"
            + text[end_index + 1:]
        )
        changed = True
        search_from = needle_index + len(needle)
    return text, changed

storage_needle = "__TONGSIM_SKIP_OPTIONAL_STORAGE_PATCH__"
storage_root = Path("/claude_code/src")
storage_candidate_paths = [
    storage_root / "utils" / "toolResultStorage.ts",
    storage_root / "utils" / "mcpOutputStorage.ts",
    storage_root / "tools" / "MCPTool" / "MCPTool.ts",
    storage_root / "tools" / "MCPTool" / "MCPTool.tsx",
    storage_root / "QueryEngine.ts",
    storage_root / "query.ts",
]
try:
    grep_result = subprocess.run(
        [
            "grep",
            "-RIl",
            "--exclude-dir=node_modules",
            "--exclude=*.map",
            storage_needle,
            str(storage_root),
        ],
        capture_output=True,
        text=True,
        timeout=8,
    )
    if grep_result.returncode in (0, 1):
        storage_candidate_paths.extend(
            Path(line.strip())
            for line in grep_result.stdout.splitlines()
            if line.strip()
        )
    else:
        print("[TongSIM] ClaudeCode storage grep failed:", grep_result.stderr or grep_result.stdout)
except Exception as storage_grep_error:
    print("[TongSIM] ClaudeCode storage grep skipped:", storage_grep_error)

storage_candidates = []
seen_storage_paths = set()
for storage_path in storage_candidate_paths:
    if storage_path in seen_storage_paths:
        continue
    seen_storage_paths.add(storage_path)
    if storage_path.suffix not in {".ts", ".tsx", ".js", ".jsx"} or not storage_path.exists():
        continue
    try:
        if storage_path.stat().st_size > 2_000_000:
            print("[TongSIM] skipping oversized ClaudeCode storage candidate:", storage_path)
            continue
        storage_text = storage_path.read_text(encoding="utf-8")
    except OSError:
        continue
    if storage_needle in storage_text:
        storage_candidates.append((storage_path, storage_text))

patched_storage_paths = []
if not storage_candidates:
    print("[TongSIM] large-result storage marker not found; skip storage patch")
else:
    for storage_path, storage_text in storage_candidates:
        patched_storage_text, storage_changed = _disable_oversized_tool_result_storage(
            storage_text,
            storage_needle,
        )
        if storage_changed:
            if "TongSIM direct image tool results" not in patched_storage_text:
                print(f"[TongSIM] ClaudeCode large tool-result storage patch produced no marker for {storage_path}; skipped")
                continue
            storage_path.write_text(patched_storage_text, encoding="utf-8")
            patched_storage_paths.append(str(storage_path))

    if not patched_storage_paths:
        print("[TongSIM] ClaudeCode large tool-result storage patch did not find an enclosing overflow branch; skip storage patch")
    else:
        print(
            "[TongSIM] patched ClaudeCode large tool-result storage:",
            ", ".join(patched_storage_paths),
        )
PY"""
        try:
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c", patch_cmd],
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "ClaudeCode runtime patch timed out after 240s\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            ) from exc
        if r.stdout:
            logger.info("[%s] ClaudeCode runtime patch stdout:\n%s", task_id, r.stdout.strip())
        if r.returncode != 0:
            raise RuntimeError(f"ClaudeCode runtime patch failed:\n{r.stderr or r.stdout}")
        if r.stderr:
            logger.info("[%s] ClaudeCode runtime patch stderr:\n%s", task_id, r.stderr.strip())

    def _prepare_workspace(self, task_id: str) -> None:
        r = subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "/bin/bash",
                "-lc",
                r"""
set -e
mkdir -p /tmp_workspace
cp -r /workspace/. /tmp_workspace
if [ -d /tmp_workspace/exec ]; then
  if ! find /tmp_workspace -maxdepth 2 -path /tmp_workspace/exec -prune -o -name tongsim_mcp_server.py -print -quit | grep -q .; then
    cp -r /tmp_workspace/exec/. /tmp_workspace
  fi
fi
chmod -R u+w /tmp_workspace
""".strip(),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ClaudeCode workspace copy failed:\n{r.stderr}")

    def _patch_tongsim_mcp_server_annotations(self, task_id: str) -> None:
        patch_cmd = r"""python3 -u - <<'PY'
from pathlib import Path

workspace = Path("/tmp_workspace")
if not workspace.exists():
    raise SystemExit(0)

scripts = sorted(
    {
        *workspace.rglob("tongsim_mcp_server.py"),
        *workspace.rglob("tongsim_sequence_mcp_server.py"),
    }
)
if not scripts:
    print("[TongSIM] no TongSIM MCP server scripts found under /tmp_workspace")
    raise SystemExit(0)

patched = []
for path in scripts:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        continue
    original = text
    # Avoid postponed string annotations on generated MCP servers. The Python
    # MCP package is also patched separately, but keeping annotations concrete
    # makes the generated servers less sensitive to FastMCP version changes.
    text = text.replace("from __future__ import annotations\n\n", "")
    text = text.replace("from __future__ import annotations\n", "")

    if text != original:
        path.write_text(text, encoding="utf-8")
        patched.append(str(path))

print("[TongSIM] patched MCP server annotations:", ", ".join(patched) if patched else "already concrete")
PY"""
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", patch_cmd],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ClaudeCode TongSIM MCP server annotation patch failed:\n{r.stderr or r.stdout}")

    def _copy_tmp_files(self, task_id: str, workspace_path: str) -> None:
        tmp_path = Path(workspace_path) / "tmp"
        if not tmp_path.exists():
            return
        r_mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"],
            capture_output=True,
            text=True,
        )
        if r_mkdir.returncode != 0:
            raise RuntimeError(f"ClaudeCode tmp directory setup failed:\n{r_mkdir.stderr}")
        r_cp = subprocess.run(
            ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode != 0:
            raise RuntimeError(f"ClaudeCode tmp copy failed:\n{r_cp.stderr}")

    def _run_prompt(
        self,
        task_id: str,
        prompt: str,
        model: str,
        timeout_seconds: int,
        output_dir: Path,
        runtime_options: dict[str, Any] | None = None,
        resume_session_id: str | None = None,
        resume_session_file: str | None = None,
        tools_enabled: bool = True,
        api_base_url_override: str | None = None,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        claudecode_options = self._runtime_claudecode_options(runtime_options)
        mcp_config_path = self._container_mcp_config_path(runtime_options)
        mcp_config = self._render_mcp_config(runtime_options)
        mcp_servers = mcp_config.get("mcpServers", {}) if isinstance(mcp_config, dict) else {}
        allowed_tools: list[str] = []
        allowed_flags = ""
        disallowed_tools = self._disallowed_tools(runtime_options)
        mcp_flags = ""
        mcp_config_arg = ""
        if mcp_servers and tools_enabled:
            mcp_config_arg = json.dumps(mcp_config, separators=(",", ":"), ensure_ascii=False)
            allowed_tools = self._allowed_mcp_tools(runtime_options)
            if allowed_tools:
                allowed_flags = f"--allowedTools {shlex.quote(','.join(allowed_tools))} "
            disallowed_flags = ""
            if disallowed_tools:
                disallowed_flags = f"--disallowedTools {shlex.quote(','.join(disallowed_tools))} "
            mcp_flags = (
                f"--mcp-config {shlex.quote(mcp_config_arg)} "
                f"--strict-mcp-config "
                f"--permission-mode dontAsk "
                f"{allowed_flags}"
                f"{disallowed_flags}"
            )
        max_turns = claudecode_options.get("max_iterations")
        max_turns_flag = ""
        if max_turns is not None:
            max_turns_flag = f"--max-turns {int(max_turns)} "
        if not tools_enabled:
            max_turns_flag = "--max-turns 1 "
        resume_flags = ""
        if resume_session_file:
            resume_flags = f"--resume {shlex.quote(resume_session_file)} "
        elif resume_session_id:
            resume_flags = f"--resume {shlex.quote(resume_session_id)} "
        output_format = "json" if resume_session_id and not tools_enabled else "stream-json"
        tools_flags = "--strict-mcp-config --tools '' " if not tools_enabled else ""
        cmd = (
            f"cd /claude_code && "
            f"export PATH=\"$HOME/.bun/bin:$PATH\" && "
            f"export CLAUDE_CODE_FULL_LOG=\"${{CLAUDE_CODE_FULL_LOG:-1}}\" && "
            f"{('export ANTHROPIC_BASE_URL=' + shlex.quote(api_base_url_override) + ' && ') if api_base_url_override else ''}"
            f"bun src/entrypoints/cli.tsx --output-format {output_format} --verbose --debug --debug-file /tmp_workspace/claude_debug.log "
            f"--add-dir /tmp_workspace "
            f"{mcp_flags}"
            f"{tools_flags}"
            f"{max_turns_flag}"
            f"{resume_flags}"
            f"-p {shlex.quote(prompt)} "
            f"--model {shlex.quote(model)}"
        )
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        (output_dir / "agent.log").write_text(
            (r.stdout or "") + ("\n" if r.stdout else "") + (r.stderr or ""),
            encoding="utf-8",
        )
        self._write_run_debug(
            output_dir,
            {
                "task_id": task_id,
                "returncode": r.returncode,
                "model": model,
                "prompt_chars": len(prompt),
                "mcp_config_path": mcp_config_path if mcp_servers else "",
                "mcp_config_mode": "runtime_inline_json" if mcp_servers else "",
                "mcp_config_arg_chars": len(mcp_config_arg),
                "mcp_server_names": list(mcp_servers.keys()),
                "allowed_tools": allowed_tools,
                "allowed_tools_filter_enabled": bool(allowed_flags),
                "disallowed_tools": disallowed_tools,
                "max_turns": max_turns,
                "resume_session_id": resume_session_id or "",
                "tools_enabled": bool(tools_enabled),
                "stdout_tail": self._tail_text(r.stdout),
                "stderr_tail": self._tail_text(r.stderr),
            },
        )
        if r.returncode != 0:
            diagnostics = self._collect_failure_diagnostics(task_id, output_dir)
            details = (r.stderr or r.stdout or diagnostics).strip()
            hint = f"See {output_dir / 'agent.log'} and {output_dir / 'claudecode_failure_diagnostics.txt'}"
            if details:
                raise RuntimeError(f"ClaudeCode run failed (rc={r.returncode}). {hint}\n{details}")
            raise RuntimeError(f"ClaudeCode run failed (rc={r.returncode}). {hint}")
        final_response, _ = self._extract_post_task_response(r.stdout)
        return final_response

    def _run_post_task_prompt(
        self,
        task_id: str,
        post_task_prompt: str | None,
        model: str,
        timeout_seconds: int,
        output_dir: Path,
        api_base_url_override: str | None = None,
    ) -> AgentPostTaskResult | None:
        prompt = str(post_task_prompt or "").strip()
        if not prompt:
            return None
        cmd = (
            "cd /claude_code && "
            'export PATH="$HOME/.bun/bin:$PATH" && '
            'export CLAUDE_CODE_FULL_LOG="${CLAUDE_CODE_FULL_LOG:-1}" && '
            f"{('export ANTHROPIC_BASE_URL=' + shlex.quote(api_base_url_override) + ' && ') if api_base_url_override else ''}"
            "bun src/entrypoints/cli.tsx --output-format json --verbose "
            "--add-dir /tmp_workspace --continue --strict-mcp-config "
            "--permission-mode dontAsk --tools '' --max-turns 1 "
            f"-p {shlex.quote(prompt)} --model {shlex.quote(model)}"
        )
        try:
            result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"ClaudeCode post-task summary timed out after {timeout_seconds} seconds.",
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        combined = (result.stdout or "") + ("\n" if result.stdout else "") + (result.stderr or "")
        (output_dir / "post_task.log").write_text(combined, encoding="utf-8")
        if result.returncode != 0:
            return AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"ClaudeCode post-task summary failed (rc={result.returncode}): {combined[-4000:]}",
            )
        final_response, api_calls = self._extract_post_task_response(result.stdout)
        final_response = extract_skill_text(final_response)
        if not final_response:
            return AgentPostTaskResult.from_payload(
                prompt,
                {"api_calls": api_calls},
                error="ClaudeCode post-task summary returned no assistant text.",
            )
        return AgentPostTaskResult.from_payload(
            prompt,
            {
                "final_response": final_response,
                "completed": True,
                "api_calls": api_calls,
            },
        )

    @staticmethod
    def _extract_post_task_response(raw_output: str | None) -> tuple[str, int]:
        text = str(raw_output or "").strip()
        if not text:
            return "", 0
        payloads: list[Any] = []

        def append_payload(parsed: Any) -> None:
            if isinstance(parsed, list):
                payloads.extend(parsed)
            else:
                payloads.append(parsed)

        try:
            append_payload(json.loads(text))
        except json.JSONDecodeError:
            for line in text.splitlines():
                try:
                    append_payload(json.loads(line))
                except json.JSONDecodeError:
                    continue
        final_response = ""
        api_calls = 0
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            candidate = payload.get("result")
            if isinstance(candidate, str) and candidate.strip():
                final_response = candidate.strip()
            message = payload.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    final_response = content.strip()
                elif isinstance(content, list):
                    chunks = [
                        str(item.get("text") or "").strip()
                        for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    if any(chunks):
                        final_response = "\n\n".join(chunk for chunk in chunks if chunk)
            try:
                api_calls = max(api_calls, int(payload.get("num_turns") or 0))
            except (TypeError, ValueError):
                pass
        return final_response, api_calls or (1 if final_response else 0)

    @staticmethod
    def _tail_text(text: str | None, *, limit: int = 12000) -> str:
        if not text:
            return ""
        return text[-limit:]

    @staticmethod
    def _write_run_debug(output_dir: Path, payload: dict[str, Any]) -> None:
        try:
            (output_dir / "claudecode_run_debug.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write ClaudeCode run debug file: %s", exc)

    def _collect_failure_diagnostics(self, task_id: str, output_dir: Path) -> str:
        diagnostics: list[str] = []
        commands = [
            (
                "container_paths",
                "ls -la /claude_code 2>&1; "
                "ls -la /claude_code/log 2>&1 || true; "
                "ls -la /tmp_workspace 2>&1 | head -120",
            ),
            (
                "start_help",
                "cd /claude_code && (./start.sh --help || true) 2>&1 | head -200",
            ),
            (
                "candidate_logs",
                "find /claude_code /root/.claude /tmp "
                "-maxdepth 5 -type f "
                "\\( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \\) "
                "-print 2>/dev/null | head -200",
            ),
            (
                "recent_log_tails",
                "for f in $(find /claude_code /root/.claude /tmp "
                "-maxdepth 5 -type f "
                "\\( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \\) "
                "-print 2>/dev/null | head -20); do "
                "echo '===== '\"$f\"; tail -n 80 \"$f\" 2>&1; done",
            ),
        ]
        for label, shell_cmd in commands:
            result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", shell_cmd],
                capture_output=True,
                text=True,
            )
            diagnostics.append(f"## {label} (rc={result.returncode})")
            diagnostics.append(self._tail_text((result.stdout or "") + (result.stderr or "")))
            diagnostics.append("")
        content = "\n".join(diagnostics).strip()
        try:
            (output_dir / "claudecode_failure_diagnostics.txt").write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.warning("[%s] Failed to write ClaudeCode failure diagnostics: %s", task_id, exc)
        return content

    def _ensure_python_mcp_runtime(
        self,
        task_id: str,
        runtime_options: dict[str, Any] | None,
    ) -> None:
        mcp_servers = self._runtime_mcp_servers(runtime_options)
        if not mcp_servers:
            return
        if os.environ.get("CLAUDECODE_SKIP_MCP_PYTHON_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
            logger.info("[%s] Skipping ClaudeCode Python MCP dependency check by env override", task_id)
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
            if probe_result.returncode != 0:
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
                        "ClaudeCode MCP server dependency is missing and automatic install failed. "
                        f"Command: {python_command} -m pip install 'mcp==1.16.0'\n"
                        f"{install_result.stderr or install_result.stdout}"
                    )
                probe_result = subprocess.run(
                    ["docker", "exec", task_id, "/bin/bash", "-c", probe],
                    capture_output=True,
                    text=True,
                )
                if probe_result.returncode != 0:
                    raise RuntimeError(
                        "ClaudeCode MCP server dependency is installed but still cannot be imported. "
                        f"Command: {python_command}\n"
                        f"{probe_result.stderr or probe_result.stdout}"
                    )
            self._patch_python_fastmcp_runtime(task_id, python_command)

    def _patch_python_fastmcp_runtime(self, task_id: str, python_command: str) -> None:
        patch = (
            f"{shlex.quote(python_command)} - <<'PY'\n"
            "from pathlib import Path\n"
            "import mcp.server.fastmcp.tools.base as base\n"
            "\n"
            "path = Path(base.__file__)\n"
            "text = path.read_text(encoding='utf-8')\n"
            "original = text\n"
            "\n"
            "if 'import inspect' not in text:\n"
            "    if 'import json' in text:\n"
            "        text = text.replace('import json', 'import json\\nimport inspect', 1)\n"
            "    else:\n"
            "        lines = text.splitlines()\n"
            "        insert_at = 0\n"
            "        while insert_at < len(lines) and (not lines[insert_at].strip() or lines[insert_at].startswith('from __future__')):\n"
            "            insert_at += 1\n"
            "        lines.insert(insert_at, 'import inspect')\n"
            "        text = '\\n'.join(lines) + ('\\n' if original.endswith('\\n') else '')\n"
            "\n"
            "old = 'if issubclass(param.annotation, Context):'\n"
            "new = 'if inspect.isclass(param.annotation) and issubclass(param.annotation, Context):'\n"
            "if old in text:\n"
            "    text = text.replace(old, new)\n"
            "elif new not in text:\n"
            "    print('[TongSIM] FastMCP issubclass guard marker not found; leaving package unchanged:', path)\n"
            "    raise SystemExit(0)\n"
            "\n"
            "if text != original:\n"
            "    path.write_text(text, encoding='utf-8')\n"
            "    print('[TongSIM] patched FastMCP issubclass guard:', path)\n"
            "else:\n"
            "    print('[TongSIM] FastMCP issubclass guard already patched:', path)\n"
            "PY"
        )
        result = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", patch],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ClaudeCode FastMCP runtime patch failed. "
                f"Command: {python_command}\n{result.stderr or result.stdout}"
            )

    def _configure_mcp_servers(
        self,
        task_id: str,
        runtime_options: dict[str, Any] | None,
        output_dir: Path,
    ) -> None:
        mcp_config = self._render_mcp_config(runtime_options)
        if not mcp_config:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "claudecode_mcp_config.json").write_text(
            json.dumps(mcp_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        server_names = list(mcp_config.get("mcpServers", {}))
        approval_settings = {
            "enableAllProjectMcpServers": True,
            "enabledMcpjsonServers": server_names,
            "disabledMcpjsonServers": [],
        }
        (output_dir / "claudecode_settings.json").write_text(
            json.dumps(approval_settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        config_tmp = None
        settings_tmp = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
                json.dump(mcp_config, tmp_file, indent=2, ensure_ascii=False)
                config_tmp = tmp_file.name
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
                json.dump(approval_settings, tmp_file, indent=2, ensure_ascii=False)
                settings_tmp = tmp_file.name
            container_path = self._container_mcp_config_path(runtime_options)
            if not container_path:
                return
            mkdir_result = subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", str(PurePosixPath(container_path).parent)],
                capture_output=True,
                text=True,
            )
            if mkdir_result.returncode != 0:
                raise RuntimeError(f"ClaudeCode MCP config mkdir failed:\n{mkdir_result.stderr}")
            copied = subprocess.run(
                ["docker", "cp", config_tmp, f"{task_id}:{container_path}"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"ClaudeCode MCP config copy failed:\n{copied.stderr}")
            for project_config_path in ("/claude_code/.mcp.json", "/tmp_workspace/.mcp.json"):
                copied_project = subprocess.run(
                    ["docker", "cp", config_tmp, f"{task_id}:{project_config_path}"],
                    capture_output=True,
                    text=True,
                )
                if copied_project.returncode != 0:
                    raise RuntimeError(
                        f"ClaudeCode project MCP config copy failed ({project_config_path}):\n"
                        f"{copied_project.stderr}"
                    )
            for settings_path in (
                "/root/.claude/settings.json",
                "/claude_code/.claude/settings.json",
                "/tmp_workspace/.claude/settings.json",
            ):
                mkdir_settings_result = subprocess.run(
                    ["docker", "exec", task_id, "mkdir", "-p", str(PurePosixPath(settings_path).parent)],
                    capture_output=True,
                    text=True,
                )
                if mkdir_settings_result.returncode != 0:
                    raise RuntimeError(f"ClaudeCode settings mkdir failed:\n{mkdir_settings_result.stderr}")
                copied_settings = subprocess.run(
                    ["docker", "cp", settings_tmp, f"{task_id}:{settings_path}"],
                    capture_output=True,
                    text=True,
                )
                if copied_settings.returncode != 0:
                    raise RuntimeError(
                        f"ClaudeCode settings copy failed ({settings_path}):\n"
                        f"{copied_settings.stderr}"
                    )
        finally:
            if config_tmp:
                Path(config_tmp).unlink(missing_ok=True)
            if settings_tmp:
                Path(settings_tmp).unlink(missing_ok=True)
        logger.info("[%s] Configured %d ClaudeCode MCP server(s)", task_id, len(mcp_config.get("mcpServers", {})))

    def _extract_usage_from_logs(self, log_dir: Path) -> dict[str, Any]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not log_dir.exists():
            return totals

        for file_path in log_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".json", ".jsonl", ".log", ".txt"}:
                continue
            self._accumulate_from_file(file_path, totals)

        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_tokens"]
            + totals["cache_write_tokens"]
        )
        totals["cost_usd"] = round(self._estimate_cost(totals), 6)
        return totals

    def _extract_usage_from_usage_json(self, usage_path: Path) -> dict[str, Any]:
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not usage_path.exists():
            return totals

        try:
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
        except Exception:
            return totals

        if not isinstance(payload, dict):
            return totals

        # Support both known schemas:
        # 1) totalInputTokens / totalOutputTokens / totalCostUSD / modelUsage
        # 2) total_cost_usd + nested usage.{input_tokens,output_tokens,...}
        usage_block = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}

        totals["input_tokens"] = int(
            self._num(
                payload.get("totalInputTokens", usage_block.get("input_tokens")),
            )
        )
        totals["output_tokens"] = int(
            self._num(
                payload.get("totalOutputTokens", usage_block.get("output_tokens")),
            )
        )
        totals["cache_read_tokens"] = int(
            self._num(
                payload.get("totalCacheReadInputTokens", usage_block.get("cache_read_input_tokens")),
            )
        )
        totals["cache_write_tokens"] = int(
            self._num(
                payload.get("totalCacheCreationInputTokens", usage_block.get("cache_creation_input_tokens")),
            )
        )
        totals["cost_usd"] = round(
            self._num(
                payload.get("totalCostUSD", payload.get("total_cost_usd")),
                default=0.0,
            ),
            6,
        )

        totals["total_tokens"] = (
            totals["input_tokens"]
            + totals["output_tokens"]
            + totals["cache_read_tokens"]
            + totals["cache_write_tokens"]
        )

        totals["request_count"] = self._request_count_from_model_usage(payload.get("modelUsage"))
        return totals

    def _request_count_from_model_usage(self, model_usage: Any) -> int:
        if isinstance(model_usage, list):
            total = 0
            for item in model_usage:
                if not isinstance(item, dict):
                    continue
                total += int(
                    self._num(
                        item.get("requestCount", item.get("requests", item.get("count"))),
                        default=0,
                    )
                )
            return total

        if isinstance(model_usage, dict):
            total = 0
            for value in model_usage.values():
                if not isinstance(value, dict):
                    continue
                total += int(
                    self._num(
                        value.get("requestCount", value.get("requests", value.get("count"))),
                        default=0,
                    )
                )
            return total

        return 0

    def _extract_request_count_from_chat_json(self, chat_path: Path) -> int:
        if not chat_path.exists():
            return 0
        try:
            content = chat_path.read_text(encoding="utf-8")
        except Exception:
            return 0

        try:
            payload = json.loads(content)
        except Exception:
            payload = None

        if isinstance(payload, list):
            assistant_messages = [
                m
                for m in payload
                if isinstance(m, dict)
                and str(m.get("role", "")).lower() == "assistant"
            ]
            return len(assistant_messages)

        if isinstance(payload, dict):
            if str(payload.get("event", "")).lower() == "query_start":
                return 1
            return 0

        count = 0
        for line in content.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and str(row.get("event", "")).lower() == "query_start":
                count += 1
        if count > 0:
            return count
        return 0

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

    def _accumulate_from_file(self, file_path: Path, totals: dict[str, Any]) -> None:
        try:
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return

        for line in lines:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = self._find_usage(payload)
            if usage is None:
                continue
            totals["request_count"] += 1
            totals["input_tokens"] += int(usage.get("input_tokens", usage.get("input", 0)) or 0)
            totals["output_tokens"] += int(usage.get("output_tokens", usage.get("output", 0)) or 0)
            totals["cache_read_tokens"] += int(
                usage.get("cache_read_input_tokens", usage.get("cacheRead", 0)) or 0
            )
            totals["cache_write_tokens"] += int(
                usage.get("cache_creation_input_tokens", usage.get("cacheWrite", 0)) or 0
            )

    def _find_usage(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if "usage" in payload and isinstance(payload["usage"], dict):
            return payload["usage"]
        message = payload.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            return message["usage"]
        return None

    def _estimate_cost(self, totals: dict[str, Any]) -> float:
        input_price = float(os.environ.get("CLAUDECODE_INPUT_PRICE_PER_MTOK", "0"))
        output_price = float(os.environ.get("CLAUDECODE_OUTPUT_PRICE_PER_MTOK", "0"))
        cache_read_price = float(os.environ.get("CLAUDECODE_CACHE_READ_PRICE_PER_MTOK", "0"))
        cache_write_price = float(os.environ.get("CLAUDECODE_CACHE_WRITE_PRICE_PER_MTOK", "0"))
        return (
            totals["input_tokens"] / 1_000_000 * input_price
            + totals["output_tokens"] / 1_000_000 * output_price
            + totals["cache_read_tokens"] / 1_000_000 * cache_read_price
            + totals["cache_write_tokens"] / 1_000_000 * cache_write_price
        )
