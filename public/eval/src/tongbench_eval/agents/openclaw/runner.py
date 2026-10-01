from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from tongbench_eval.agents.base import (
    AgentCheckpoint,
    AgentExecution,
    AgentPostTaskResult,
    AgentTaskSpec,
    BaseAgent,
)
from tongbench_eval.agents.openclaw.skill_text import extract_skill_text
from tongbench_eval.utils.grading import extract_usage_from_jsonl
from tongbench_eval.utils.docker_utils import (
    inject_lobster_workspace,
    inject_openclaw_models,
    run_background,
    run_warmup,
    setup_skills,
    setup_workspace,
    start_container,
)
from tongbench_eval.utils.native_runtime import is_native_runtime, native_path

load_dotenv()

logger = logging.getLogger(__name__)

OPENCLAW_TONGSIM_MCP_PLUGIN_ID = "tongsim-mcp-bridge"
OPENCLAW_TONGSIM_MCP_PLUGIN_PATH = "/root/.openclaw/extensions/tongsim-mcp-bridge"
TONGSIM_MCP_TOOL_NAMES = (
    "observe_current_state",
    "choose_action",
    "check_task_status",
)


class OpenClawAgent(BaseAgent):
    def __init__(
        self,
        gateway_port: int,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        image_model: str | None = None,
    ) -> None:
        self.gateway_port = gateway_port
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.image_model = image_model if image_model is not None else os.environ.get("OPENCLAW_IMAGE_MODEL", "").strip()

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return "/root/.openclaw/agents/main/sessions/chat.jsonl"

    def _gateway_port_for_task(self, task_id: str) -> int:
        if not is_native_runtime():
            return self.gateway_port

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
        raise RuntimeError("Unable to allocate an OpenClaw gateway port")

    @staticmethod
    def _gateway_token_for_task(task_id: str) -> str:
        return hashlib.sha256(f"tongbench-openclaw:{task_id}".encode("utf-8")).hexdigest()

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = float(spec.timeout_seconds)
        timed_out = False

        try:
            task_run_id = f"{spec.task_id}:{spec.output_dir.resolve()}"
            gateway_port = self._gateway_port_for_task(task_run_id)
            gateway_token = self._gateway_token_for_task(task_run_id)
            runtime_api_key, runtime_base_url = self._resolve_runtime_provider(
                spec.model,
                spec.models_config,
            )
            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            start_container(
                spec.task_id,
                exec_path,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
                bind_mounts=spec.task.get("bind_mounts"),
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            setup_workspace(spec.task_id, thinking=spec.thinking)
            setup_skills(spec.task_id, spec.task.get("skills", ""), spec.task.get("skills_path", ""))
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            if spec.models_config:
                inject_openclaw_models(spec.task_id, spec.models_config)

            self._ensure_python_mcp_runtime(spec.task_id, spec.runtime_options)
            self._configure_mcp_servers(spec.task_id, spec.runtime_options, spec.output_dir)

            runtime_model = self._resolve_runtime_model_ref(spec.model, spec.models_config)
            self._set_model(spec.task_id, runtime_model)
            self._inject_openrouter_key(spec.task_id, runtime_api_key)
            image_model = self._resolve_runtime_model_ref(self.image_model or spec.model, spec.models_config)
            self._set_image_model(spec.task_id, image_model)

            if spec.resume_checkpoint is not None:
                checkpoint = AgentCheckpoint.load(
                    spec.resume_checkpoint.path,
                    expected_backend="openclaw",
                )
                self._install_resume_checkpoint(spec.task_id, checkpoint)
            if not spec.tools_enabled:
                self._disable_tools_for_post_task(spec.task_id)

            gateway_flags = "--allow-unconfigured " if is_native_runtime() else ""
            gateway_proc = run_background(
                spec.task_id,
                bash_cmd=(
                    f"export OPENROUTER_API_KEY='{runtime_api_key}' && "
                    f"export OPENROUTER_BASE_URL='{runtime_base_url}' && "
                    f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)} && "
                    f"openclaw gateway {gateway_flags}--port {gateway_port}"
                ),
                log_path=spec.output_dir / "gateway.log",
            )
            logger.info("[%s] Waiting for gateway to be ready (2s)...", spec.task_id)
            time.sleep(2)

            safe_prompt = spec.prompt.replace("'", "'\\''")
            json_flag = "--json " if spec.resume_checkpoint is not None else ""
            start_time = time.perf_counter()
            agent_proc = run_background(
                spec.task_id,
                bash_cmd=(
                    f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:{gateway_port} && "
                    f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)} && "
                    f"openclaw agent --session-id chat --timeout {spec.timeout_seconds} "
                    f"{json_flag}--message '{safe_prompt}'"
                ),
                log_path=spec.output_dir / "agent.log",
            )

            logger.info("[%s] Waiting for agent to finish...", spec.task_id)
            try:
                agent_proc.wait(timeout=spec.timeout_seconds)
                elapsed_time = time.perf_counter() - start_time
                logger.info(
                    "[%s] Agent finished successfully, elapsed: %.2f seconds",
                    spec.task_id,
                    elapsed_time,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                logger.info("[%s] Agent timed out...", spec.task_id)
                elapsed_time = float(spec.timeout_seconds)
                agent_proc.kill()
                agent_proc.wait()

            logger.info("[%s] Agent exit code: %s", spec.task_id, agent_proc.returncode)
            final_response = self._extract_openclaw_response_text(
                (spec.output_dir / "agent.log").read_text(encoding="utf-8", errors="replace")
                if (spec.output_dir / "agent.log").exists()
                else ""
            )
            post_task_result, replacement_gateway_proc = self._run_post_task_prompt(
                task_id=spec.task_id,
                post_task_prompt=spec.post_task_prompt,
                timeout_seconds=spec.timeout_seconds,
                output_dir=spec.output_dir,
                gateway_proc=gateway_proc,
                runtime_api_key=runtime_api_key,
                runtime_base_url=runtime_base_url,
                gateway_port=gateway_port,
                gateway_token=gateway_token,
            )
            if replacement_gateway_proc is not None:
                gateway_proc = replacement_gateway_proc
            if timed_out:
                execution_error = f"OpenClaw agent timed out after {spec.timeout_seconds:g} seconds"
            elif agent_proc.returncode != 0:
                execution_error = f"OpenClaw agent exited with code {agent_proc.returncode}"
            else:
                execution_error = None
            return AgentExecution(
                elapsed_time=elapsed_time,
                error=execution_error,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
                post_task_result=post_task_result,
                final_response=final_response,
            )
        except Exception as exc:
            logger.error("[%s] Execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    def export_checkpoint(
        self,
        task_id: str,
        checkpoint_dir: Path,
        *,
        metadata: dict | None = None,
    ) -> AgentCheckpoint:
        checkpoint_path = checkpoint_dir.resolve()
        if checkpoint_path.exists() and any(checkpoint_path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
        native_dir = checkpoint_path / "native" / "sessions"
        native_dir.mkdir(parents=True, exist_ok=True)
        copied = subprocess.run(
            [
                "docker",
                "cp",
                f"{task_id}:/root/.openclaw/agents/main/sessions/.",
                str(native_dir),
            ],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"OpenClaw checkpoint export failed: {copied.stderr.strip()}")
        if not any(path.is_file() for path in native_dir.rglob("*")):
            raise FileNotFoundError("OpenClaw checkpoint contains no native session files.")
        return AgentCheckpoint.create(
            backend="openclaw",
            path=checkpoint_path,
            session_id="chat",
            metadata=metadata,
        )

    @staticmethod
    def _install_resume_checkpoint(task_id: str, checkpoint: AgentCheckpoint) -> None:
        sessions_dir = checkpoint.path / "native" / "sessions"
        container_dir = "/root/.openclaw/agents/main/sessions"
        mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", container_dir],
            capture_output=True,
            text=True,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(f"OpenClaw checkpoint restore mkdir failed: {mkdir.stderr.strip()}")
        copied = subprocess.run(
            ["docker", "cp", f"{sessions_dir}/.", f"{task_id}:{container_dir}/"],
            capture_output=True,
            text=True,
        )
        if copied.returncode != 0:
            raise RuntimeError(f"OpenClaw checkpoint restore failed: {copied.stderr.strip()}")

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict:
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
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "request_count": 0,
            }
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def _run_post_task_prompt(
        self,
        *,
        task_id: str,
        post_task_prompt: str | None,
        timeout_seconds: int,
        output_dir: Path,
        gateway_proc: subprocess.Popen[str] | None,
        runtime_api_key: str,
        runtime_base_url: str,
        gateway_port: int,
        gateway_token: str,
    ) -> tuple[AgentPostTaskResult | None, subprocess.Popen[str] | None]:
        prompt = str(post_task_prompt or "").strip()
        if not prompt:
            return None, gateway_proc
        try:
            self._disable_tools_for_post_task(task_id)
            self._stop_gateway(task_id, gateway_proc)
            post_task_gateway_log = output_dir / "post_task_gateway.log"
            replacement_gateway = run_background(
                task_id,
                bash_cmd=(
                    f"export OPENROUTER_API_KEY={shlex.quote(runtime_api_key)} && "
                    f"export OPENROUTER_BASE_URL={shlex.quote(runtime_base_url)} && "
                    f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)} && "
                    f"openclaw gateway --port {gateway_port}"
                ),
                log_path=post_task_gateway_log,
            )
            self._wait_for_gateway_ready(replacement_gateway, post_task_gateway_log)
            cmd = (
                f"export OPENCLAW_GATEWAY_URL=ws://127.0.0.1:{gateway_port} && "
                f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)} && "
                f"openclaw agent --session-id chat --timeout {int(timeout_seconds)} "
                f"--json --message {shlex.quote(prompt)}"
            )
            result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", cmd],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return (
                AgentPostTaskResult.from_payload(
                    prompt,
                    None,
                    error=f"OpenClaw post-task summary timed out after {timeout_seconds} seconds.",
                ),
                locals().get("replacement_gateway", gateway_proc),
            )
        except Exception as exc:
            return (
                AgentPostTaskResult.from_payload(
                    prompt,
                    None,
                    error=f"OpenClaw post-task summary setup failed: {type(exc).__name__}: {exc}",
                ),
                locals().get("replacement_gateway", gateway_proc),
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        combined = (result.stdout or "") + ("\n" if result.stdout else "") + (result.stderr or "")
        (output_dir / "post_task.log").write_text(combined, encoding="utf-8")
        if result.returncode != 0:
            post_result = AgentPostTaskResult.from_payload(
                prompt,
                None,
                error=f"OpenClaw post-task summary failed (rc={result.returncode}): {combined[-4000:]}",
            )
        else:
            final_response = self._extract_openclaw_response_text(result.stdout)
            if final_response:
                post_result = AgentPostTaskResult.from_payload(
                    prompt,
                    {
                        "final_response": final_response,
                        "completed": True,
                        "api_calls": 1,
                    },
                )
            else:
                post_result = AgentPostTaskResult.from_payload(
                    prompt,
                    None,
                    error="OpenClaw post-task summary returned no assistant text.",
                )
        return post_result, replacement_gateway

    @staticmethod
    def _disable_tools_for_post_task(task_id: str) -> None:
        command = """python3 - <<'PY'
import json
from pathlib import Path

path = Path('/root/.openclaw/openclaw.json')
config = json.loads(path.read_text()) if path.exists() else {}
tools = config.setdefault('tools', {})
tools['profile'] = 'minimal'
tools['allow'] = []
tools['deny'] = ['*']
plugins = config.setdefault('plugins', {})
entry = plugins.setdefault('entries', {}).setdefault('tongsim-mcp-bridge', {})
entry['enabled'] = False
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
PY"""
        result = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "Failed to disable OpenClaw tools.")

    @staticmethod
    def _stop_gateway(task_id: str, gateway_proc: subprocess.Popen[str] | None) -> None:
        stop_result = subprocess.run(
            [
                "docker",
                "exec",
                task_id,
                "/bin/bash",
                "-lc",
                """
pkill -TERM -f '^openclaw-gateway($| )' 2>/dev/null || true
for _ in $(seq 1 40); do
  pgrep -f '^openclaw-gateway($| )' >/dev/null || exit 0
  sleep 0.25
done
pkill -KILL -f '^openclaw-gateway($| )' 2>/dev/null || true
pgrep -f '^openclaw-gateway($| )' >/dev/null && exit 1
exit 0
""",
            ],
            capture_output=True,
            text=True,
        )
        if stop_result.returncode != 0:
            raise RuntimeError(
                stop_result.stderr or stop_result.stdout or "Failed to stop the OpenClaw gateway."
            )
        if gateway_proc is None:
            return
        try:
            gateway_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            gateway_proc.terminate()

    @staticmethod
    def _wait_for_gateway_ready(
        gateway_proc: subprocess.Popen[str],
        log_path: Path,
        timeout_seconds: float = 15.0,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_log = ""
        while time.monotonic() < deadline:
            if log_path.exists():
                last_log = log_path.read_text(encoding="utf-8", errors="replace")
                if "listening on ws://" in last_log:
                    return
            returncode = gateway_proc.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"OpenClaw post-task gateway exited before becoming ready (rc={returncode}): "
                    f"{last_log[-4000:]}"
                )
            time.sleep(0.25)
        raise RuntimeError(
            f"OpenClaw post-task gateway was not ready within {timeout_seconds:g} seconds: "
            f"{last_log[-4000:]}"
        )

    @classmethod
    def _extract_openclaw_response_text(cls, raw_output: str | None) -> str:
        return extract_skill_text(raw_output)

    def _ensure_python_mcp_runtime(
        self,
        task_id: str,
        runtime_options: dict | None,
    ) -> None:
        mcp_servers = self._runtime_mcp_servers(runtime_options)
        if not mcp_servers:
            return
        if os.environ.get("OPENCLAW_SKIP_MCP_PYTHON_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
            logger.info("[%s] Skipping OpenClaw Python MCP dependency check by env override", task_id)
            return

        python_commands = {
            str(server.get("command") or "").strip()
            for server in mcp_servers.values()
            if isinstance(server, dict)
            and "python" in str(server.get("command") or "").lower()
        }
        for python_command in sorted(command for command in python_commands if command):
            probe = (
                f"{shlex.quote(python_command)} - <<'PY'\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "PY"
            )
            probe_result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", probe],
                capture_output=True,
                text=True,
            )
            if probe_result.returncode == 0:
                continue
            install = (
                f"{shlex.quote(python_command)} -m pip install --quiet "
                "'mcp==1.16.0'"
            )
            install_result = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", install],
                capture_output=True,
                text=True,
            )
            if install_result.returncode != 0:
                raise RuntimeError(
                    "OpenClaw MCP server dependency is missing and automatic install failed. "
                    f"Command: {python_command} -m pip install 'mcp==1.16.0'\n"
                    f"{install_result.stderr or install_result.stdout}"
                )

    def _configure_mcp_servers(
        self,
        task_id: str,
        runtime_options: dict | None,
        output_dir: Path,
    ) -> None:
        mcp_servers = self._render_openclaw_mcp_servers(runtime_options)
        if not isinstance(mcp_servers, dict) or not mcp_servers:
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        plugin_payload = self._render_openclaw_mcp_plugin_payload(runtime_options)
        plugin_config = plugin_payload["plugin_config"]
        plugin_files = plugin_payload["plugin_files"]
        if is_native_runtime():
            # The config is read by a host process in native mode, so the
            # plugin path must be the task-local host path rather than the
            # container path used by Docker mode.
            plugin_config = dict(plugin_config)
            plugin_config["plugin_path"] = native_path(
                task_id, OPENCLAW_TONGSIM_MCP_PLUGIN_PATH
            )
        (output_dir / "openclaw_mcp_servers.json").write_text(
            json.dumps(plugin_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        debug_plugin_dir = output_dir / "openclaw_tongsim_mcp_bridge_plugin"
        debug_plugin_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in plugin_files.items():
            (debug_plugin_dir / filename).write_text(content, encoding="utf-8")

        config_tmp = None
        plugin_tmp_dir = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
                json.dump(plugin_config, tmp_file, indent=2, ensure_ascii=False)
                config_tmp = tmp_file.name

            plugin_tmp_dir_obj = tempfile.TemporaryDirectory()
            plugin_tmp_dir = plugin_tmp_dir_obj.name
            plugin_tmp_path = Path(plugin_tmp_dir)
            for filename, content in plugin_files.items():
                (plugin_tmp_path / filename).write_text(content, encoding="utf-8")

            prepared = subprocess.run(
                [
                    "docker",
                    "exec",
                    task_id,
                    "/bin/bash",
                    "-lc",
                    f"rm -rf {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} && "
                    f"mkdir -p {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)}",
                ],
                capture_output=True,
                text=True,
            )
            if prepared.returncode != 0:
                raise RuntimeError(f"OpenClaw MCP bridge plugin directory setup failed:\n{prepared.stderr}")

            plugin_copied = subprocess.run(
                ["docker", "cp", f"{plugin_tmp_path}/.", f"{task_id}:{OPENCLAW_TONGSIM_MCP_PLUGIN_PATH}/"],
                capture_output=True,
                text=True,
            )
            if plugin_copied.returncode != 0:
                raise RuntimeError(f"OpenClaw MCP bridge plugin copy failed:\n{plugin_copied.stderr}")

            if is_native_runtime():
                # Native mode runs as the target user, so chowning to the
                # container image's root user is neither needed nor allowed.
                ownership = subprocess.run(
                    [
                        "docker",
                        "exec",
                        task_id,
                        "/bin/bash",
                        "-lc",
                        (
                            f"find {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} -type d -exec chmod 755 {{}} + && "
                            f"find {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} -type f -exec chmod 644 {{}} +"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                )
            else:
                ownership = subprocess.run(
                    [
                        "docker",
                        "exec",
                        task_id,
                        "/bin/bash",
                        "-lc",
                        (
                            f"chown -R root:root {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} && "
                            f"find {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} -type d -exec chmod 755 {{}} + && "
                            f"find {shlex.quote(OPENCLAW_TONGSIM_MCP_PLUGIN_PATH)} -type f -exec chmod 644 {{}} +"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                )
            if ownership.returncode != 0:
                raise RuntimeError(f"OpenClaw MCP bridge plugin ownership setup failed:\n{ownership.stderr}")

            copied = subprocess.run(
                ["docker", "cp", config_tmp, f"{task_id}:/tmp/openclaw_tongsim_mcp_bridge_config.json"],
                capture_output=True,
                text=True,
            )
            if copied.returncode != 0:
                raise RuntimeError(f"OpenClaw MCP config copy failed:\n{copied.stderr}")

            merge_cmd = """python3 - <<'PY'
import json
import pathlib

config_path = pathlib.Path("/root/.openclaw/openclaw.json")
payload_path = pathlib.Path("/tmp/openclaw_tongsim_mcp_bridge_config.json")

config = json.loads(config_path.read_text()) if config_path.exists() else {}
payload = json.loads(payload_path.read_text())
config.pop("mcp", None)
plugins = config.setdefault("plugins", {})
plugins.setdefault("load", {}).setdefault("paths", [])
load_paths = plugins["load"]["paths"]
plugin_path = payload["plugin_path"]
if plugin_path not in load_paths:
    load_paths.append(plugin_path)
entries = plugins.setdefault("entries", {})
entry = entries.setdefault(payload["plugin_id"], {})
entry["enabled"] = True
entry["config"] = payload["config"]
deny = plugins.get("deny")
if isinstance(deny, list):
    plugins["deny"] = [item for item in deny if item != payload["plugin_id"]]
allow = plugins.get("allow")
if not isinstance(allow, list):
    plugins["allow"] = [payload["plugin_id"]]
elif payload["plugin_id"] not in allow:
    allow.append(payload["plugin_id"])
tool_names = payload.get("tool_names") or []
if tool_names:
    tools = config.setdefault("tools", {})
    tools["profile"] = "full"
    tools["allow"] = tool_names
    tools["deny"] = []
    tools.pop("alsoAllow", None)
config_path.parent.mkdir(parents=True, exist_ok=True)
config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
PY"""
            merged = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", merge_cmd],
                capture_output=True,
                text=True,
            )
            if merged.returncode != 0:
                raise RuntimeError(f"OpenClaw MCP config merge failed:\n{merged.stderr or merged.stdout}")
        finally:
            if config_tmp:
                Path(config_tmp).unlink(missing_ok=True)
            if plugin_tmp_dir:
                plugin_tmp_dir_obj.cleanup()

        logger.info("[%s] Configured %d OpenClaw MCP server(s)", task_id, len(mcp_servers))

    @classmethod
    def _render_openclaw_mcp_plugin_payload(cls, runtime_options: dict | None) -> dict[str, object]:
        servers = cls._render_openclaw_mcp_servers(runtime_options)
        if not servers:
            return {}
        tool_names = cls._openclaw_mcp_tool_names(servers)
        plugin_config = {
            "plugin_id": OPENCLAW_TONGSIM_MCP_PLUGIN_ID,
            "plugin_path": OPENCLAW_TONGSIM_MCP_PLUGIN_PATH,
            "tool_names": tool_names,
            "config": {
                "servers": servers,
                "decisionOnly": bool(cls._runtime_openclaw_options(runtime_options).get("decision_only", False)),
            },
        }
        return {
            "plugin_config": plugin_config,
            "plugin_files": cls._render_openclaw_mcp_plugin_files(
                tool_names,
                embedded_config=plugin_config["config"],
            ),
        }

    @classmethod
    def _openclaw_mcp_tool_names(cls, servers: dict[str, dict]) -> list[str]:
        multi_server = len(servers) > 1
        names: list[str] = []
        for server_name, server in servers.items():
            for tool_name in cls._server_tool_names(server):
                names.append(cls._openclaw_mcp_tool_name(server_name, tool_name, multi_server=multi_server))
        return names

    @staticmethod
    def _server_tool_names(server: dict) -> tuple[str, ...]:
        configured = server.get("toolNames")
        if isinstance(configured, list) and configured:
            return tuple(
                str(name)
                for name in configured
                if str(name).strip() and str(name) not in {"finish", "finish_task"}
            )
        return TONGSIM_MCP_TOOL_NAMES

    @staticmethod
    def _openclaw_mcp_tool_name(server_name: str, tool_name: str, *, multi_server: bool) -> str:
        if not multi_server:
            return tool_name
        safe_server = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(server_name))
        return f"mcp__{safe_server}__{tool_name}"

    @classmethod
    def _render_openclaw_mcp_plugin_files(
        cls,
        tool_names: list[str],
        *,
        embedded_config: dict[str, object],
    ) -> dict[str, str]:
        manifest = {
            "id": OPENCLAW_TONGSIM_MCP_PLUGIN_ID,
            "name": "TongSIM MCP Bridge",
            "description": "Expose TongSIM MCP servers as OpenClaw agent tools.",
            "version": "0.1.0",
            "kind": "tool",
            "contracts": {"tools": tool_names},
            "toolMetadata": {
                name: {"replaySafe": False}
                for name in tool_names
            },
            "activation": {"onStartup": True, "onCapabilities": ["tool"]},
            "configSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisionOnly": {"type": "boolean"},
                    "servers": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                },
            },
        }
        package_json = {
            "name": OPENCLAW_TONGSIM_MCP_PLUGIN_ID,
            "version": "0.1.0",
            "type": "module",
            "main": "index.mjs",
            "openclaw": {"extensions": ["./index.mjs"]},
        }
        return {
            "openclaw.plugin.json": json.dumps(manifest, indent=2, ensure_ascii=False),
            "package.json": json.dumps(package_json, indent=2, ensure_ascii=False),
            "index.mjs": cls._render_openclaw_mcp_plugin_module(embedded_config=embedded_config),
        }

    @classmethod
    def _render_openclaw_mcp_plugin_module(cls, embedded_config: dict[str, object] | None = None) -> str:
        tool_names_json = json.dumps(list(TONGSIM_MCP_TOOL_NAMES), indent=2)
        embedded_config_json = json.dumps(embedded_config or {"servers": {}, "decisionOnly": False}, indent=2)
        return f"""import {{ spawn }} from "node:child_process";

const DEFAULT_MCP_TOOL_NAMES = {tool_names_json};
const EMBEDDED_CONFIG = {embedded_config_json};

function pluginConfig(api) {{
  const candidates = [
    api?.pluginConfig,
    api?.config,
    api?.entry?.config,
    api?.manifest?.config,
    EMBEDDED_CONFIG,
  ];
  for (const candidate of candidates) {{
    if (candidate && typeof candidate === "object" && Object.keys(candidate.servers || {{}}).length > 0) {{
      return candidate;
    }}
  }}
  return EMBEDDED_CONFIG;
}}

function sanitizeServerName(value) {{
  return String(value).replace(/[^A-Za-z0-9_]/g, "_");
}}

function openclawToolName(serverName, toolName, multiServer) {{
  return multiServer ? `mcp__${{sanitizeServerName(serverName)}}__${{toolName}}` : toolName;
}}

function serverToolNames(server) {{
  const configured = Array.isArray(server?.toolNames) ? server.toolNames : [];
  return configured.length > 0 ? configured.map(String) : DEFAULT_MCP_TOOL_NAMES;
}}

function chooseActionSchema(decisionOnly) {{
  const properties = {{
    template_id: {{
      type: "string",
      description: "Exact exposed action template name, for example look_at_{{object}}.",
    }},
    role_bindings: {{
      type: "object",
      description: "Map each template placeholder to one provided object_id.",
      additionalProperties: {{ type: "string" }},
    }},
    action_level: {{
      type: "string",
      default: "atomic",
      description: "Action level. Use atomic unless explicitly instructed otherwise.",
    }},
  }};
  const required = ["template_id", "role_bindings"];
  if (!decisionOnly) {{
    Object.assign(properties, {{
      action_type: {{ type: "string", default: "" }},
      brief_reason: {{ type: "string", default: "" }},
      perceived_state_predicates: {{
        type: "array",
        items: {{ type: "string" }},
        default: [],
      }},
      uncertain_predicates: {{
        type: "array",
        items: {{ type: "string" }},
        default: [],
      }},
      visual_evidence: {{
        type: "array",
        items: {{ type: "object", additionalProperties: true }},
        default: [],
      }},
    }});
  }}
  return {{ type: "object", properties, required, additionalProperties: !decisionOnly }};
}}

function toolSchema(toolName, decisionOnly) {{
  if (toolName === "choose_action") {{
    return chooseActionSchema(decisionOnly);
  }}
  if (toolName === "select_option") {{
    return {{
      type: "object",
      properties: {{
        option_id: {{ type: "string", description: "Exactly one currently listed option letter." }},
        brief_reason: {{ type: "string", default: "" }},
      }},
      required: ["option_id"],
      additionalProperties: false,
    }};
  }}
  return {{ type: "object", properties: {{}}, additionalProperties: false }};
}}

function toolDescription(toolName) {{
  switch (toolName) {{
    case "observe_current_state":
      return "Get the current TongSIM observation including task, candidates, feedback, done status, and the current image.";
    case "choose_action":
      return "Submit one TongSIM action decision using an exposed template_id and exact role_bindings.";
    case "check_task_status":
      return "Check whether the current TongSIM episode is done and whether the goal is reached.";
    case "observe":
      return "Get the current household task, current choices, feedback, and observation image.";
    case "select_option":
      return "Select exactly one listed option letter for the current household task.";
    case "status":
      return "Check the current safe-choice episode status.";
    default:
      return `TongSIM MCP tool ${{toolName}}`;
  }}
}}

function encodeMessage(message) {{
  return Buffer.from(`${{JSON.stringify(message)}}\\n`, "utf8");
}}

function decodeMessages(buffer) {{
  const messages = [];
  let rest = buffer;
  const delimiter = Buffer.from("\\n", "ascii");
  while (true) {{
    const headerEnd = rest.indexOf(delimiter);
    if (headerEnd < 0) break;
    const line = rest.subarray(0, headerEnd).toString("utf8").trim();
    rest = rest.subarray(headerEnd + delimiter.length);
    if (!line) {{
      continue;
    }}
    messages.push(JSON.parse(line));
  }}
  return [messages, rest];
}}

async function callMcpTool(server, toolName, args) {{
  return await new Promise((resolve, reject) => {{
    const command = server.command;
    if (!command) {{
      reject(new Error("MCP server command is missing"));
      return;
    }}
    const child = spawn(command, Array.isArray(server.args) ? server.args : [], {{
      env: {{ ...process.env, ...(server.env || {{}}) }},
      cwd: server.cwd || process.cwd(),
      stdio: ["pipe", "pipe", "pipe"],
    }});
    let buffer = Buffer.alloc(0);
    let stderr = "";
    const pending = new Map();

    function stderrTail() {{
      return stderr ? stderr.slice(-4000) : "";
    }}

    const timer = setTimeout(() => {{
      child.kill("SIGTERM");
      const waiting = Array.from(pending.values()).map(item => item.method).join(", ") || "none";
      reject(new Error(`MCP tool timed out calling ${{toolName}}; waiting=${{waiting}}; stderr=${{stderrTail()}}`));
    }}, Number(server.requestTimeoutMs || 120000));

    function send(id, method, params) {{
      pending.set(id, {{ method }});
      child.stdin.write(encodeMessage({{ jsonrpc: "2.0", id, method, params }}));
    }}

    function sendNotification(method, params = {{}}) {{
      child.stdin.write(encodeMessage({{ jsonrpc: "2.0", method, params }}));
    }}

    function completeWithError(error) {{
      clearTimeout(timer);
      child.kill("SIGTERM");
      reject(error);
    }}

    child.stderr.on("data", chunk => {{
      stderr += chunk.toString();
    }});
    child.stdout.on("data", chunk => {{
      try {{
        buffer = Buffer.concat([buffer, chunk]);
        const decoded = decodeMessages(buffer);
        const messages = decoded[0];
        buffer = decoded[1];
        for (const message of messages) {{
          if (!message || message.id === undefined) continue;
          const request = pending.get(message.id);
          pending.delete(message.id);
          if (message.error) {{
            completeWithError(new Error(JSON.stringify(message.error)));
            return;
          }}
          if (request?.method === "initialize") {{
            sendNotification("notifications/initialized");
            send(2, "tools/call", {{ name: toolName, arguments: args || {{}} }});
          }} else if (request?.method === "tools/call") {{
            clearTimeout(timer);
            child.kill("SIGTERM");
            resolve(message.result || {{}});
          }}
        }}
      }} catch (error) {{
        completeWithError(error);
      }}
    }});
    child.on("error", error => completeWithError(error));
    child.on("exit", code => {{
      if (pending.size > 0) {{
        clearTimeout(timer);
        reject(new Error(`MCP process exited before response (code=${{code}}): ${{stderr}}`));
      }}
    }});

    send(1, "initialize", {{
      protocolVersion: "2024-11-05",
      capabilities: {{}},
      clientInfo: {{ name: "openclaw-tongsim-mcp-bridge", version: "0.1.0" }},
    }});
  }});
}}

function extractImageValue(value) {{
  if (!value) return null;
  if (typeof value === "string") return value;
  if (typeof value === "object") {{
    return value.url || value.image || value.data || value.image_url || null;
  }}
  return null;
}}

function convertMcpImageBlock(block, piImageContent) {{
  const mimeType = String(block?.mimeType || block?.mime_type || block?.media_type || "image/jpeg");
  let imageValue = extractImageValue(block?.image || block?.image_url || block?.url || block?.data);
  if (!imageValue && block?.source && typeof block.source === "object") {{
    imageValue = extractImageValue(block.source.data || block.source.url || block.source);
  }}
  if (!imageValue) {{
    return null;
  }}
  imageValue = String(imageValue);
  if (piImageContent) {{
    const marker = ";base64,";
    const markerIndex = imageValue.indexOf(marker);
    return {{
      type: "image",
      data: markerIndex >= 0 ? imageValue.slice(markerIndex + marker.length) : imageValue,
      mimeType,
    }};
  }}
  const image = imageValue.startsWith("data:") ? imageValue : `data:${{mimeType}};base64,${{imageValue}}`;
  return {{
    type: "image_url",
    image_url: {{
      url: image,
    }},
  }};
}}

function convertMcpResult(result, server) {{
  const piImageContent = ["1", "true", "yes", "on"].includes(
    String(server?.env?.TONGSIM_MCP_PI_IMAGE_CONTENT || "").toLowerCase(),
  );
  if (Array.isArray(result?.content)) {{
    const textParts = [];
    const imageParts = [];
    for (const block of result.content) {{
      if (typeof block === "string") {{
        textParts.push(normalizeOpenClawToolText(block));
        continue;
      }}
      if (block?.type === "text" && typeof block.text === "string") {{
        textParts.push(normalizeOpenClawToolText(block.text));
        continue;
      }}
      if (block?.type === "image") {{
        const imageBlock = convertMcpImageBlock(block, piImageContent);
        if (imageBlock) {{
          imageParts.push(imageBlock);
          continue;
        }}
        textParts.push("TongSIM observation image was present, but the image block could not be converted.");
        continue;
      }}
      textParts.push(JSON.stringify(block || {{}}, null, 2));
    }}
    const content = [];
    const text = textParts.join("\\n\\n");
    if (text) {{
      content.push({{
        type: "text",
        text,
      }});
    }}
    content.push(...imageParts);
    if (content.length === 0) {{
      content.push({{
        type: "text",
        text: "",
      }});
    }}
    return {{
      content,
    }};
  }}
  return {{
    content: [
      {{
        type: "text",
        text: JSON.stringify(result || {{}}, null, 2),
      }},
    ],
  }};
}}

function normalizeOpenClawToolText(text) {{
  return String(text || "");
}}

function makeTool(serverName, server, toolName, multiServer, decisionOnly) {{
  const name = openclawToolName(serverName, toolName, multiServer);
  return {{
    name,
    label: name,
    description: toolDescription(toolName),
    parameters: toolSchema(toolName, decisionOnly),
    async execute(first, second) {{
      const params = second === undefined ? first : second;
      const result = await callMcpTool(server, toolName, params || {{}});
      return convertMcpResult(result, server);
    }},
  }};
}}

export default {{
  id: "{OPENCLAW_TONGSIM_MCP_PLUGIN_ID}",
  name: "TongSIM MCP Bridge",
  register(api) {{
    const config = pluginConfig(api);
    const servers = config.servers || {{}};
    const decisionOnly = Boolean(config.decisionOnly);
    const entries = Object.entries(servers);
    const multiServer = entries.length > 1;
    const registeredNames = [];
    for (const [serverName, server] of entries) {{
      for (const toolName of serverToolNames(server)) {{
        const name = openclawToolName(serverName, toolName, multiServer);
        registeredNames.push(name);
        api.registerTool(() => makeTool(serverName, server, toolName, multiServer, decisionOnly));
      }}
    }}
    api?.logger?.info?.(`TongSIM MCP Bridge registered tools: ${{registeredNames.join(", ")}}`);
  }},
}};
"""

    @classmethod
    def _render_openclaw_mcp_servers(cls, runtime_options: dict | None) -> dict[str, dict]:
        rendered: dict[str, dict] = {}
        for server_name, server in cls._runtime_mcp_servers(runtime_options).items():
            if not isinstance(server, dict):
                continue
            normalized = cls._render_openclaw_mcp_server(server)
            if normalized:
                rendered[str(server_name)] = normalized
        return rendered

    @classmethod
    def _render_openclaw_mcp_server(cls, server: dict) -> dict:
        config: dict = {}
        for key in (
            "command",
            "url",
            "transport",
            "cwd",
            "auth",
            "headers",
            "enabled",
            "supportsParallelToolCalls",
        ):
            if key in server and server.get(key) is not None:
                config[key] = server[key]
        args = server.get("args")
        if isinstance(args, list):
            config["args"] = [str(value) for value in args]
        env = server.get("env")
        if isinstance(env, dict) and env:
            config["env"] = {str(key): str(value) for key, value in env.items()}
        enabled_tools = server.get("enabled_tools")
        if isinstance(enabled_tools, list) and enabled_tools:
            config["toolNames"] = [str(value) for value in enabled_tools]
        tool_filter = server.get("toolFilter")
        if isinstance(tool_filter, dict):
            config["toolFilter"] = tool_filter

        request_timeout = server.get("requestTimeoutMs")
        if request_timeout is None:
            request_timeout = server.get("tool_timeout_sec", server.get("timeout"))
            if request_timeout is not None:
                request_timeout = cls._seconds_to_milliseconds(request_timeout)
        if request_timeout is not None:
            config["requestTimeoutMs"] = int(request_timeout)

        connection_timeout = server.get("connectionTimeoutMs")
        if connection_timeout is None:
            connection_timeout = server.get("startup_timeout_sec", server.get("connect_timeout"))
            if connection_timeout is not None:
                connection_timeout = cls._seconds_to_milliseconds(connection_timeout)
        if connection_timeout is not None:
            config["connectionTimeoutMs"] = int(connection_timeout)

        if not config.get("command") and not config.get("url"):
            return {}
        return config

    @staticmethod
    def _seconds_to_milliseconds(value: object) -> int:
        return int(float(value) * 1000)

    @staticmethod
    def _runtime_mcp_servers(runtime_options: dict | None) -> dict:
        if not isinstance(runtime_options, dict):
            return {}
        openclaw_options = runtime_options.get("openclaw", {})
        if isinstance(openclaw_options, dict) and isinstance(openclaw_options.get("mcp_servers"), dict):
            return openclaw_options["mcp_servers"]
        hermes_options = runtime_options.get("hermes", {})
        if isinstance(hermes_options, dict) and isinstance(hermes_options.get("mcp_servers"), dict):
            return hermes_options["mcp_servers"]
        codex_options = runtime_options.get("codex", {})
        if isinstance(codex_options, dict) and isinstance(codex_options.get("mcp_servers"), dict):
            return codex_options["mcp_servers"]
        if isinstance(runtime_options.get("mcp_servers"), dict):
            return runtime_options["mcp_servers"]
        return {}

    @staticmethod
    def _runtime_openclaw_options(runtime_options: dict | None) -> dict:
        if not isinstance(runtime_options, dict):
            return {}
        openclaw_options = runtime_options.get("openclaw", {})
        return openclaw_options if isinstance(openclaw_options, dict) else {}

    @classmethod
    def _resolve_runtime_model_ref(cls, model: str, models_config: dict | None) -> str:
        raw_model = str(model or "").strip()
        if not raw_model:
            return raw_model
        providers = models_config.get("providers", {}) if isinstance(models_config, dict) else {}
        if isinstance(providers, dict):
            if "/" in raw_model:
                provider_name, provider_model = raw_model.split("/", 1)
                if provider_name in providers:
                    return f"{provider_name}/{provider_model}"
            for provider_name, provider in providers.items():
                if not isinstance(provider, dict):
                    continue
                for entry in provider.get("models", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    entry_id = str(entry.get("id") or "").strip()
                    if not entry_id:
                        continue
                    if entry_id == raw_model:
                        return f"{provider_name}/{entry_id}"
                    if entry_id == raw_model.split("/", 1)[-1]:
                        return f"{provider_name}/{entry_id}"
                    if raw_model == f"{provider_name}/{entry_id}":
                        return raw_model
            if len(providers) == 1 and "/" not in raw_model:
                provider_name = next(iter(providers))
                return f"{provider_name}/{raw_model}"
        return raw_model

    def _set_model(self, task_id: str, model: str) -> None:
        safe_model = shlex.quote(model)
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"openclaw models set {safe_model}"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Model setup failed:\n{r.stderr}")
        logger.info("[%s] Model set: %s", task_id, model)

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
            for entry in provider.get("models", []):
                if isinstance(entry, dict) and entry.get("id") == model:
                    return provider.get("apiKey", ""), provider.get("baseUrl", "")
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return first.get("apiKey", ""), first.get("baseUrl", "")
        return "", ""

    def _inject_openrouter_key(self, task_id: str, runtime_api_key: str) -> None:
        if not runtime_api_key:
            return

        auth_profile_path = "/root/.openclaw/agents/main/agent/auth-profiles.json"
        inject_cmd = f"""python3 - <<'PY'
import json
import pathlib

p = pathlib.Path("{auth_profile_path}")
d = json.loads(p.read_text()) if p.exists() else {{"version": 1, "profiles": {{}}}}
d.setdefault("profiles", {{}})["openrouter:default"] = {{
    "type": "api_key",
    "provider": "openrouter",
    "key": {json.dumps(runtime_api_key)}
}}
p.write_text(json.dumps(d, indent=2))
PY"""
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", inject_cmd],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] Injected OPENROUTER_API_KEY into auth-profiles.json", task_id)

    def _set_image_model(self, task_id: str, model: str) -> None:
        safe_model = shlex.quote(model)
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", f"openclaw config set agents.defaults.imageModel.primary {safe_model}"],
            capture_output=True,
            text=True,
        )
        logger.info("[%s] imageModel set: %s", task_id, model)
