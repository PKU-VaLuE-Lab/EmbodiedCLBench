from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

TMP_WORKSPACE = os.environ.get("TMP_WORKSPACE", "/tmp_workspace")
OPENCLAW_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"
CODEX_PROMPT_PATH = "/tmp/codex_prompt.txt"
CODEX_LAST_MESSAGE_PATH = "/tmp/codex_last_message.txt"
CONTAINER_CODEX_HOME = "/root/.codex"
DEFAULT_CODEX_NPM_PACKAGE = os.environ.get("CODEX_NPM_PACKAGE", "@openai/codex")
DEFAULT_CODEX_NPM_VERSION = os.environ.get("CODEX_NPM_VERSION", "0.117.0")
CODEX_BOOTSTRAP_RETRIES = int(os.environ.get("CODEX_BOOTSTRAP_RETRIES", "2"))
CODEX_BOOTSTRAP_RETRY_BASE_DELAY = float(os.environ.get("CODEX_BOOTSTRAP_RETRY_BASE_DELAY", "3"))
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _copy_text_to_container(task_id: str, container_path: str, text: str) -> None:
    container_target = str(PurePosixPath(container_path))
    container_dir = str(PurePosixPath(container_target).parent)
    mkdir_result = subprocess.run(
        ["docker", "exec", "-u", "0", task_id, "mkdir", "-p", container_dir],
        capture_output=True,
        text=True,
    )
    if mkdir_result.returncode != 0:
        raise RuntimeError(
            f"Failed to create container directory {container_dir}:\n"
            f"{mkdir_result.stderr or mkdir_result.stdout}"
        )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp_file:
        tmp_file.write(text)
        tmp_host_path = tmp_file.name

    try:
        copy_result = subprocess.run(
            ["docker", "cp", tmp_host_path, f"{task_id}:{container_target}"],
            capture_output=True,
            text=True,
        )
        if copy_result.returncode != 0:
            debug_result = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "0",
                    task_id,
                    "/bin/sh",
                    "-lc",
                    (
                        f"ls -ld {shlex.quote(container_dir)} 2>&1 || true; "
                        "id -un 2>/dev/null || true; "
                        "echo HOME=${HOME:-}"
                    ),
                ],
                capture_output=True,
                text=True,
            )
            raise RuntimeError(
                "Failed to copy file into container:\n"
                f"{copy_result.stderr}"
                "Container path debug:\n"
                f"{debug_result.stdout or debug_result.stderr}"
            )
    finally:
        Path(tmp_host_path).unlink(missing_ok=True)


def read_text_from_container(task_id: str, container_path: str) -> str:
    result = subprocess.run(
        ["docker", "exec", "-u", "0", task_id, "cat", container_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def discover_codex_auth_sources(host_codex_home: str | Path | None = None) -> list[tuple[Path, str]]:
    _ = host_codex_home
    return []


def setup_codex_auth(task_id: str, host_codex_home: str | Path | None = None) -> list[str]:
    _ = host_codex_home
    if os.environ.get("OPENROUTER_API_KEY"):
        logger.info("[%s] Codex auth file copy disabled; relying on OpenRouter env vars for Codex", task_id)
    else:
        logger.warning("[%s] Codex auth file copy disabled and OPENROUTER_API_KEY is empty", task_id)
    return []


def normalize_codex_model(model: str) -> str:
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def build_codex_config_toml(base_url: str, model: str) -> str:
    return f"""\
model_provider = "openrouter"
model = "{normalize_codex_model(model)}"

[model_providers.openrouter]
name = "openrouter"
base_url = "{base_url}"
env_key = "OPENROUTER_API_KEY"

[tools]
apply_patch = true
bash = true
file_read = true
file_write = true
web_search = true
"""


def setup_codex_config(task_id: str, model: str) -> None:
    base_url = os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_OPENROUTER_BASE_URL
    config_text = build_codex_config_toml(
        base_url=base_url,
        model=model,
    )
    _copy_text_to_container(task_id, f"{CONTAINER_CODEX_HOME}/config.toml", config_text)


def build_codex_bootstrap_command(
    package: str = DEFAULT_CODEX_NPM_PACKAGE,
    version: str | None = DEFAULT_CODEX_NPM_VERSION,
) -> str:
    package_spec = package if not version else f"{package}@{version}"
    return (
        "if ! command -v codex >/dev/null 2>&1; then "
        f"npm install -g {shlex.quote(package_spec)}; "
        "fi"
    )


def looks_like_transient_bootstrap_failure(output: str) -> bool:
    lowered = output.lower()
    transient_markers = [
        "econnreset",
        "network connectivity",
        "proxy",
        "network is unreachable",
        "connection reset",
        "connection broken",
        "read timed out",
        "temporary failure in name resolution",
        "sslerror",
    ]
    return any(marker in lowered for marker in transient_markers)


def ensure_codex_cli(task_id: str) -> None:
    bootstrap_cmd = build_codex_bootstrap_command()
    last_error_output = ""
    for attempt in range(CODEX_BOOTSTRAP_RETRIES + 1):
        result = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", bootstrap_cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("[%s] Codex CLI is available inside container", task_id)
            return

        last_error_output = result.stderr or result.stdout
        if not looks_like_transient_bootstrap_failure(last_error_output) or attempt >= CODEX_BOOTSTRAP_RETRIES:
            raise RuntimeError(f"Failed to bootstrap Codex CLI in container:\n{last_error_output}")

        delay_seconds = CODEX_BOOTSTRAP_RETRY_BASE_DELAY * (2 ** attempt)
        logger.warning(
            "[%s] Codex bootstrap failed with a transient network error; retrying in %.1fs (%d/%d)",
            task_id,
            delay_seconds,
            attempt + 1,
            CODEX_BOOTSTRAP_RETRIES + 1,
        )
        time.sleep(delay_seconds)


def load_skill_documents(
    skills: str,
    skills_path: str,
    container_skill_root: str = "/root/skills",
) -> list[dict[str, str]]:
    loaded_skills: list[dict[str, str]] = []
    for line in skills.splitlines():
        skill_name = line.strip()
        if not skill_name:
            continue

        skill_rel = skill_name.replace("\\", "/").strip("/")
        skill_leaf = PurePosixPath(skill_rel).name
        if not skill_leaf:
            logger.warning("Invalid skill path for Codex prompt injection: %s", skill_name)
            continue

        skill_file = Path(skills_path) / skill_rel / "SKILL.md"
        if not skill_file.is_file():
            logger.warning("Skill file not found for Codex prompt injection: %s", skill_file)
            continue

        content = skill_file.read_text(encoding="utf-8")
        content = content.replace("{baseDir}", f"{container_skill_root}/{skill_leaf}")
        loaded_skills.append({"name": skill_name, "content": content})

    return loaded_skills


def build_codex_prompt(base_prompt: str, skill_docs: list[dict[str, str]]) -> str:
    if not skill_docs:
        return base_prompt

    sections = [
        "You may use the following local skill references. These are instructions and workflows available in this task environment.",
    ]
    for skill in skill_docs:
        sections.append(f"## Skill: {skill['name']}\n\n{skill['content'].strip()}")
    sections.append("## Task\n\n" + base_prompt.strip())
    return "\n\n".join(sections).strip() + "\n"


def get_codex_provider_env() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    for key in ["OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"]:
        value = os.environ.get(key)
        if value:
            env_vars[key] = value
    return env_vars


def build_codex_exec_command(
    model: str,
    prompt_path: str = CODEX_PROMPT_PATH,
    env_vars: dict[str, str] | None = None,
) -> str:
    normalized_model = normalize_codex_model(model)
    model_arg = f"--model {shlex.quote(normalized_model)} " if normalized_model else ""
    env_prefix = ""
    for key, value in (env_vars or {}).items():
        env_prefix += f"export {key}={shlex.quote(value)} && "
    return (
        f"{env_prefix}cat {shlex.quote(prompt_path)} | "
        f"codex exec --json --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox "
        f"--cd {shlex.quote(TMP_WORKSPACE)} "
        f"{model_arg}"
        f"--output-last-message {shlex.quote(CODEX_LAST_MESSAGE_PATH)} -"
    )


def parse_codex_json_events(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    events: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type"):
            events.append(event)
    return events


def _message_entry(content: str | list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if content is None:
        content = []
    elif isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return {"type": "message", "message": {"role": "assistant", "content": content}}


def _build_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = int(usage.get("input_tokens", usage.get("inputTokens", 0)) or 0)
    cache_read_tokens = int(usage.get("cached_input_tokens", usage.get("cacheReadTokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("outputTokens", 0)) or 0)
    total_tokens = int(
        usage.get("total_tokens", usage.get("totalTokens", input_tokens + output_tokens + cache_read_tokens)) or 0
    )

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": cache_read_tokens,
        "cacheWrite": 0,
        "totalTokens": total_tokens,
        "cost": {"total": 0.0},
    }


def _tool_call_entry(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _message_entry([{"type": "toolCall", "name": name, "arguments": payload}])


def _message_contains_text(message: dict[str, Any]) -> bool:
    content = message.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "text" for block in content)


def codex_events_to_openclaw_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compat_messages: list[dict[str, Any]] = []
    pending_usage: dict[str, Any] | None = None

    for event in events:
        event_type = event.get("type")

        if event_type == "item.completed":
            item = event.get("item") or {}
            item_type = item.get("type") or item.get("item_type")

            if item_type in {"agent_message", "assistant_message"}:
                text = item.get("text")
                if not text:
                    content = item.get("content")
                    if isinstance(content, str):
                        text = content
                    elif isinstance(content, list):
                        text_chunks = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_chunks.append(block.get("text", ""))
                        text = "\n".join(chunk for chunk in text_chunks if chunk)
                compat_messages.append(_message_entry(text or ""))
            elif item_type == "command_execution":
                compat_messages.append(
                    _tool_call_entry(
                        "exec_command",
                        {
                            "cmd": item.get("command", ""),
                            "parsed_cmd": item.get("parsed_cmd"),
                            "status": item.get("status"),
                            "exit_code": item.get("exit_code"),
                            "output": item.get("aggregated_output", ""),
                        },
                    )
                )
            elif item_type in {"mcp_tool_call", "tool_call"}:
                compat_messages.append(
                    _tool_call_entry(
                        str(item.get("tool_name") or item.get("name") or item_type),
                        {
                            "server": item.get("server"),
                            "arguments": item.get("arguments") or item.get("input") or {},
                        },
                    )
                )
            elif item_type == "web_search":
                compat_messages.append(
                    _tool_call_entry(
                        "web_search",
                        {
                            "query": item.get("query"),
                            "result_count": item.get("result_count"),
                        },
                    )
                )
            elif item_type == "file_change":
                compat_messages.append(
                    _tool_call_entry(
                        "write_file",
                        {
                            "path": item.get("path"),
                            "change_type": item.get("change_type"),
                            "description": item.get("description"),
                        },
                    )
                )
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, str) and summary.strip():
                    compat_messages.append(_message_entry(summary.strip()))
            elif item_type == "error":
                message = str(item.get("message") or item.get("text") or "Codex item error")
                compat_messages.append(_message_entry(message))

        elif event_type in {"turn.completed", "task_complete"}:
            pending_usage = _build_usage(event.get("usage"))
        elif event_type == "error":
            compat_messages.append(_message_entry(str(event.get("message") or "Codex execution error")))

    if pending_usage is not None:
        for message in reversed(compat_messages):
            if (
                message.get("type") == "message"
                and message.get("message", {}).get("role") == "assistant"
                and _message_contains_text(message)
            ):
                message["message"]["usage"] = pending_usage
                break
        else:
            synthetic = _message_entry("")
            synthetic["message"]["usage"] = pending_usage
            compat_messages.append(synthetic)

    return compat_messages


def write_openclaw_compat_transcript(
    task_id: str,
    events: list[dict[str, Any]],
    container_path: str = OPENCLAW_TRANSCRIPT_PATH,
) -> int:
    compat_messages = codex_events_to_openclaw_messages(events)
    transcript_text = ""
    if compat_messages:
        transcript_text = "\n".join(json.dumps(message, ensure_ascii=False) for message in compat_messages) + "\n"
    _copy_text_to_container(task_id, container_path, transcript_text)
    return len(compat_messages)


def prepare_codex_prompt(task_id: str, prompt: str, container_path: str = CODEX_PROMPT_PATH) -> str:
    _copy_text_to_container(task_id, container_path, prompt)
    return container_path


def run_codex_process(
    task_id: str,
    model: str,
    prompt: str,
    timeout_seconds: int,
    output_dir: Path,
    run_background_fn,
) -> tuple[subprocess.Popen | None, subprocess.Popen | None, float]:
    gateway_proc = None
    agent_proc = None
    elapsed_time = float(timeout_seconds)

    prepare_codex_prompt(task_id, prompt)
    codex_cmd = build_codex_exec_command(model, env_vars=get_codex_provider_env())
    start_time = time.perf_counter()
    agent_proc = run_background_fn(
        task_id,
        bash_cmd=codex_cmd,
        log_path=output_dir / "agent.log",
    )

    logger.info("[%s] Waiting for Codex to finish...", task_id)
    try:
        agent_proc.wait(timeout=timeout_seconds)
        elapsed_time = time.perf_counter() - start_time
        logger.info("[%s] Codex finished successfully, elapsed: %.2f seconds", task_id, elapsed_time)
    except subprocess.TimeoutExpired:
        logger.info("[%s] Codex timed out...", task_id)
        elapsed_time = timeout_seconds
        agent_proc.kill()
        agent_proc.wait()

    logger.info("[%s] Codex exit code: %s", task_id, agent_proc.returncode)
    return gateway_proc, agent_proc, elapsed_time
