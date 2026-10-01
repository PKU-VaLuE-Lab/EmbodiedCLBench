from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tongbench_eval.agents.base import AgentTaskSpec, BaseAgent
from tongbench_eval.cli.framework.protocol import (
    build_protocol_manifest,
    claudecode_allowed_tools,
    codex_mcp_servers,
    mcp_image_env,
    protocol_tool_names,
)
from tongbench_eval.utils.docker_utils import close_proc_log, remove_container
from tongbench_eval.graph_env.framework_prompt import build_framework_agent_prompt
from tongbench_eval.graph_env.framework_runtime import TongSimFrameworkRuntime
from tongbench_eval.graph_env.framework_diagnostics import write_sidecar_decision_diagnostics
from tongbench_eval.graph_env.loader import load_tongsim_task
from tongbench_eval.utils.native_runtime import (
    activate_runtime,
    is_native_runtime,
    native_bundle_root,
    native_path,
)
from tongbench_eval.utils.python_utils import current_python


UPDATED_ATOMIC_TEMPLATE_NAME = "atomic_templates_updated_env_v3.json"
UPDATED_SUBTASK_TEMPLATE_NAME = "subtask_templates_compressed_updated_env_v3.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _task_level(task_id: str) -> str:
    match = re.search(r"task_(L\d+)_", str(task_id))
    return match.group(1) if match else "L?"


def _allow_container_bridge_writes(bridge_dir: Path) -> None:
    """Let framework containers write MCP request/response files on bind mounts."""
    for path in (
        bridge_dir,
        bridge_dir / "requests",
        bridge_dir / "responses",
        bridge_dir / "current_observation",
    ):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o777)


def _nativeize_session_configs(public_workspace: Path, bridge_dir: str | Path) -> None:
    """Rewrite container-only bridge paths for MCP processes started natively."""
    # The native MCP server runs as a host process.  It must use the actual
    # host bridge directory, not the container path or the native task root.
    bridge_path = str(Path(bridge_dir).resolve())
    for config_path in (
        public_workspace / ".tongsim_session.json",
        public_workspace / "exec" / ".tongsim_session.json",
    ):
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payload["bridge_mount"] = bridge_path
            config_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _load_models_config(path: Path | None) -> dict | None:
    if path is None:
        return None
    raw_config = path.read_text(encoding="utf-8")
    expanded = re.sub(
        r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}",
        lambda match: os.environ.get(match.group(1), ""),
        raw_config,
    )
    parsed = json.loads(expanded)
    if not isinstance(parsed, dict):
        raise ValueError(f"Models config must be a JSON object: {path}")
    return parsed


def _apply_provider_overrides(
    models_config: dict | None,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict | None:
    if not api_key and not base_url:
        return models_config
    config = dict(models_config or {})
    providers = dict(config.get("providers", {}) or {})
    target_name = ""
    target_provider: dict | None = None
    for provider_name, provider in providers.items():
        if not isinstance(provider, dict):
            continue
        for item in provider.get("models", []) or []:
            if isinstance(item, dict) and item.get("id") == model:
                target_name = str(provider_name)
                target_provider = dict(provider)
                break
        if target_provider is not None:
            break
    if target_provider is None:
        target_name = "dashscope" if "dashscope.aliyuncs.com" in str(base_url or "").lower() else "cli-override"
        target_provider = {"models": [{"id": model, "name": model}]}
    if api_key:
        target_provider["apiKey"] = api_key
    if base_url:
        target_provider["baseUrl"] = base_url
    providers[target_name] = target_provider
    config["providers"] = providers
    return config


def _template_search_dirs(*extra_roots: Path | None) -> list[Path]:
    seeds: list[Path] = [ROOT, ROOT.parent]
    seeds.extend(root for root in extra_roots if root is not None)
    candidates: list[Path] = []
    for seed in seeds:
        candidates.extend(
            [
                seed / "outputs",
                seed / "TongBench" / "outputs",
                seed / "Tongbench" / "outputs",
            ]
        )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate.resolve()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _is_legacy_fixture_template_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "tests" in parts and "fixtures" in parts and path.name.lower() in {
        "atomic_templates.json",
        "subtask_templates.json",
    }


def _resolve_default_template_path(
    raw_path: str | None,
    env_var: str,
    filename: str,
    *extra_roots: Path | None,
) -> Path | None:
    explicit_path = Path(raw_path).resolve() if raw_path else None
    if explicit_path is not None and not _is_legacy_fixture_template_path(explicit_path):
        return explicit_path
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).resolve()
    for directory in _template_search_dirs(*extra_roots):
        candidate = directory / filename
        if candidate.exists():
            return candidate.resolve()
    return explicit_path


def _make_backend(
    name: str,
    *,
    gateway_port: int | None = None,
    openclaw_image_model: str | None = None,
) -> BaseAgent:
    if name == "openclaw":
        from tongbench_eval.agents.openclaw import OpenClawAgent

        resolved_gateway_port = int(gateway_port or os.environ.get("GATEWAY_PORT", "18789"))
        return OpenClawAgent(gateway_port=resolved_gateway_port, image_model=openclaw_image_model)
    if name == "codex":
        from tongbench_eval.agents.codex import CodexAgent

        return CodexAgent()
    if name == "claudecode":
        from tongbench_eval.agents.claudecode import ClaudeCodeAgent

        return ClaudeCodeAgent()
    if name == "hermesagent":
        from tongbench_eval.agents.hermesagent import HermesAgentAgent

        return HermesAgentAgent()
    raise ValueError(f"Unsupported TongSIM framework backend: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TongSIM through a framework-controlled hidden session bridge.")
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["openclaw", "codex", "claudecode", "hermesagent"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--runtime",
        choices=["docker", "native"],
        default=os.environ.get("TONGBENCH_RUNTIME", "docker"),
        help="Execution runtime. docker uses the Docker daemon; native uses the bundled user-space runtime.",
    )
    parser.add_argument(
        "--description-field",
        default="description",
        help="Task JSON description field exposed to the model. Defaults to the legacy description field.",
    )
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=90, help="Hermes max tool/agent iterations in semantic_tool mode.")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--thinking")
    parser.add_argument("--responses-non-stream", action="store_true")
    parser.add_argument("--context-mode", choices=["full", "task_compact"], default="full")
    parser.add_argument("--models-config")
    parser.add_argument("--api-key", help="Override the selected model provider API key without using an environment variable.")
    parser.add_argument("--base-url", help="Override the selected model provider OpenAI-compatible base URL.")
    parser.add_argument("--gateway-port", type=int, help="OpenClaw gateway port. Defaults to GATEWAY_PORT or 18789.")
    parser.add_argument("--openclaw-image-model", help="Optional OpenClaw image model override. Defaults to --model.")
    parser.add_argument(
        "--mcp-python-command",
        help=(
            "Python executable used inside the agent container for TongSIM MCP servers. "
            "Defaults to /opt/hermes/.venv/bin/python3 for hermesagent and python3 for openclaw/codex/claudecode."
        ),
    )
    parser.add_argument("--action-interface", choices=["full_action", "factorized", "global_factorized", "library_factorized"], default="library_factorized")
    parser.add_argument("--action-library-mode", choices=["atomic_only", "subtask_only", "atomic_plus_subtask"], default="atomic_only")
    parser.add_argument("--public-interface-mode", choices=["structured", "natural_language"], default="structured")
    parser.add_argument("--framework-interaction-mode", choices=["command_line", "semantic_tool"], default="command_line")
    parser.add_argument(
        "--protocol-surface",
        choices=["legacy", "safe_choice"],
        default="legacy",
        help="Model-facing protocol surface. safe_choice exposes only lettered options.",
    )
    parser.add_argument("--choice-count", type=int, default=4)
    parser.add_argument("--choice-seed", type=int, default=0)
    parser.add_argument("--choice-bank-jsonl", help="task_id/state_id keyed JSONL option bank for safe_choice.")
    parser.add_argument("--write-human-trace-md", action="store_true")
    parser.add_argument("--atomic-template-path")
    parser.add_argument("--subtask-template-path")
    parser.add_argument("--exclusive-in-view", choices=["true", "false"], default="true")
    parser.add_argument("--memory-file")
    parser.add_argument("--unique-state-image-dir")
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help="In semantic_tool mode, expose only action fields in choose_action and do not ask the model for brief_reason or diagnostics.",
    )
    parser.add_argument(
        "--generic-invalid-action-feedback",
        action="store_true",
        help="Return a generic non-executable message for invalid actions instead of revealing missing preconditions such as in_view or with_reach.",
    )
    parser.add_argument(
        "--no-image-input",
        action="store_true",
        help="Do not attach observation images to the acting model; keep the same text/options for language-bias controls.",
    )
    parser.add_argument(
        "--codex-direct-image-input",
        action="store_true",
        help=(
            "For Codex/OpenClaw semantic_tool runs, attach observation images directly in MCP tool results "
            "instead of text-only image paths plus .wildclaw_image.py."
        ),
    )
    parser.add_argument(
        "--codex-mcp-only-tools",
        action="store_true",
        help=(
            "For the Codex backend, route Responses requests through the in-container DashScope adapter "
            "and forward only TongSIM MCP tools to the model."
        ),
    )
    parser.add_argument(
        "--mcp-compress-images",
        action="store_true",
        help="Compress MCP observation images before returning image content blocks. Disabled by default.",
    )
    parser.add_argument(
        "--mcp-image-max-side",
        type=int,
        default=768,
        help="Max long side in pixels when --mcp-compress-images is enabled.",
    )
    parser.add_argument(
        "--mcp-image-max-base64-chars",
        type=int,
        default=70000,
        help="Target max base64 character count per MCP image when --mcp-compress-images is enabled.",
    )
    parser.add_argument(
        "--mcp-image-jpeg-quality",
        type=int,
        default=70,
        help="Initial JPEG quality when --mcp-compress-images is enabled.",
    )
    parser.add_argument(
        "--export-checkpoint-dir",
        help="Export the completed native agent session to this directory before cleanup.",
    )
    parser.add_argument(
        "--decision-diagnostics",
        action="store_true",
        help="After the run, write sidecar image/action diagnostics without feeding them back to the acting model.",
    )
    args = parser.parse_args()
    runtime_mode = activate_runtime(args.runtime)
    if is_native_runtime(runtime_mode):
        # Native harness CLIs are shipped beside the evaluation Python. Make
        # that bundle visible to Codex/Claude just as the Docker image PATH
        # would make its CLI visible in container mode.
        # Keep the lexical venv path: resolve() follows the hermes_python
        # symlink and loses the enclosing native bundle directory.
        bundle_root = native_bundle_root()
        if bundle_root is None or not (bundle_root / "hermes").is_dir():
            bundle_root = Path(current_python()).absolute().parents[3]
        # Fall back to the active Python only when no valid shared bundle was
        # supplied. The evaluation Python may live outside the native bundle.
        os.environ["TONGBENCH_NATIVE_BUNDLE_ROOT"] = str(bundle_root)
    if args.protocol_surface == "safe_choice":
        if not args.choice_bank_jsonl:
            raise ValueError("--protocol-surface safe_choice requires --choice-bank-jsonl; generated fallback choices are not allowed for formal runs.")
        args.action_interface = "full_action"

    semantic_tool_backends = {"hermesagent", "openclaw", "codex", "claudecode"}
    if args.framework_interaction_mode == "semantic_tool" and args.backend not in semantic_tool_backends:
        raise ValueError(
            "--framework-interaction-mode semantic_tool is currently supported only with "
            "--backend hermesagent, openclaw, codex, or claudecode."
        )

    task_dir = Path(args.task_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_template_path = _resolve_default_template_path(
        args.atomic_template_path,
        "TONGSIM_ATOMIC_TEMPLATE_PATH",
        UPDATED_ATOMIC_TEMPLATE_NAME,
        task_dir.parent,
        output_dir.parent,
    )
    subtask_template_path = _resolve_default_template_path(
        args.subtask_template_path,
        "TONGSIM_SUBTASK_TEMPLATE_PATH",
        UPDATED_SUBTASK_TEMPLATE_NAME,
        task_dir.parent,
        output_dir.parent,
    )

    runtime_root = output_dir / "private_runtime"
    runtime = TongSimFrameworkRuntime(runtime_root)
    session = runtime.create_session(
        task_dir=task_dir,
        output_dir=output_dir,
        action_interface=args.action_interface,
        action_library_mode=args.action_library_mode,
        atomic_template_path=atomic_template_path,
        subtask_template_path=subtask_template_path,
        exclusive_in_view=str(args.exclusive_in_view).lower() == "true",
        max_steps=int(args.max_steps),
        public_interface_mode=args.public_interface_mode,
        memory_file=Path(args.memory_file).resolve() if args.memory_file else None,
        unique_state_image_dir=Path(args.unique_state_image_dir).resolve() if args.unique_state_image_dir else None,
        decision_only=bool(args.decision_only),
        generic_invalid_action_feedback=bool(args.generic_invalid_action_feedback),
        description_field=args.description_field,
        protocol_surface=args.protocol_surface,
        choice_count=int(args.choice_count),
        choice_seed=int(args.choice_seed),
        choice_bank_path=Path(args.choice_bank_jsonl).resolve() if args.choice_bank_jsonl else None,
        write_human_trace_md=bool(args.write_human_trace_md),
    )
    _allow_container_bridge_writes(Path(session.bridge_dir))

    task_id = f"tongsim_{uuid.uuid4().hex[:8]}_{args.backend}_{session.task_id}"
    public_workspace = Path(session.public_workspace_dir)
    if is_native_runtime(runtime_mode):
        _nativeize_session_configs(public_workspace, session.bridge_dir)

    prompt = build_framework_agent_prompt(
        load_tongsim_task(task_dir, description_field=args.description_field),
        max_steps=int(args.max_steps),
        action_interface=args.action_interface,
        action_library_mode=args.action_library_mode,
        public_interface_mode=args.public_interface_mode,
        interaction_mode=args.framework_interaction_mode,
        decision_only=bool(args.decision_only),
        protocol_surface=args.protocol_surface,
    )
    _write_json(
        output_dir / "protocol_manifest.json",
        build_protocol_manifest(
            task_id=session.task_id,
            phase="single",
            self_evolution_mode="basic",
            protocol_surface=args.protocol_surface,
            prompt=prompt,
            post_task_prompt=None,
            choice_count=int(args.choice_count),
            choice_seed=int(args.choice_seed),
            choice_bank_path=str(args.choice_bank_jsonl or ""),
            steps=[
                {
                    "index": 1,
                    "task_id": session.task_id,
                    "level": _task_level(session.task_id),
                    "max_steps": int(args.max_steps),
                }
            ],
        ),
    )

    readme_text = prompt
    for target in (public_workspace / "README_TONGSIM.md", public_workspace / "exec" / "README_TONGSIM.md"):
        target.write_text(readme_text, encoding="utf-8")

    bridge_script = ROOT / "scripts" / "tongsim_framework_bridge.py"
    bridge_stdout_handle = (output_dir / "bridge_stdout.log").open("w", encoding="utf-8")
    bridge_stderr_handle = (output_dir / "bridge_stderr.log").open("w", encoding="utf-8")
    bridge_proc = subprocess.Popen(
        [
            current_python(),
            str(bridge_script),
            "serve",
            "--runtime-root",
            str(runtime_root),
            "--session-id",
            session.session_id,
        ],
        cwd=ROOT,
        stdout=bridge_stdout_handle,
        stderr=bridge_stderr_handle,
        text=True,
    )

    backend = _make_backend(
        args.backend,
        gateway_port=args.gateway_port,
        openclaw_image_model=args.openclaw_image_model,
    )
    models_config = _load_models_config(Path(args.models_config).resolve()) if args.models_config else None
    models_config = _apply_provider_overrides(
        models_config,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    # Put the unique run id before the long task id. Some framework Docker
    # backends truncate container names; a suffix-only nonce can be chopped off
    # and collide when different models run the same task/backend in parallel.
    execution = None
    exported_checkpoint = None
    usage: dict | None = None
    result_error: str | None = None

    runtime_options = {
        "runtime": runtime_mode,
        "stop_when_finalized": args.backend != "hermesagent",
        "runtime_meta_path": str(
            runtime_root / "sessions" / session.session_id / "runtime_meta.json"
        ),
    }
    request_overrides = {}
    if args.temperature is not None:
        request_overrides["temperature"] = float(args.temperature)
    if args.seed is not None:
        request_overrides["seed"] = int(args.seed)
    if args.responses_non_stream:
        request_overrides["__responses_non_stream__"] = True
    if request_overrides:
        runtime_options["hermes"] = {"request_overrides": request_overrides}
    if args.framework_interaction_mode == "semantic_tool":
        mcp_python_command = args.mcp_python_command
        if not mcp_python_command:
            if is_native_runtime(runtime_mode):
                configured_python = os.environ.get(
                    "TONGBENCH_NATIVE_MCP_PYTHON", ""
                ).strip()
                bundle_root = native_bundle_root()
                bundled_python = (
                    bundle_root / "hermes" / ".venv" / "bin" / "python3"
                    if bundle_root is not None
                    else None
                )
                if configured_python:
                    mcp_python_command = configured_python
                elif bundled_python is not None and bundled_python.is_file():
                    # The native bundle owns the MCP dependencies used by all
                    # four harnesses. The evaluation Python may not include
                    # the `mcp` package and cannot safely launch this server.
                    mcp_python_command = str(bundled_python)
                else:
                    mcp_python_command = current_python()
            else:
                mcp_python_command = (
                    "/opt/hermes/.venv/bin/python3"
                    if args.backend == "hermesagent"
                    else "python3"
                )
        direct_image_input = args.backend in {"codex", "openclaw"} and bool(args.codex_direct_image_input)
        codex_direct_image_input = args.backend == "codex" and bool(args.codex_direct_image_input)
        tool_names = protocol_tool_names(args.protocol_surface)
        mcp_env = mcp_image_env(
            backend=args.backend,
            no_image_input=bool(args.no_image_input),
            direct_image_input=direct_image_input,
            compress_images=bool(args.mcp_compress_images),
            context_mode=args.context_mode,
            image_max_side=args.mcp_image_max_side,
            image_max_base64_chars=args.mcp_image_max_base64_chars,
            image_jpeg_quality=args.mcp_image_jpeg_quality,
        )
        if args.backend in {"codex", "claudecode", "openclaw"} and not is_native_runtime(runtime_mode):
            mcp_env["PATH"] = "/root/miniconda3/envs/eval/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        mcp_servers = {
            "tongsim": {
                "command": mcp_python_command,
                "args": [
                    native_path(task_id, "/tmp_workspace/tongsim_mcp_server.py")
                    if is_native_runtime(runtime_mode)
                    else "/tmp_workspace/tongsim_mcp_server.py",
                    "--session-config",
                    native_path(task_id, "/tmp_workspace/.tongsim_session.json")
                    if is_native_runtime(runtime_mode)
                    else "/tmp_workspace/.tongsim_session.json",
                ],
                "env": mcp_env,
                "timeout": 120,
                "connect_timeout": 60,
                "enabled_tools": list(tool_names),
            }
        }
        if is_native_runtime(runtime_mode):
            def _nativeize_mcp_value(value):
                if isinstance(value, str) and value.startswith("/tmp_workspace"):
                    return native_path(task_id, value)
                if isinstance(value, list):
                    return [_nativeize_mcp_value(item) for item in value]
                if isinstance(value, dict):
                    return {key: _nativeize_mcp_value(item) for key, item in value.items()}
                return value

            mcp_servers = _nativeize_mcp_value(mcp_servers)
        if args.backend == "hermesagent":
            enabled_toolsets = ["mcp-tongsim"]
            hermes_options = runtime_options.setdefault("hermes", {})
            hermes_options.update(
                {
                    "mcp_servers": mcp_servers,
                    "enabled_toolsets": enabled_toolsets,
                    "max_iterations": int(args.max_iterations),
                    "context_mode": args.context_mode,
                }
            )
        elif args.backend == "codex":
            runtime_options["codex"] = {
                "mcp_servers": codex_mcp_servers(mcp_servers, tool_names=tool_names),
                "disable_image_helper": codex_direct_image_input,
                "dashscope_responses_adapter": bool(codex_direct_image_input or args.codex_mcp_only_tools),
                "mcp_only_tools": bool(args.codex_mcp_only_tools),
            }
        elif args.backend == "claudecode":
            runtime_options["claudecode"] = {
                "mcp_servers": mcp_servers,
                "max_iterations": int(args.max_iterations),
                "allowed_tools": claudecode_allowed_tools(
                    mcp_servers,
                    tool_names=tool_names,
                ),
            }
        elif args.backend == "openclaw":
            runtime_options["openclaw"] = {
                "mcp_servers": mcp_servers,
                "decision_only": bool(args.decision_only),
            }

    spec = AgentTaskSpec(
        task_id=task_id,
        task={
            "task_id": session.task_id,
            "category": "TongSIM",
            "env": "",
            "skills": "",
            "warmup": "",
            "skills_path": str((ROOT / "skills").resolve()),
            "bind_mounts": [
                {
                    "host_path": str(Path(session.bridge_dir).resolve()),
                    "container_path": "/tongsim_bridge",
                    "mode": "rw",
                }
            ],
        },
        workspace_path=str(public_workspace),
        prompt=prompt,
        timeout_seconds=int(args.timeout_sec),
        output_dir=output_dir,
        model=args.model,
        thinking=args.thinking,
        models_config=models_config,
        runtime_options=runtime_options,
    )

    try:
        execution = backend.run_task(spec)
        result_error = execution.error
        if args.export_checkpoint_dir:
            if result_error:
                raise RuntimeError(
                    f"Cannot export a failed run as a checkpoint: {result_error}"
                )
            exported_checkpoint = backend.export_checkpoint(
                task_id,
                Path(args.export_checkpoint_dir).expanduser().resolve(),
                metadata={
                    "task_id": session.task_id,
                    "backend": args.backend,
                    "model": args.model,
                    "runtime": runtime_mode,
                    "max_steps": int(args.max_steps),
                },
            )
        try:
            backend.prepare_grading_transcript(task_id)
        except Exception:
            pass
        usage = backend.collect_usage(task_id, output_dir, execution.elapsed_time)
    finally:
        try:
            runtime.finish(session.session_id)
        except Exception:
            pass
        Path(session.bridge_dir, "STOP").write_text("stop\n", encoding="utf-8")
        try:
            bridge_proc.wait(timeout=5)
        except Exception:
            bridge_proc.terminate()
        bridge_stdout_handle.close()
        bridge_stderr_handle.close()
        if execution is not None:
            if execution.gateway_proc is not None:
                try:
                    execution.gateway_proc.terminate()
                except Exception:
                    pass
            for proc in (execution.gateway_proc, execution.agent_proc):
                if proc is not None:
                    try:
                        close_proc_log(proc)
                    except Exception:
                        pass
        remove_container(task_id)

    result = runtime.export_outputs(session.session_id, output_dir)
    if exported_checkpoint is not None:
        for name in ("task_outcome.json", "task_outcome.md"):
            outcome_path = output_dir / "task_output" / name
            if outcome_path.is_file():
                shutil.copy2(outcome_path, exported_checkpoint.path / name)
        result["exported_checkpoint"] = exported_checkpoint.as_dict()
        _write_json(output_dir / "exported_checkpoint.json", exported_checkpoint.as_dict())
    diagnostics_path = None
    diagnostics_error = None
    if args.decision_diagnostics:
        try:
            diagnostics_api_key = ""
            diagnostics_base_url = ""
            if hasattr(backend, "_resolve_runtime_provider"):
                diagnostics_api_key, diagnostics_base_url = backend._resolve_runtime_provider(args.model, models_config)
            diagnostics_path = write_sidecar_decision_diagnostics(
                output_dir=output_dir,
                api_key=diagnostics_api_key,
                base_url=diagnostics_base_url,
                model=args.model,
            )
            result["sidecar_decision_diagnostics_path"] = str(diagnostics_path)
        except Exception as exc:
            diagnostics_error = f"{type(exc).__name__}: {exc}"
            result["sidecar_decision_diagnostics_error"] = diagnostics_error
    if usage is None:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
            "elapsed_time": float(args.timeout_sec),
        }
    usage["framework_backend"] = args.backend
    usage["atomic_template_path"] = str(atomic_template_path or "")
    usage["subtask_template_path"] = str(subtask_template_path or "")
    if execution is not None and execution.agent_proc is not None:
        usage["agent_returncode"] = execution.agent_proc.returncode
    if execution is not None and execution.gateway_proc is not None:
        usage["gateway_returncode"] = execution.gateway_proc.returncode
    if diagnostics_path is not None:
        usage["sidecar_decision_diagnostics_path"] = str(diagnostics_path)
    if diagnostics_error is not None:
        usage["sidecar_decision_diagnostics_error"] = diagnostics_error
    if result_error:
        usage["error"] = result_error
    (output_dir / "usage.json").write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["score"], indent=2, ensure_ascii=False))
    return 0 if not result_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
