#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


SETTINGS = ("basic_l2", "zero_shot_l4", "in_context_l4", "skill_l4")
HARNESSES = ("hermesagent", "codex", "claudecode", "openclaw")
QWEN_ONLY = ("qwen",)
SUPPORTED_MODELS = ("qwen", "glm", "luna")

_LUNA_GATEWAY_PROCESS: subprocess.Popen[str] | None = None
_LUNA_GATEWAY_LOG_HANDLE = None


def split_words(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def normalize_openai_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    return base_url if base_url.endswith("/v1") else base_url + "/v1"


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _stop_luna_gateway() -> None:
    global _LUNA_GATEWAY_PROCESS, _LUNA_GATEWAY_LOG_HANDLE
    process = _LUNA_GATEWAY_PROCESS
    _LUNA_GATEWAY_PROCESS = None
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if _LUNA_GATEWAY_LOG_HANDLE is not None:
        _LUNA_GATEWAY_LOG_HANDLE.close()
        _LUNA_GATEWAY_LOG_HANDLE = None


def start_luna_gateway(args: argparse.Namespace) -> None:
    """Start the Luna-only Chat/Anthropic-to-Responses protocol bridge."""
    global _LUNA_GATEWAY_PROCESS, _LUNA_GATEWAY_LOG_HANDLE
    target_base_url = normalize_openai_base_url(
        env_first("LUNA_BASE_URL") or "https://www.dmxapi.cn/v1"
    )
    api_key = env_first("LUNA_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("Missing Luna API key. Set LUNA_API_KEY.")
    if args.runtime != "native":
        raise RuntimeError(
            "The Luna protocol bridge currently requires --runtime native so all harnesses "
            "can reach the same loopback endpoint."
        )
    if args.dry_run:
        os.environ["LUNA_GATEWAY_BASE_URL"] = "http://127.0.0.1:1"
        return

    port = _free_loopback_port()
    gateway_base_url = f"http://127.0.0.1:{port}"
    log_path = args.output_root / args.stage / "_protocol_gateways" / f"luna_{os.getpid()}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LUNA_GATEWAY_LOG_HANDLE = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "TONGBENCH_GATEWAY_BASE_URL": target_base_url,
            "TONGBENCH_GATEWAY_API_KEY": api_key,
            "TONGBENCH_GATEWAY_MODEL": args.model_override or env_first("LUNA_MODEL") or "gpt-5.6-luna",
            "TONGBENCH_GATEWAY_REASONING_EFFORT": args.thinking or "medium",
            "TONGBENCH_GATEWAY_ATTACH_LOCAL_IMAGES": "1",
            # DMX may route consecutive Responses requests to different Azure
            # resources for Terra.  Do not replay provider-owned encrypted
            # reasoning state in that case; benchmark content is unchanged.
            "TONGBENCH_GATEWAY_DROP_ENCRYPTED_REASONING": (
                "1" if (args.model_override or env_first("LUNA_MODEL") or "gpt-5.6-luna") == "gpt-5.6-terra" else "0"
            ),
        }
    )
    _LUNA_GATEWAY_PROCESS = subprocess.Popen(
        [
            str(args.python_bin),
            "-m",
            "tongbench_eval.agents.responses_protocol_gateway",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=_LUNA_GATEWAY_LOG_HANDLE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    atexit.register(_stop_luna_gateway)
    health_url = gateway_base_url + "/health"
    last_error = "gateway did not answer"
    for _ in range(100):
        if _LUNA_GATEWAY_PROCESS.poll() is not None:
            last_error = f"gateway exited with code {_LUNA_GATEWAY_PROCESS.returncode}"
            break
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    os.environ["LUNA_GATEWAY_BASE_URL"] = gateway_base_url
                    return
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.1)
    _stop_luna_gateway()
    raise RuntimeError(f"Luna protocol gateway failed to start: {last_error}; log={log_path}")


def resolve_template_path(
    repo_root: Path,
    project_root: Path,
    explicit_path: Path | None,
    env_name: str,
    filename: str,
) -> Path:
    """Find a tracked template before falling back to the legacy project layout."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    configured = os.environ.get(env_name)
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            repo_root / "For_user" / "templates" / filename,
            project_root / "important_results_taskgen" / "8.10" / "input" / filename,
        ]
    )
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing required task template {filename}. Searched:\n  {searched}\n"
        f"Set {env_name} to an explicit file if using a custom layout."
    )


def model_config(
    model_key: str,
    backend: str,
    model_override: str | None = None,
) -> dict[str, str | list[str]]:
    if model_key == "qwen":
        base_url = env_first("QWEN_BASE_URL", "DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if backend == "claudecode":
            base_url = env_first("QWEN_CLAUDECODE_BASE_URL") or "https://dashscope.aliyuncs.com/apps/anthropic"
        return {
            "model": model_override or env_first("QWEN_MODEL") or "qwen3.8-flash",
            "base_url": base_url,
            "api_key": env_first("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
            "extra_args": [],
        }
    if model_key == "glm":
        base_url = env_first("GLM_BASE_URL", "DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if backend == "claudecode":
            # Claude Code is routed through the local DashScope protocol
            # gateway, so its upstream must remain the OpenAI-compatible
            # DashScope endpoint.
            base_url = env_first("GLM_BASE_URL", "DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return {
            "model": model_override or env_first("GLM_MODEL") or "ZHIPU/GLM-5.3-Flash",
            "base_url": base_url,
            "api_key": env_first("GLM_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            "extra_args": [],
        }
    if model_key == "luna":
        upstream_model = model_override or env_first("LUNA_MODEL") or "gpt-5.6-luna"
        target_base_url = normalize_openai_base_url(
            env_first("LUNA_BASE_URL") or "https://www.dmxapi.cn/v1"
        )
        gateway_base_url = env_first("LUNA_GATEWAY_BASE_URL").rstrip("/")
        if not gateway_base_url:
            base_url = target_base_url
        elif backend == "claudecode":
            base_url = gateway_base_url
        else:
            base_url = gateway_base_url + "/v1"
        # Hermes auto-selects its Responses execution path for any gpt-5*
        # client model name. Use a local protocol alias so its proven Chat MCP
        # image path is retained; the gateway still sends upstream_model.
        client_model = (
            env_first("LUNA_HERMES_CLIENT_MODEL") or "luna-5.6-chat"
            if backend == "hermesagent"
            else upstream_model
        )
        return {
            "model": client_model,
            "base_url": base_url,
            "api_key": env_first("LUNA_API_KEY"),
            # This Responses model rejects sampling seed and temperature.
            # Choice placement remains deterministic through --choice-seed.
            "extra_args": ["--no-model-seed", "--no-model-temperature"],
        }
    raise ValueError(f"Unknown model key: {model_key}")


def backend_extra_args(backend: str) -> list[str]:
    if backend == "hermesagent":
        return []
    if backend == "codex":
        return ["--codex-direct-image-input", "--codex-mcp-only-tools"]
    if backend == "claudecode":
        return []
    if backend == "openclaw":
        return ["--codex-direct-image-input"]
    raise ValueError(f"Unknown harness/backend: {backend}")


def common_eval_args(args: argparse.Namespace, model_key: str, backend: str) -> list[str]:
    cfg = model_config(model_key, backend, args.model_override)
    api_key = str(cfg["api_key"])
    if not api_key and args.dry_run:
        api_key = "DRY_RUN_API_KEY"
    if not api_key:
        raise RuntimeError(
            f"Missing API key for model={model_key}. Set LUNA_API_KEY, GLM_API_KEY, "
            "QWEN_API_KEY, or DASHSCOPE_API_KEY."
        )
    common = [
        "--results-root",
        str(args.merged_results_root),
        "--backend",
        backend,
        "--model",
        str(cfg["model"]),
        "--runtime",
        str(args.runtime),
        "--api-key",
        api_key,
        "--base-url",
        str(cfg["base_url"]),
        "--description-field",
        "description_highlevel_hard",
        "--framework-interaction-mode",
        "semantic_tool",
        "--public-interface-mode",
        "natural_language",
        "--protocol-surface",
        "safe_choice",
        "--choice-bank-jsonl",
        str(args.merged_results_root / "option_bank_highlevel_v2_smoke.jsonl"),
        "--choice-count",
        "4",
        "--choice-seed",
        str(args.seed),
        "--write-human-trace-md",
        "--generic-invalid-action-feedback",
        "--max-steps-extra",
        str(args.extra_steps),
        "--timeout-sec",
        str(args.timeout_sec),
        "--max-iterations",
        str(args.max_iterations),
        "--context-mode",
        str(args.context_mode),
        "--atomic-template-path",
        str(args.atomic_template_path),
        "--subtask-template-path",
        str(args.subtask_template_path),
        *list(cfg["extra_args"]),
        *backend_extra_args(backend),
    ]
    if args.thinking:
        common.extend(["--thinking", str(args.thinking)])
    return common


def redacted_command(cmd: list[str]) -> list[str]:
    redacted = list(cmd)
    for index, part in enumerate(redacted[:-1]):
        if part == "--api-key":
            redacted[index + 1] = "***"
    return redacted


def run_command(cmd: list[str], dry_run: bool, *, model_key: str | None = None) -> int:
    print("\n$ " + " ".join(shlex.quote(part) for part in redacted_command(cmd)), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    if model_key == "luna":
        env["TONGBENCH_MCP_CANONICAL_IMAGE_CONTENT"] = "1"
    else:
        env.pop("TONGBENCH_MCP_CANONICAL_IMAGE_CONTENT", None)
    return subprocess.run(cmd, env=env).returncode


def output_dir(args: argparse.Namespace, model_key: str, backend: str, setting: str) -> Path:
    return args.output_root / args.stage / model_key / backend / setting


def _append_task_ids(command: list[str], task_ids: list[str] | None) -> None:
    for task_id in task_ids or []:
        command.extend(["--task-id", task_id])


def run_setting(
    args: argparse.Namespace,
    model_key: str,
    backend: str,
    setting: str,
    *,
    task_ids: list[str] | None = None,
    export_checkpoint_root: Path | None = None,
    reuse_manifest: Path | None = None,
) -> int:
    common = common_eval_args(args, model_key, backend)
    py = str(args.python_bin)
    out = output_dir(args, model_key, backend, setting)
    out.mkdir(parents=True, exist_ok=True)
    if setting == "basic_l2":
        cmd = [
            py,
            "-m",
            "tongbench_eval.cli.framework.ready_single_tasks",
            "--ready-json",
            str(args.subset_root / "ready_l2_tasks_with_images.json"),
            "--output-dir",
            str(out),
            "--composite-only",
            "--parallel-jobs",
            str(args.parallel_jobs),
            "--skip-existing",
            "--l2-only",
            *common,
        ]
        _append_task_ids(cmd, task_ids)
        if export_checkpoint_root is not None:
            cmd.extend(["--export-checkpoint-root", str(export_checkpoint_root)])
        return run_command(cmd, args.dry_run, model_key=model_key)
    if setting == "zero_shot_l4":
        cmd = [
            py,
            "-m",
            "tongbench_eval.cli.framework.ready_single_tasks",
            "--ready-json",
            str(args.subset_root / "ready_l4_tasks_with_images.json"),
            "--output-dir",
            str(out),
            "--composite-only",
            "--parallel-jobs",
            str(args.parallel_jobs),
            "--skip-existing",
            *common,
        ]
        _append_task_ids(cmd, task_ids)
        return run_command(cmd, args.dry_run, model_key=model_key)
    if setting in {"in_context_l4", "skill_l4"}:
        mode = "in_context" if setting == "in_context_l4" else "skill"
        cmd = [
            py,
            "-m",
            "tongbench_eval.cli.framework.dialogue",
            "--ready-json",
            str(args.subset_root / "ready_l4_tasks_with_images.json"),
            "--output-dir",
            str(out),
            "--hide-decomposition",
            "--self-evolution-mode",
            mode,
            "--learning-plan-json",
            str(args.subset_root / "learning_plan.json"),
            "--learning-checkpoint-root",
            str(args.output_root / args.stage / "_learning_checkpoints" / model_key / backend),
            "--parallel-jobs",
            str(args.self_evolution_parallel_jobs),
            "--skip-existing",
            *common,
        ]
        _append_task_ids(cmd, task_ids)
        if reuse_manifest is not None:
            cmd.extend(["--reuse-learning-manifest", str(reuse_manifest)])
        return run_command(cmd, args.dry_run, model_key=model_key)
    raise ValueError(f"Unknown setting: {setting}")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ready_task_ids(path: Path) -> list[str]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Ready task JSON must be a list: {path}")
    task_ids = [
        str(item.get("task_id") or "").strip()
        for item in payload
        if isinstance(item, dict)
    ]
    if not task_ids or any(not task_id for task_id in task_ids):
        raise ValueError(f"Ready task JSON has missing task_id values: {path}")
    return task_ids


def learning_targets(path: Path) -> list[dict[str, object]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("targets"), list):
        raise ValueError(f"Learning plan must contain a targets list: {path}")
    targets: list[dict[str, object]] = []
    for item in payload["targets"]:
        if not isinstance(item, dict):
            continue
        target_task_id = str(item.get("target_task_id") or item.get("task_id") or "").strip()
        learning_ids = item.get("learning_task_ids") or item.get("icl_learning_task_ids") or []
        learning_task_ids = [str(value).strip() for value in learning_ids if str(value).strip()]
        if target_task_id and learning_task_ids:
            targets.append(
                {
                    "target_task_id": target_task_id,
                    "learning_task_ids": learning_task_ids,
                }
            )
    if not targets:
        raise ValueError(f"Learning plan has no usable target mappings: {path}")
    return targets


def stage_selection(args: argparse.Namespace) -> tuple[list[str], list[dict[str, object]], list[str]]:
    l2_ids = ready_task_ids(args.subset_root / "ready_l2_tasks_with_images.json")
    l4_ids = ready_task_ids(args.subset_root / "ready_l4_tasks_with_images.json")
    targets_by_id = {
        str(item["target_task_id"]): item
        for item in learning_targets(args.subset_root / "learning_plan.json")
    }
    selected_l4_ids = l4_ids[:1] if args.stage == "stage0_smoke" else l4_ids
    missing_targets = [task_id for task_id in selected_l4_ids if task_id not in targets_by_id]
    if missing_targets:
        raise KeyError(f"Learning plan is missing selected L4 targets: {missing_targets}")
    selected_targets = [targets_by_id[task_id] for task_id in selected_l4_ids]

    if args.stage == "stage0_smoke":
        selected_l2_ids = list(
            dict.fromkeys(
                task_id
                for target in selected_targets
                for task_id in target["learning_task_ids"]
            )
        )
    else:
        selected_l2_ids = l2_ids

    unknown_l2_ids = sorted(set(selected_l2_ids) - set(l2_ids))
    if unknown_l2_ids:
        raise KeyError(f"Learning plan references L2 tasks outside the selected subset: {unknown_l2_ids}")
    return selected_l2_ids, selected_targets, selected_l4_ids


def exported_checkpoint_dir(basic_output: Path, task_id: str) -> Path:
    exported_path = basic_output / "task_runs" / task_id / "exported_checkpoint.json"
    if not exported_path.is_file():
        raise FileNotFoundError(f"Missing exported L2 checkpoint record: {exported_path}")
    payload = read_json(exported_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid exported checkpoint record: {exported_path}")
    checkpoint_value = str(payload.get("path") or payload.get("checkpoint_dir") or "").strip()
    checkpoint_dir = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_value or not (checkpoint_dir / "agent_checkpoint.json").is_file():
        raise FileNotFoundError(f"Incomplete exported L2 checkpoint: {checkpoint_dir}")
    return checkpoint_dir


def build_learning_archive(
    args: argparse.Namespace,
    model_key: str,
    backend: str,
    l2_task_ids: list[str],
    targets: list[dict[str, object]],
) -> Path:
    cfg = model_config(model_key, backend, args.model_override)
    source_manifest = (
        args.output_root
        / args.stage
        / "_learning_archive_sources"
        / model_key
        / backend
        / "source_manifest.json"
    )
    archive_root = (
        args.output_root
        / args.stage
        / "_learning_archives"
        / model_key
        / backend
    )
    basic_output = output_dir(args, model_key, backend, "basic_l2")
    if args.dry_run:
        print(f"\n[dry-run] build learning archive at {archive_root}", flush=True)
        return archive_root / "learning_reuse_manifest.json"

    sources = [
        {
            "task_id": task_id,
            "checkpoint_dir": str(exported_checkpoint_dir(basic_output, task_id)),
        }
        for task_id in l2_task_ids
    ]
    write_json(
        source_manifest,
        {
            "schema_version": "tongbench_l2_zero_shot_source_manifest_v1",
            "mode": "independent_l2_zero_shot_archive",
            "backend": backend,
            "model": str(cfg["model"]),
            "l2_tasks": sources,
            "targets": targets,
        },
    )
    command = [
        str(args.python_bin),
        "-m",
        "tongbench_eval.cli.framework.build_learning_archive",
        "--source-manifest",
        str(source_manifest),
        "--output-root",
        str(archive_root),
        "--backend",
        backend,
        "--model",
        str(cfg["model"]),
        "--force",
    ]
    code = run_command(command, False)
    if code:
        raise RuntimeError(f"Learning archive construction failed for {model_key}/{backend}: rc={code}")
    reuse_manifest = archive_root / "learning_reuse_manifest.json"
    if not reuse_manifest.is_file():
        raise FileNotFoundError(f"Learning archive did not create reuse manifest: {reuse_manifest}")
    return reuse_manifest


def run_combo(
    args: argparse.Namespace,
    model_key: str,
    backend: str,
    settings: list[str],
    l2_task_ids: list[str],
    targets: list[dict[str, object]],
    l4_task_ids: list[str],
) -> int:
    needs_learning = any(setting in settings for setting in ("in_context_l4", "skill_l4"))
    needs_l2 = "basic_l2" in settings
    reuse_manifest: Path | None = None

    if needs_l2:
        checkpoint_root = output_dir(args, model_key, backend, "basic_l2") / "exported_checkpoints"
        code = run_setting(
            args,
            model_key,
            backend,
            "basic_l2",
            task_ids=l2_task_ids,
            export_checkpoint_root=checkpoint_root,
        )
        if code:
            return code
    if needs_learning:
        if args.learning_reuse_manifest is not None:
            reuse_manifest = args.learning_reuse_manifest
            if not args.dry_run and not reuse_manifest.is_file():
                raise FileNotFoundError(
                    f"Learning reuse manifest does not exist: {reuse_manifest}. "
                    "Run the Basic L2 stage and prepare the archive first."
                )
        elif "basic_l2" in settings:
            reuse_manifest = build_learning_archive(
                args,
                model_key,
                backend,
                l2_task_ids,
                targets,
            )
        else:
            raise RuntimeError(
                f"{model_key}/{backend} requested learning evaluation without a reuse manifest. "
                "Run Basic L2 once, prepare the learning archive, then reuse it."
            )

    for setting in ("zero_shot_l4", "in_context_l4", "skill_l4"):
        if setting not in settings:
            continue
        code = run_setting(
            args,
            model_key,
            backend,
            setting,
            task_ids=l4_task_ids,
            reuse_manifest=reuse_manifest if setting in {"in_context_l4", "skill_l4"} else None,
        )
        if code:
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run For_user L2/L4 eval stage.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=("stage0_smoke", "stage2_20pct", "stage3_full"),
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--subset-root", default=None, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python-bin", default=sys.executable, type=Path)
    parser.add_argument("--atomic-template-path", default=None, type=Path)
    parser.add_argument("--subtask-template-path", default=None, type=Path)
    parser.add_argument(
        "--runtime",
        choices=("docker", "native"),
        default=os.environ.get("TONGBENCH_RUNTIME", "native"),
        help="Execution runtime passed to every task: docker or native.",
    )
    parser.add_argument("--models", default="")
    parser.add_argument("--harnesses", default="")
    parser.add_argument("--settings", default="")
    parser.add_argument(
        "--model",
        dest="model_override",
        default=None,
        help="Provider model id override, e.g. qwen3.8-flash.",
    )
    parser.add_argument(
        "--thinking",
        default="",
        help="Reasoning/thinking setting forwarded to the underlying harness, e.g. medium.",
    )
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--self-evolution-parallel-jobs", type=int, default=2)
    parser.add_argument("--combo-parallel-jobs", type=int, default=1)
    parser.add_argument(
        "--learning-reuse-manifest",
        default=None,
        type=Path,
        help="Existing L2 zero-shot learning archive manifest. Learning settings never rerun Basic L2 when this is set.",
    )
    parser.add_argument(
        "--prepare-learning-archive",
        action="store_true",
        help="Build reusable learning archives from already completed Basic L2 checkpoints; do not run evaluation tasks.",
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-iterations", type=int, default=320)
    parser.add_argument("--extra-steps", type=int, default=6)
    parser.add_argument("--context-mode", choices=("full", "task_compact"), default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-full", action="store_true")
    args = parser.parse_args()

    if args.stage == "stage3_full" and not args.confirm_full:
        raise SystemExit("stage3_full is prepared but blocked by default. Pass --confirm-full to run it.")

    defaults_by_stage = {
        "stage0_smoke": {"models": QWEN_ONLY, "harnesses": HARNESSES, "settings": SETTINGS},
        "stage2_20pct": {"models": QWEN_ONLY, "harnesses": HARNESSES, "settings": SETTINGS},
        "stage3_full": {"models": QWEN_ONLY, "harnesses": HARNESSES, "settings": SETTINGS},
    }
    defaults = defaults_by_stage[args.stage]
    models = split_words(args.models, tuple(defaults["models"]))
    harnesses = split_words(args.harnesses, tuple(defaults["harnesses"]))
    settings = split_words(args.settings, tuple(defaults["settings"]))
    args.repo_root = args.repo_root.resolve()
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.learning_reuse_manifest is not None:
        args.learning_reuse_manifest = args.learning_reuse_manifest.resolve()
    args.atomic_template_path = resolve_template_path(
        args.repo_root,
        args.project_root,
        args.atomic_template_path,
        "ATOMIC_TEMPLATE_PATH",
        "atomic_templates_updated_env_v3.json",
    )
    args.subtask_template_path = resolve_template_path(
        args.repo_root,
        args.project_root,
        args.subtask_template_path,
        "SUBTASK_TEMPLATE_PATH",
        "subtask_templates_compressed_updated_env_v3.json",
    )
    args.subset_root = (
        args.subset_root.resolve()
        if args.subset_root is not None
        else args.data_root / "subsets" / args.stage
    )
    args.merged_results_root = args.data_root / "input" / "merged"

    env = os.environ.copy()
    pythonpath = str(args.repo_root / "public/eval/src")
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    os.environ["PYTHONPATH"] = pythonpath

    if "luna" in models and not args.prepare_learning_archive:
        start_luna_gateway(args)

    if not args.subset_root.exists():
        raise FileNotFoundError(f"Missing subset root: {args.subset_root}. Run prepare_inputs.sh first.")
    if not args.merged_results_root.exists():
        raise FileNotFoundError(f"Missing merged results root: {args.merged_results_root}. Run prepare_inputs.sh first.")

    invalid_settings = sorted(set(settings) - set(SETTINGS))
    if invalid_settings:
        raise SystemExit(f"Unknown setting(s): {', '.join(invalid_settings)}")
    invalid_models = sorted(set(models) - set(SUPPORTED_MODELS))
    if invalid_models:
        raise SystemExit(f"Unknown model key(s): {', '.join(invalid_models)}")
    invalid_harnesses = sorted(set(harnesses) - set(HARNESSES))
    if invalid_harnesses:
        raise SystemExit(f"Unknown harness/backend(s): {', '.join(invalid_harnesses)}")

    selected_l2_ids, selected_targets, selected_l4_ids = stage_selection(args)
    print(
        f"Stage {args.stage}: selected {len(selected_l2_ids)} L2 source tasks, "
        f"{len(selected_l4_ids)} L4 target tasks, settings={settings}",
        flush=True,
    )

    combos = [(model_key, backend) for model_key in models for backend in harnesses]

    if args.prepare_learning_archive:
        def prepare_one_combo(model_key: str, backend: str) -> tuple[str, str, int, str]:
            try:
                manifest = build_learning_archive(
                    args,
                    model_key,
                    backend,
                    selected_l2_ids,
                    selected_targets,
                )
                print(f"Prepared learning archive {model_key}/{backend}: {manifest}", flush=True)
                return model_key, backend, 0, ""
            except Exception as exc:
                return model_key, backend, 1, str(exc)

        if args.combo_parallel_jobs <= 1 or len(combos) <= 1:
            archive_results = [prepare_one_combo(model_key, backend) for model_key, backend in combos]
        else:
            with ThreadPoolExecutor(max_workers=args.combo_parallel_jobs) as executor:
                futures = {
                    executor.submit(prepare_one_combo, model_key, backend): (model_key, backend)
                    for model_key, backend in combos
                }
                archive_results = [future.result() for future in as_completed(futures)]
        failures = [result for result in archive_results if result[2]]
        if failures:
            print("\nLearning archive failures:", flush=True)
            for model_key, backend, code, error in failures:
                print(f"  {model_key}/{backend}: {code}: {error}", flush=True)
            return 1
        return 0

    if args.learning_reuse_manifest is None and any(
        setting in settings for setting in ("in_context_l4", "skill_l4")
    ):
        if len(combos) != 1:
            raise SystemExit(
                "Learning settings with multiple model/harness combos require one manifest per combo; "
                "use the stage wrapper or pass --learning-reuse-manifest per combo."
            )
        model_key, backend = combos[0]
        auto_manifest = (
            args.output_root
            / args.stage
            / "_learning_archives"
            / model_key
            / backend
            / "learning_reuse_manifest.json"
        )
        if not args.dry_run and not auto_manifest.is_file():
            raise SystemExit(
                f"Missing reusable learning archive: {auto_manifest}. "
                "Run Basic L2 first, then run with --prepare-learning-archive."
            )
        args.learning_reuse_manifest = auto_manifest

    def run_one_combo(model_key: str, backend: str) -> tuple[str, str, int, str]:
        try:
            code = run_combo(
                args,
                model_key,
                backend,
                settings,
                selected_l2_ids,
                selected_targets,
                selected_l4_ids,
            )
            return model_key, backend, code, ""
        except Exception as exc:
            return model_key, backend, 1, str(exc)

    results: list[tuple[str, str, int, str]] = []
    if args.combo_parallel_jobs <= 1 or len(combos) <= 1:
        results = [run_one_combo(model_key, backend) for model_key, backend in combos]
    else:
        with ThreadPoolExecutor(max_workers=args.combo_parallel_jobs) as executor:
            futures = {
                executor.submit(run_one_combo, model_key, backend): (model_key, backend)
                for model_key, backend in combos
            }
            for future in as_completed(futures):
                results.append(future.result())

    failures = [result for result in results if result[2]]
    if failures:
        print("\nFailures:", flush=True)
        for model_key, backend, code, error in failures:
            suffix = f": {error}" if error else ""
            print(f"  {model_key}/{backend}: {code}{suffix}", flush=True)
        return 1
    print("\nAll selected model/harness pipelines completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
