"""Runtime selection and path helpers for Docker and user-space execution.

The evaluation protocol is intentionally unaware of the runtime.  Docker mode
uses the host's Docker CLI; native mode prepends the bundled compatibility CLI
to PATH and maps the small container filesystem used by the runners into a
per-task directory under ``TONGBENCH_NATIVE_ROOT``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


RUNTIME_ENV = "TONGBENCH_RUNTIME"
NATIVE_ROOT_ENV = "TONGBENCH_NATIVE_ROOT"
NATIVE_MCP_PYTHON_ENV = "TONGBENCH_NATIVE_MCP_PYTHON"
NATIVE_BIN_DIR_ENV = "TONGBENCH_NATIVE_BIN_DIR"
NATIVE_BUNDLE_ROOT_ENV = "TONGBENCH_NATIVE_BUNDLE_ROOT"
VALID_RUNTIMES = {"docker", "native"}


def normalize_runtime(value: str | None = None) -> str:
    runtime = str(value or os.environ.get(RUNTIME_ENV, "docker")).strip().lower()
    if runtime not in VALID_RUNTIMES:
        raise ValueError(
            f"Unsupported runtime {runtime!r}; choose one of: docker, native"
        )
    return runtime


def native_root() -> Path:
    raw = os.environ.get(NATIVE_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".cache" / "tongbench" / "native-runtime").resolve()


def native_bin_dir() -> Path:
    # .../public/eval/src/tongbench_eval/utils/native_runtime.py
    return Path(__file__).resolve().parents[3] / "runtime" / "native" / "bin"


def native_bundle_root() -> Path | None:
    """Return the optional shared native harness bundle directory."""

    raw = os.environ.get(NATIVE_BUNDLE_ROOT_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def activate_runtime(runtime: str | None = None) -> str:
    """Activate a runtime for this process and child processes.

    This only changes environment variables.  It does not start a container or
    create a native task directory; the runtime CLI creates that at ``run``.
    """

    selected = normalize_runtime(runtime)
    os.environ[RUNTIME_ENV] = selected
    if selected == "native":
        bindir = native_bin_dir()
        if not bindir.is_dir():
            raise FileNotFoundError(f"Native runtime CLI directory not found: {bindir}")
        os.environ[NATIVE_ROOT_ENV] = str(native_root())
        extra_bin = os.environ.get(NATIVE_BIN_DIR_ENV, "").strip()
        path_prefix = [str(bindir)]
        if extra_bin:
            path_prefix.append(str(Path(extra_bin).expanduser().resolve()))
        bundle = native_bundle_root()
        if bundle is not None:
            bundle_bin = bundle / "bin"
            if bundle_bin.is_dir():
                path_prefix.append(str(bundle_bin))
        os.environ["PATH"] = os.pathsep.join(path_prefix) + os.pathsep + os.environ.get("PATH", "")
    return selected


def is_native_runtime(runtime: str | None = None) -> bool:
    return normalize_runtime(runtime) == "native"


def _safe_task_id(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip(".") or "task"


def native_task_root(task_id: str) -> Path:
    return native_root() / "tasks" / _safe_task_id(task_id)


def native_path(task_id: str, container_path: str) -> str:
    """Return the host path corresponding to a native task path.

    This is mainly needed for MCP configuration, because the agent CLI reads
    that configuration directly and therefore cannot rely on the runtime CLI
    to translate its JSON values later.
    """

    path = str(container_path or "/")
    return str(native_task_root(task_id) / path.lstrip("/"))
