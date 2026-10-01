from __future__ import annotations

import shutil
import sys
import os
from pathlib import Path


def _which_with_expanded_path(command: str) -> str | None:
    direct = shutil.which(command)
    if direct:
        return direct
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        candidate = Path(entry).expanduser() / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def current_python() -> str:
    """Return a usable Python executable even when embedded launchers leave sys.executable empty."""
    return sys.executable or _which_with_expanded_path("python") or _which_with_expanded_path("python3") or "python3"
