#!/usr/bin/env bash

# Resolve the interpreter for all handoff scripts.  A silently selected system
# Python can be an older version than the evaluation package supports.
resolve_python_bin() {
  local candidate="${PYTHON_BIN:-}"

  if [[ -z "$candidate" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    candidate="$CONDA_PREFIX/bin/python"
  fi
  if [[ -z "$candidate" && -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    candidate="$VIRTUAL_ENV/bin/python"
  fi
  if [[ -z "$candidate" ]]; then
    echo "PYTHON_BIN is not set. Activate the documented conda/virtualenv or set PYTHON_BIN=/path/to/env/bin/python." >&2
    return 1
  fi

  if [[ "$candidate" != */* ]]; then
    candidate="$(command -v "$candidate" || true)"
  fi
  if [[ -z "$candidate" || ! -x "$candidate" ]]; then
    echo "Python executable is not usable: ${PYTHON_BIN:-$candidate}" >&2
    return 1
  fi

  if ! "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    echo "Python 3.10+ is required, but the selected interpreter is: $candidate" >&2
    "$candidate" --version >&2 || true
    return 1
  fi

  printf '%s\n' "$candidate"
}
