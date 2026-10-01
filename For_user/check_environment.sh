#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHECK_DATA="${CHECK_DATA:-1}"
RUNTIME="${RUNTIME:-${TONGBENCH_RUNTIME:-native}}"
if [[ -z "${PYTHON_BIN:-}" && "$RUNTIME" == "native" && -n "${TONGBENCH_NATIVE_BUNDLE_ROOT:-}" && -x "$TONGBENCH_NATIVE_BUNDLE_ROOT/hermes/.venv/bin/python3" ]]; then
  PYTHON_BIN="$TONGBENCH_NATIVE_BUNDLE_ROOT/hermes/.venv/bin/python3"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
fi

FAILED=0

fail() {
  FAILED=1
  echo "FAIL: $*" >&2
}

ok() {
  echo "OK: $*"
}

section() {
  echo
  echo "== $* =="
}

section "Python"
if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  fail "Python not found. Set PYTHON_BIN or activate the conda environment."
else
  "$PYTHON_BIN" --version || fail "Python executable is not usable: $PYTHON_BIN"
  PYTHON_OK="$($PYTHON_BIN -c 'import sys; print(int(sys.version_info >= (3, 10)))' 2>/dev/null || echo 0)"
  if [[ "$PYTHON_OK" == "1" ]]; then
    ok "Python version is >= 3.10"
  else
    fail "Python >= 3.10 is required. Set PYTHON_BIN to the project environment."
  fi
fi

if [[ -x "$PYTHON_BIN" ]]; then
  PYTHONPATH="$REPO_ROOT/public/eval/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import importlib
import sys

modules = [
    ("tongbench_eval", "local package; run: python -m pip install -e public/eval"),
    ("openai", "public/eval/requirements.txt"),
    ("dashscope", "public/eval/requirements-api-vlm.txt"),
    ("dotenv", "public/eval/requirements.txt"),
    ("yaml", "public/eval/requirements.txt"),
    ("PIL", "public/eval/requirements.txt"),
    ("mcp", "python -m pip install 'mcp==1.16.0'"),
    ("huggingface_hub", "python -m pip install huggingface_hub; needed to download private Hugging Face files"),
    ("matplotlib", "python -m pip install matplotlib"),
]

missing = []
for module, hint in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, hint, str(exc)))

if missing:
    for module, hint, exc in missing:
        print(f"FAIL: import {module}: {exc}. Install hint: {hint}", file=sys.stderr)
    raise SystemExit(1)
print("OK: required Python modules import successfully")
PY
  if [[ $? -ne 0 ]]; then
    FAILED=1
  fi

  PYTHONPATH="$REPO_ROOT/public/eval/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m tongbench_eval.cli.framework.ready_single_tasks --help >/dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    ok "ready_single_tasks CLI starts"
  else
    fail "ready_single_tasks CLI failed"
  fi

  PYTHONPATH="$REPO_ROOT/public/eval/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m tongbench_eval.cli.framework.dialogue --help >/dev/null 2>&1
  if [[ $? -eq 0 ]]; then
    ok "dialogue CLI starts"
  else
    fail "dialogue CLI failed"
  fi
fi

if [[ "$RUNTIME" == "native" ]]; then
  section "Native runtime"
  native_cli="$REPO_ROOT/public/eval/runtime/native/bin/docker"
  if [[ -x "$native_cli" ]]; then
    ok "native runtime CLI exists: ${native_cli#$REPO_ROOT/}"
  else
    fail "native runtime CLI missing or not executable: $native_cli"
  fi
  if [[ -n "${TONGBENCH_NATIVE_MCP_PYTHON:-}" && ! -x "${TONGBENCH_NATIVE_MCP_PYTHON}" ]]; then
    fail "TONGBENCH_NATIVE_MCP_PYTHON is not executable: $TONGBENCH_NATIVE_MCP_PYTHON"
  fi
  if [[ -n "${TONGBENCH_NATIVE_BIN_DIR:-}" && ! -d "${TONGBENCH_NATIVE_BIN_DIR}" ]]; then
    fail "TONGBENCH_NATIVE_BIN_DIR does not exist: $TONGBENCH_NATIVE_BIN_DIR"
  fi
else
  section "Docker"
  if ! command -v docker >/dev/null 2>&1; then
    fail "docker command not found. Install Docker Engine first."
  else
    docker --version || fail "docker --version failed"
    docker info >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
      ok "current user can access Docker daemon"
    else
      fail "current user cannot access Docker daemon. Check docker group or daemon status."
    fi
  fi

  section "Docker Images"
  images=(
    "${DOCKER_IMAGE:-wildclawbench-ubuntu:v1.3}"
    "${HERMES_DOCKER_IMAGE:-wildclawbench-hermes-agent:v0.5}"
    "${DOCKER_IMAGE_CODEX:-wildclawbench-codex-ubuntu:v0.0}"
    "${DOCKER_IMAGE_CLAUDECODE:-wildclawbench-claudecode-ubuntu:v0.2}"
  )

  if command -v docker >/dev/null 2>&1; then
    for image in "${images[@]}"; do
      docker image inspect "$image" >/dev/null 2>&1
      if [[ $? -eq 0 ]]; then
        ok "Docker image exists: $image"
      else
        fail "Docker image missing: $image. Install the required Docker image before using docker runtime."
      fi
    done
  fi
fi

if [[ "$CHECK_DATA" != "0" ]]; then
  section "For_user Data"
  required_dirs=(
    "$REPO_ROOT/For_user/data/input/merged"
    "$REPO_ROOT/For_user/data/subsets/stage2_20pct"
    "$REPO_ROOT/For_user/data/subsets/stage3_full"
  )
  for path in "${required_dirs[@]}"; do
    if [[ -d "$path" ]]; then
      ok "data directory exists: ${path#$REPO_ROOT/}"
    else
      fail "data directory missing: ${path#$REPO_ROOT/}. Run For_user/scripts/download_input_from_huggingface.sh."
    fi
  done
  if [[ -d "$REPO_ROOT/For_user/data/subsets/stage0_smoke" || -d "$REPO_ROOT/For_user/data/subsets/stage2_20pct" ]]; then
    ok "smoke subset available (stage0_smoke or stage2_20pct fallback)"
  else
    fail "no smoke subset found. Run For_user/scripts/download_input_from_huggingface.sh."
  fi
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
  echo "Environment check passed."
else
  echo "Environment check failed." >&2
fi
exit "$FAILED"
