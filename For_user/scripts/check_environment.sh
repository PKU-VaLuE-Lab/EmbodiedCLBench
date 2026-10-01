#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CHECK_DATA="${CHECK_DATA:-1}"
RUNTIME="${RUNTIME:-${TONGBENCH_RUNTIME:-native}}"
PYTHON_BIN="$(resolve_python_bin 2>/dev/null || true)"

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
    ("dotenv", "public/eval/requirements.txt"),
    ("yaml", "public/eval/requirements.txt"),
    ("PIL", "public/eval/requirements.txt"),
]
optional_modules = [
    ("dashscope", "public/eval/requirements-api-vlm.txt; only needed by legacy API-VLM tools"),
    ("huggingface_hub", "python -m pip install huggingface_hub; needed to download private Hugging Face files"),
    ("matplotlib", "python -m pip install matplotlib; needed only for result plots"),
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
for module, hint in optional_modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"WARN: optional import {module}: {exc}. Install hint: {hint}")
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

section "Task templates"
template_paths=(
  "$REPO_ROOT/For_user/templates/atomic_templates_updated_env_v3.json"
  "$REPO_ROOT/For_user/templates/subtask_templates_compressed_updated_env_v3.json"
)
for template_path in "${template_paths[@]}"; do
  if [[ -f "$template_path" ]]; then
    ok "task template exists: ${template_path#$REPO_ROOT/}"
  else
    fail "task template missing: $template_path"
  fi
done

if [[ "$RUNTIME" == "native" ]]; then
  section "Native runtime"
  native_bundle_root="${TONGBENCH_NATIVE_BUNDLE_ROOT:-}"
  native_bin_dir="${TONGBENCH_NATIVE_BIN_DIR:-}"
  native_mcp_python="${TONGBENCH_NATIVE_MCP_PYTHON:-}"
  if [[ -z "$native_bundle_root" || -z "$native_bin_dir" || -z "$native_mcp_python" ]]; then
    fail "source the native runtime env.sh first (TONGBENCH_NATIVE_BUNDLE_ROOT, TONGBENCH_NATIVE_BIN_DIR, and TONGBENCH_NATIVE_MCP_PYTHON are required)"
  fi
  if [[ "$native_bundle_root" == *'"'* || "$native_bin_dir" == *'"'* || "$native_mcp_python" == *'"'* ]]; then
    fail "native runtime environment contains literal quote characters; regenerate env.sh with the repository download script"
  fi
  if [[ -n "$native_bundle_root" && ! -d "$native_bundle_root" ]]; then
    fail "TONGBENCH_NATIVE_BUNDLE_ROOT does not exist: $native_bundle_root"
  fi
  if [[ -n "$native_bin_dir" && ! -d "$native_bin_dir" ]]; then
    fail "TONGBENCH_NATIVE_BIN_DIR does not exist: $native_bin_dir"
  fi
  if [[ -n "$native_mcp_python" && ! -x "$native_mcp_python" ]]; then
    fail "TONGBENCH_NATIVE_MCP_PYTHON is not executable: $native_mcp_python"
  fi
  if [[ -f "$native_bundle_root/SHA256SUMS" ]]; then
    if (cd "$native_bundle_root" && sha256sum -c SHA256SUMS >/dev/null 2>&1); then
      ok "native runtime checksum passed"
    else
      fail "native runtime checksum failed: $native_bundle_root/SHA256SUMS"
    fi
  else
    fail "native runtime checksum file missing: $native_bundle_root/SHA256SUMS"
  fi
  if [[ -x "$native_mcp_python" ]]; then
    "$native_mcp_python" - <<'PY'
import importlib
from importlib.metadata import version

for name in ("fire", "mcp", "openai", "pydantic"):
    importlib.import_module(name)
parts = tuple(int(part) for part in version("mcp").split(".")[:2])
if parts < (1, 27):
    raise SystemExit(f"mcp=={version('mcp')} is too old; native bundle requires mcp>=1.27")
print(f"native bundled Python imports passed (mcp=={version('mcp')})")
PY
    if [[ $? -eq 0 ]]; then
      ok "native bundled Python dependencies import"
    else
      fail "native bundled Python dependency probe failed"
    fi
  fi
  native_cli="$REPO_ROOT/public/eval/runtime/native/bin/docker"
  if [[ -x "$native_cli" ]]; then
    ok "native runtime CLI exists: ${native_cli#$REPO_ROOT/}"
  else
    fail "native runtime CLI missing or not executable: $native_cli"
  fi
  if [[ -x "$native_bin_dir/hermes" && -x "$native_bin_dir/codex" && -x "$native_bin_dir/bun" && -x "$native_bin_dir/openclaw" ]]; then
    ok "all native harness entrypoints are executable"
  else
    fail "one or more native harness entrypoints are missing or not executable under: $native_bin_dir"
  fi
  if [[ -f "$native_bundle_root/claudecode/app/src/entrypoints/cli.tsx" ]]; then
    ok "Claude Code entrypoint exists in native bundle"
  else
    fail "Claude Code entrypoint missing from native bundle"
  fi
  native_rg_count="$(find "$native_bundle_root/claudecode" -type f -path '*/vendor/*/rg' -perm -111 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$native_rg_count" -gt 0 ]]; then
    ok "Claude Code bundled ripgrep is executable"
  else
    fail "Claude Code bundled ripgrep is missing or not executable"
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
