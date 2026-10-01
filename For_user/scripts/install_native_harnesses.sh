#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  install_native_harnesses.sh --bundle BUNDLE.tar.gz [--prefix DIR]
  install_native_harnesses.sh --bundle-dir DIR [--prefix DIR]

The installation is entirely user-space. It does not require sudo or Docker.
EOF
}

die() {
  echo "install_native_harnesses: $*" >&2
  exit 1
}

BUNDLE_ARCHIVE=""
BUNDLE_DIR=""
PREFIX="${HOME}/.local/share/tongbench/harnesses"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle)
      [[ $# -ge 2 ]] || die "--bundle needs a path"
      BUNDLE_ARCHIVE="$2"
      shift 2
      ;;
    --bundle-dir)
      [[ $# -ge 2 ]] || die "--bundle-dir needs a path"
      BUNDLE_DIR="$2"
      shift 2
      ;;
    --prefix)
      [[ $# -ge 2 ]] || die "--prefix needs a path"
      PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "$BUNDLE_ARCHIVE" || -n "$BUNDLE_DIR" ]] || { usage; exit 2; }
[[ -z "$BUNDLE_ARCHIVE" || -z "$BUNDLE_DIR" ]] || die "use only one of --bundle and --bundle-dir"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

TMP_DIR=""
INSTALL_STAGE=""
cleanup() {
  if [[ -n "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
  if [[ -n "$INSTALL_STAGE" ]]; then
    rm -rf "$INSTALL_STAGE"
  fi
}
trap cleanup EXIT

if [[ -n "$BUNDLE_ARCHIVE" ]]; then
  [[ -f "$BUNDLE_ARCHIVE" ]] || die "bundle archive not found: $BUNDLE_ARCHIVE"
  TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tongbench-native-install.XXXXXX")"
  tar -xzf "$BUNDLE_ARCHIVE" -C "$TMP_DIR"
  # Archives may contain one top-level directory; locate the manifest instead
  # of assuming that extraction itself is the bundle root.
  if [[ -f "$TMP_DIR/native_harness_manifest.json" ]]; then
    BUNDLE_DIR="$TMP_DIR"
  else
    MANIFEST_PATH="$(find "$TMP_DIR" -type f -name native_harness_manifest.json -print -quit)"
    [[ -n "$MANIFEST_PATH" ]] || die "invalid bundle archive: manifest missing"
    BUNDLE_DIR="$(dirname "$MANIFEST_PATH")"
  fi
fi

BUNDLE_DIR="$(cd "$BUNDLE_DIR" && pwd)"
[[ -f "$BUNDLE_DIR/native_harness_manifest.json" ]] || die "invalid bundle: manifest missing"
[[ "$(uname -m)" == "x86_64" ]] || die "this bundle currently supports Linux x86_64 only"

PARENT_DIR="$(dirname "$PREFIX")"
mkdir -p "$PARENT_DIR"
if [[ -e "$PREFIX" && -n "$(find "$PREFIX" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  die "install prefix is not empty: $PREFIX"
fi

# Build and validate beside the final directory.  The final prefix is only
# published after every file and runtime probe succeeds, so an interrupted
# copy cannot leave a plausible-looking partial installation behind.
if [[ -d "$PREFIX" ]]; then
  rmdir "$PREFIX" || die "could not remove empty install prefix: $PREFIX"
fi
INSTALL_STAGE="$(mktemp -d "$PARENT_DIR/.tongbench-native-install.XXXXXX")"
cp -a "$BUNDLE_DIR/." "$INSTALL_STAGE/"
chmod -R a+rX "$INSTALL_STAGE"
chmod a+rx "$INSTALL_STAGE/bin"/*
find "$INSTALL_STAGE/claudecode" -type f -path '*/vendor/*/rg' -exec chmod a+rx {} + 2>/dev/null || true

for entrypoint in hermes codex bun openclaw; do
  [[ -x "$INSTALL_STAGE/bin/$entrypoint" ]] || die "bundle entrypoint missing: bin/$entrypoint"
done
[[ -x "$INSTALL_STAGE/hermes/.venv/bin/python3" ]] || die "Hermes bundled Python entrypoint missing"
[[ -x "$INSTALL_STAGE/hermes_python/bin/python3.12" ]] || die "Hermes bundled interpreter missing"
[[ -f "$INSTALL_STAGE/claudecode/app/src/entrypoints/cli.tsx" ]] || die "Claude Code entrypoint missing"

if [[ -f "$INSTALL_STAGE/SHA256SUMS" ]]; then
  (cd "$INSTALL_STAGE" && sha256sum -c SHA256SUMS >/dev/null)
fi

"$INSTALL_STAGE/hermes/.venv/bin/python3" - <<'PY'
import importlib
from importlib.metadata import version

for name in ("fire", "mcp", "openai", "pydantic"):
    importlib.import_module(name)
mcp_version = tuple(int(part) for part in version("mcp").split(".")[:2])
if mcp_version < (1, 27):
    raise SystemExit(f"bundled mcp is too old: {version('mcp')} (need >= 1.27)")
print("bundled Python runtime imports passed; mcp=" + version("mcp"))
PY

mv "$INSTALL_STAGE" "$PREFIX"
INSTALL_STAGE=""

echo "Native harness bundle installed: $PREFIX"
echo
echo "Before running TongBench:"
echo "  export TONGBENCH_NATIVE_BUNDLE_ROOT=$PREFIX"
echo "  export TONGBENCH_NATIVE_BIN_DIR=$PREFIX/bin"
echo "  export TONGBENCH_RUNTIME=native"
echo
echo "CLI checks:"
echo "  $PREFIX/bin/hermes --help"
echo "  $PREFIX/bin/codex --help"
echo "  $PREFIX/bin/bun --version"
echo "  $PREFIX/bin/openclaw --help"
