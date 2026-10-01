#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$REPO_ROOT/For_user/runtime_download}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local/share/tongbench/harnesses}"
HF_REPO_ID="${HF_REPO_ID:-PKU-VaLuE-Lab/EmbodiedCLBench-Native-Harnesses}"
ARCHIVE_NAME="${ARCHIVE_NAME:-native_harness_bundle.tar.gz}"

"$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/huggingface_sync.py" \
  --repo-id "$HF_REPO_ID" \
  --local-dir "$DOWNLOAD_ROOT" \
  --allow-pattern "runtime/$ARCHIVE_NAME" \
  --allow-pattern "runtime/SHA256SUMS.txt" \
  --max-workers "${HF_MAX_WORKERS:-4}"

ARCHIVE="$DOWNLOAD_ROOT/runtime/$ARCHIVE_NAME"
CHECKSUMS="$DOWNLOAD_ROOT/runtime/SHA256SUMS.txt"
[[ -f "$ARCHIVE" ]] || { echo "Missing downloaded bundle: $ARCHIVE" >&2; exit 1; }
[[ -f "$CHECKSUMS" ]] || { echo "Missing checksum file: $CHECKSUMS" >&2; exit 1; }

expected_sha256="$(awk -v archive="$ARCHIVE_NAME" '$2 == archive || $2 ~ ("/" archive "$") { print $1; exit }' "$CHECKSUMS")"
[[ "$expected_sha256" =~ ^[[:xdigit:]]{64}$ ]] || {
  echo "No SHA256 entry found for $ARCHIVE_NAME in $CHECKSUMS" >&2
  exit 1
}
actual_sha256="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
[[ "$actual_sha256" == "$expected_sha256" ]] || {
  echo "SHA256 mismatch for $ARCHIVE" >&2
  echo "  expected: $expected_sha256" >&2
  echo "  actual:   $actual_sha256" >&2
  exit 1
}
echo "SHA256 verified: $ARCHIVE"

bash "$REPO_ROOT/For_user/scripts/install_native_harnesses.sh" \
  --bundle "$ARCHIVE" \
  --prefix "$INSTALL_PREFIX"

cat > "$INSTALL_PREFIX/env.sh" <<EOF
export TONGBENCH_NATIVE_BUNDLE_ROOT=$(printf '%q' "$INSTALL_PREFIX")
export TONGBENCH_NATIVE_BIN_DIR="\$TONGBENCH_NATIVE_BUNDLE_ROOT/bin"
export TONGBENCH_NATIVE_MCP_PYTHON="\$TONGBENCH_NATIVE_BUNDLE_ROOT/hermes/.venv/bin/python3"
export TONGBENCH_RUNTIME=native
EOF
chmod a+r "$INSTALL_PREFIX/env.sh"

echo
echo "Runtime files installed: $INSTALL_PREFIX"
echo "Before running TongBench, execute:"
echo "  source \"$INSTALL_PREFIX/env.sh\""
