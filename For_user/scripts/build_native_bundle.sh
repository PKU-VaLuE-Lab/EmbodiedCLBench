#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible name. Keep one bundle format and one implementation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/build_native_harness_bundle.sh" "$@"
