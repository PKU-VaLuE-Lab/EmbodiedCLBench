#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUNTIME="${RUNTIME:-${TONGBENCH_RUNTIME:-native}}"
exec "$SCRIPT_DIR/scripts/run_stage2_20pct.sh" "$@"
