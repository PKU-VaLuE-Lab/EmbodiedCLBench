#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="$(resolve_python_bin)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/For_user/output}"
REPORT_ROOT="${REPORT_ROOT:-$OUTPUT_ROOT/reports}"

"$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/summarize_l2_l4_results.py" \
  --output-root "$OUTPUT_ROOT" \
  --report-root "$REPORT_ROOT" \
  --extra-steps "${EXTRA_STEPS:-3}" \
  --stages "${STAGES:-stage0_smoke,stage2_20pct,stage3_full}"
