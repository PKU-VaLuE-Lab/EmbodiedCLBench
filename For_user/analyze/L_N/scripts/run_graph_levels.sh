#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "$ROOT/.." && pwd)}"
BUNDLE="$ROOT/For_user/analyze/L_N/input/formal_v1"
COMMON=(--repo-root "$ROOT" --external-root "$EXTERNAL_ROOT" --bundle-root "$BUNDLE" --max-workers "${GRAPH_MAX_WORKERS:-2}" --timeout-sec "${GRAPH_TIMEOUT_SEC:-7200}")
for LEVEL in L3 L5 L6; do
  python3 "$ROOT/analyze/L_N/scripts/run_graph_scaleup.py" "${COMMON[@]}" --levels "$LEVEL"
done
python3 "$ROOT/analyze/L_N/scripts/run_image_reuse_scaleup.py" \
  --repo-root "$ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --bundle-root "$BUNDLE" \
  --task-name LN_formal_v1 \
  --force
