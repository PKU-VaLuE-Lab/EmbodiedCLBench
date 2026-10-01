#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$ROOT/private/taskgen/src:$ROOT/public/eval/src${PYTHONPATH:+:$PYTHONPATH}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "$ROOT/.." && pwd)}"
BUNDLE="$ROOT/For_user/analyze/L_N/input/formal_v1"
ARGS=(--repo-root "$ROOT" --external-root "$EXTERNAL_ROOT" --bundle-root "$BUNDLE" --levels L3,L5,L6 --model "${QWEN_MODEL:-qwen3.7-plus}" --base-url "${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}" --stage1-concurrency "${API_STAGE1_CONCURRENCY:-16}" --stage2-concurrency "${API_STAGE2_CONCURRENCY:-64}" --stage3-concurrency "${API_STAGE3_CONCURRENCY:-16}" --timeout "${API_TIMEOUT_SEC:-3600}" --quota-wait-sec "${API_QUOTA_WAIT_SEC:-300}")
if [[ -n "${QWEN_API_KEY_FILE:-}" ]]; then
  ARGS+=(--api-key-file "$QWEN_API_KEY_FILE")
fi
exec python3 "$ROOT/analyze/L_N/scripts/run_api_scaleup.py" "${ARGS[@]}"
