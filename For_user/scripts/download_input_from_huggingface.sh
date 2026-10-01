#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="$(resolve_python_bin)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/For_user/data}"
HF_REPO_ID="${HF_REPO_ID:-PKU-VaLuE-Lab/EmbodiedCLBench}"
ARCHIVE_NAME="${ARCHIVE_NAME:-tongbench_l2_l4_scaleup_v1_input_subsets.tar.gz}"

"$PYTHON_BIN" "$REPO_ROOT/For_user/scripts/huggingface_sync.py" \
  --repo-id "$HF_REPO_ID" \
  --local-dir "$DATA_ROOT" \
  --allow-pattern "archives/$ARCHIVE_NAME" \
  --allow-pattern "archives/SHA256SUMS.txt" \
  --max-workers "${HF_MAX_WORKERS:-4}"

(
  cd "$DATA_ROOT/archives"
  sha256sum -c SHA256SUMS.txt
)

tar -xzf "$DATA_ROOT/archives/$ARCHIVE_NAME" -C "$DATA_ROOT"
echo "TongBench input data is ready under $DATA_ROOT/input and $DATA_ROOT/subsets."
