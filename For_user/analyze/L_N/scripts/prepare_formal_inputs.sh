#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="$ROOT/private/taskgen/src:$ROOT/public/eval/src:${PYTHONPATH:-}"
LEARNING_CHECKPOINT_ROOT="${LEARNING_CHECKPOINT_ROOT:-$ROOT/For_user/analyze/shared/main_learning_checkpoints_pending}"
EXTERNAL_ROOT="${EXTERNAL_ROOT:-$(cd "$ROOT/.." && pwd)}"
exec python3 "$ROOT/analyze/L_N/scripts/prepare_l_n_scaleup.py" \
  --repo-root "$ROOT" \
  --external-root "$EXTERNAL_ROOT" \
  --learning-checkpoint-root "$LEARNING_CHECKPOINT_ROOT" \
  --bundle-root "$ROOT/For_user/analyze/L_N/input/formal_v1" \
  --parent-ready "$ROOT/For_user/data/subsets/stage3_full/ready_l4_tasks_with_images.json" \
  --learning-plan "$ROOT/For_user/data/subsets/stage3_full/learning_plan.json" \
  --strict-chain \
  --parent-count 50 \
  --task-name LN_formal_v1 \
  --force
