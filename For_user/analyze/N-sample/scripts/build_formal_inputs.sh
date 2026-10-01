#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec python3 "$ROOT/analyze/N-sample/scripts/build_n_sample_inputs.py" \
  --source-subset-root "$ROOT/For_user/data/subsets/stage3_full" \
  --source-results-root "$ROOT/For_user/data/input/merged" \
  --output-root "$ROOT/For_user/analyze/N-sample/input/formal_child" \
  --height child \
  --target-count 50
