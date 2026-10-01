#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec python3 "$ROOT/analyze/N-sample/scripts/run_n_sample.py" \
  --phase formal \
  --project-root "$ROOT" \
  --input-root "$ROOT/For_user/analyze/N-sample/input/formal_child" \
  --source-results-root "$ROOT/For_user/data/input/merged" \
  --output-root "$ROOT/For_user/analyze/N-sample/output" \
  --ks 1,2,3,4 \
  --modes in_context,skill \
  --parallel-jobs "${N_SAMPLE_PARALLEL_JOBS:-2}" \
  --extra-steps "${EXTRA_STEPS:-6}"
