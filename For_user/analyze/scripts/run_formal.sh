#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${TONGBENCH_NATIVE_BUNDLE_ROOT:-}" && -f "${HOME}/tongbench-native-runtime-install/env.sh" ]]; then
  # Load the native harness bundle owned by the current user.
  # shellcheck disable=SC1091
  source "${HOME}/tongbench-native-runtime-install/env.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-${TONGBENCH_PYTHON:-$(command -v python3)}}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-${PYTHON_BIN}}"
PREPARED_ROOT="${PREPARED_ROOT:-${REPO_ROOT}/For_user/analyze/prepared/formal_eval_qwen38}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/For_user/analyze/output/formal_qwen38_hermes}"

export PYTHONPATH="${REPO_ROOT}/public/eval/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${REPO_ROOT}/For_user/analyze/scripts/prepare_eval_inputs.py" \
  --repo-root "${REPO_ROOT}" \
  --prepared-root "${PREPARED_ROOT}"
"${PYTHON_BIN}" "${REPO_ROOT}/For_user/analyze/scripts/run_analysis_eval.py" \
  --repo-root "${REPO_ROOT}" \
  --prepared-root "${PREPARED_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --python-bin "${EVAL_PYTHON_BIN}" \
  --mode formal \
  --experiment all \
  --job-concurrency "${ANALYSIS_JOB_CONCURRENCY:-6}" \
  --task-parallel-jobs "${ANALYSIS_TASK_PARALLEL_JOBS:-4}" \
  --extra-steps "${ANALYSIS_EXTRA_STEPS:-6}"
