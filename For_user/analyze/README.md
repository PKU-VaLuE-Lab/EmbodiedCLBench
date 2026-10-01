# Analysis Evaluation Handoff

This directory contains analysis experiments outside the main paper experiment. The current
version only evaluates prepared inputs; it does not regenerate tasks, graphs, images, or options.

## Experiments

- `N-sample`: fixed child L4 test tasks comparing nested L2 learning sets. `k=0` is the main
  zero-shot baseline, and `k=2` reuses the main experiment result.
- `L_N`: compares L3/L4/L5/L6 task complexity. L4 and the learning stages reuse existing main
  experiment results; only L3/L5/L6 are newly evaluated here.

All prepared inputs are stored under the corresponding `prepared/` directory. Task generation and
the three-stage API generation pipeline are not part of this evaluation flow.

## Reuse Rules

- Learning tasks use the independent L2 zero-shot trajectories and checkpoints completed in the
  main experiment.
- N-sample learning sets are assembled from existing L2 checkpoints and do not rerun L2. For a
  given L4, the `k+1` set strictly contains the `k` set and adds one L2 task.
- `L_N` uses the parent L4 learning archive from the main experiment and does not rerun learning.
- ICL and Skill use the same selected learning tasks; Skill adds only the required skill-summary
  call.

## Environment

Use a Python environment with the TongBench evaluation dependencies installed and set `PYTHON_BIN`
or `TONGBENCH_PYTHON`. If the native harness bundle is installed, the scripts load:

```text
$HOME/tongbench-native-runtime-install/env.sh
```

Pass API keys through environment variables only:

```bash
export QWEN_API_KEY='your_qwen_api_key'
```

The prepared evaluation uses Hermes, Qwen3.8 Flash, medium thinking, native runtime, and
`task_compact`. `--mcp-compress-images` is disabled.

## Smoke

Smoke runs one task per job, covering the selected N-sample ICL/Skill conditions and the L_N
L3/L5/L6 zero-shot, ICL, and Skill conditions:

```bash
PYTHON_BIN=/path/to/eval/python \
QWEN_API_KEY='your_qwen_api_key' \
bash For_user/analyze/scripts/run_smoke.sh
```

The default output is `For_user/analyze/output/smoke_qwen38_hermes_actual/`. Set `OUTPUT_ROOT` to
change it, and `ANALYSIS_JOB_CONCURRENCY` or `ANALYSIS_TASK_PARALLEL_JOBS` to change concurrency.
Run the formal version only after every Smoke job returns zero and has a score and trajectory.

## Formal Evaluation

```bash
PYTHON_BIN=/path/to/eval/python \
QWEN_API_KEY='your_qwen_api_key' \
bash For_user/analyze/scripts/run_formal.sh
```

Formal output defaults to `For_user/analyze/output/formal_qwen38_hermes/`. Use
`ANALYSIS_JOB_CONCURRENCY`, `ANALYSIS_TASK_PARALLEL_JOBS`, `PREPARED_ROOT`, `OUTPUT_ROOT`,
`EVAL_PYTHON_BIN`, and `ANALYSIS_EXTRA_STEPS` to adjust the run.

Do not treat quota or rate-limit failures as experiment results. Record the failed job, wait for
quota recovery, and rerun only that job. Each job stores its manifest, scheduler log, summary, and
per-task trajectory/score. The top-level manifest records the actual setting, input path, model,
and budget.

Use the analysis summarizers after a run. `reports/completion_audit.json` is the completeness gate,
and `reports/extra3.csv` contains the paper's three-extra-step results.

This directory only prepares existing inputs, reuses checkpoints, and calls the evaluation CLI. It
does not modify `For_user/data/` or regenerate task data.
