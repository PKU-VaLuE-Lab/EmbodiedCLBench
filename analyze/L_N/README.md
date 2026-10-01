# L_N: Compositional Complexity Analysis

## Purpose

This directory studies how task complexity and self-evolution gains change when a task contains
3, 4, 5, or 6 L1 actions.

- L3: remove one L1 from an existing L4.
- L4: reuse the already generated and validated parent task; do not regenerate its graph.
- L5: add one L1 in the same room as the parent L4.
- L6: add two L1 actions in the same room as the parent L4.

All Smoke tasks use the child viewpoint in scene 001. The L3/L5/L6 action combinations come from
the real L1 inventory; L3, L5, and L6 remain in the same room as their parent L4. The state count
is an approximate search-budget target, not an exact requirement.

The current Smoke combinations are in `input/design.json`, with selection rationales in
`input/highlevel/highlevel_candidates.json`. The learning stage reuses the same two L2 tasks from
`input/learning_plan.json`; ICL and Skill use the same learning tasks so the learning data stays
constant across complexity conditions.

## Directory Layout

```text
L_N/
  input/
    design.json
    learning_plan.json
    highlevel/                # L1 inventory and composition candidates
    l1_sources/               # real L1 inputs used by Smoke
    taskgen_output/           # composed tasks and graphs
    image_reuse/              # reusable image manifests
    api_pipeline/             # three-stage API input/output
  output/                     # Smoke logs and summaries
  scripts/
    prepare_l_n_smoke.py      # select compositions and export inputs
    run_graph_smoke.py        # call the existing graph search
    run_image_reuse_smoke.py  # package reusable images
    run_api_smoke.py          # call the shared three-stage API pipeline
    run_eval_smoke.py         # run the Hermes/Qwen zero-shot Smoke
```

## Smoke Sequence

Run the following commands from the project root. Replace `REPO` and `EXT` when using another
machine.

```bash
REPO=/path/to/TongBench-dev
EXT=/path/to/TongSim_jie_dev
PY=/path/to/python

cd "$REPO"

$PY analyze/L_N/scripts/prepare_l_n_smoke.py \
  --repo-root "$REPO" \
  --external-root "$EXT" \
  --output-root "$REPO/analyze/L_N" \
  --force

$PY analyze/L_N/scripts/run_graph_smoke.py \
  --repo-root "$REPO" \
  --external-root "$EXT" \
  --bundle-root "$REPO/analyze/L_N" \
  --timeout-sec 1800

$PY analyze/L_N/scripts/run_image_reuse_smoke.py \
  --repo-root "$REPO" \
  --external-root "$EXT" \
  --bundle-root "$REPO/analyze/L_N"
```

The API key is read only from `DASHSCOPE_API_KEY` or `OPENAI_API_KEY`; the scripts do not write it
to input or output. Smoke uses the Qwen OpenAI-compatible endpoint. The model, URL, concurrency,
and output length are passed to the shared pipeline by `run_api_smoke.py`.

## Checkpoints

1. `output/graph_smoke_summary.json`: verifies that L3/L5/L6 were generated, state counts stay
   within the bound, and a shortest path exists.
2. `input/image_reuse/ready_highlevel_v2_tasks_with_images.json`: verifies the image manifest.
   Non-target states without images are excluded; images must not be fabricated to meet a quota.
3. `input/api_pipeline/<task_id>/stage1_task_output.json`: the Stage 1 high-level task description.
4. `stage2_state_outputs/`: the four rewritten options for each retained state.
5. `stage3_batch_outputs/`: final cross-state rewrites. Stage 3 is the longest API stage, so use
   sufficient `--max-tokens` and a long timeout for complex tasks.
6. `stage3_run_summary.json` and `final_option_bank.jsonl`: downstream EVO/eval inputs after all
   three stages succeed.

After the three API stages produce `input/api_pipeline/final_option_bank_all.jsonl`, run the
zero-shot Smoke without a learning stage:

```bash
$PY analyze/L_N/scripts/run_eval_smoke.py \
  --repo-root "$REPO" \
  --bundle-root "$REPO/analyze/L_N" \
  --max-steps-extra 6
```

This Smoke runs one real task for each of L3, L5, and L6. The L4 graph is reused from the bundle;
its API option bank is not regenerated in this run. ICL and Skill require the fixed L2 learning
inputs and their option bank to be available first.

If a stage fails, inspect `output/api_smoke_logs/`, `api_raw/`, and `api_meta/` first. Do not rerun
all stages automatically. The shared pipeline reuses successful outputs by fingerprint; rerun only
the missing stage.

## Formal Run

This directory currently contains only a low-concurrency Smoke; it does not run the full scale.
The formal experiment reuses the same adapter, graph, image-reuse, and API pipeline while expanding
the task set and concurrency. L4 continues to reuse the validated parent task graph, while L3/L5/L6
use the formal search budget. Before a formal run, freeze the task list, learning plan, API model,
and step budget, and save every stage's input/output for traceability.

`analyze/L_N/` is an independent analysis directory. Do not modify `For_user/`, and do not treat
Smoke outputs as formal paper results.
