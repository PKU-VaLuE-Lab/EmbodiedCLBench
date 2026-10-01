# N-sample 经验数量分析

这个实验分析自进化时学习样本数量 `k` 对 hard-task 泛化能力的影响。

实验固定使用同一批 child L4 测试任务，只改变学习阶段提供的 L2 样本数量：

```text
k = 0, 1, 2, 3, 4
```

其中 `k=0` 是 zero-shot baseline；`k=1..4` 分别运行 ICL 和 Skill。所有条件使用：

- Qwen3.7-Plus
- HermesAgent
- child 高度
- 同一批 12 个 child L4
- 同一份 option bank
- `max_steps_extra=6`
- safe-choice、4 个选项、generic invalid feedback
- 不启用 `--mcp-compress-images`

## 目录

```text
N-sample/
  input/subset_10pct_child/
    ready_l2_tasks_with_images.json
    ready_l4_tasks_with_images.json
    learning_plan_k0.json
    learning_plan_k1.json
    learning_plan_k2.json
    learning_plan_k3.json
    learning_plan_k4.json
    sample_selection_manifest.json
    source_manifest.json
  scripts/
    build_n_sample_inputs.py
    run_n_sample.py
    summarize_n_sample.py
  output/
    smoke/
    full/
  reports/
    smoke/
    full/
```

图片和 option bank 不在这个分析目录中复制。输入 JSON 中的图片路径仍由 `For_user/data/input/merged` 解析，option bank 也只读复用 `For_user` 中的文件。这个实验不会修改 `For_user/`。

## 第一步：构造输入

在仓库根目录执行：

```bash
python3 analyze/N-sample/scripts/build_n_sample_inputs.py
```

脚本会从 `For_user/data/subsets/stage2_20pct` 读取数据，并生成 child-only 输入。k=2 保持原主实验的两条学习样本；k=1 使用其中第一条；k=3 和 k=4 在前缀后追加相关且尽量多样的 child L2 样本。

可以用自定义路径重新构造：

```bash
python3 analyze/N-sample/scripts/build_n_sample_inputs.py \
  --source-subset-root For_user/data/subsets/stage2_20pct \
  --source-results-root For_user/data/input/merged \
  --output-root analyze/N-sample/input/subset_10pct_child
```

## 第二步：Smoke

Smoke 默认只跑第一个 child L4，但会覆盖 zero-shot、k=1..4 的 ICL 和 Skill：

```bash
QWEN_API_KEY='YOUR_QWEN_API_KEY' \
python3 analyze/N-sample/scripts/run_n_sample.py --phase smoke
```

也可以通过环境变量指定 OpenAI-compatible endpoint 和模型：

```bash
QWEN_API_KEY='YOUR_QWEN_API_KEY' \
QWEN_BASE_URL='YOUR_BASE_URL' \
QWEN_MODEL='qwen3.7-plus' \
python3 analyze/N-sample/scripts/run_n_sample.py --phase smoke
```

Smoke 只用于检查 pipeline 是否能跑通。脚本不会自动 retry；遇到额度或网络问题时会保留已经完成的结果，并可用 `--skip-existing` 逻辑继续运行。

先做 dry-run 检查命令和路径：

```bash
python3 analyze/N-sample/scripts/run_n_sample.py --phase smoke --dry-run
```

dry-run 只写 `run_manifest_dry_run.json`，不会覆盖真实运行的 `run_manifest.json`。

## 第三步：完整小规模实验

确认 smoke 正常后，运行固定的 12 个 child L4：

```bash
QWEN_API_KEY='YOUR_QWEN_API_KEY' \
python3 analyze/N-sample/scripts/run_n_sample.py \
  --phase full \
  --parallel-jobs 2
```

`--parallel-jobs` 控制同一条件下并行运行的 target dialogue 数量。默认值为 2，用于减少 API 限流风险；需要调整时直接传入新的整数。每个 k 的 ICL 会先完成，随后同 k 的 Skill 复用该 learning checkpoint 并额外生成 skill summary。

可以只运行部分 k 或部分模式：

```bash
QWEN_API_KEY='YOUR_QWEN_API_KEY' \
python3 analyze/N-sample/scripts/run_n_sample.py \
  --phase full \
  --ks 0,2,4 \
  --modes in_context \
  --parallel-jobs 2
```

## 第四步：汇总和画图

完整实验完成后执行：

```bash
python3 analyze/N-sample/scripts/summarize_n_sample.py --phase full
```

Smoke 结果使用：

```bash
python3 analyze/N-sample/scripts/summarize_n_sample.py --phase smoke
```

主要查看：

- `reports/full/n_sample_sr.svg`：SR 随学习样本数变化的曲线
- `reports/full/n_sample_er.svg`：ER 随学习样本数变化的曲线
- `reports/full/marginal_sr.csv`：从 `k` 到 `k+1` 的 SR 边际变化
- `reports/full/n_sample_summary.md`：汇总表和缺失结果说明
- `reports/full/task_metrics.csv`：逐 task 原始指标

图中实线是 ICL，虚线是 Skill。k=0 的 zero-shot 结果作为两条曲线的共同起点。SR 按已经产生 `score.json` 的 task 计算；如果 API 额度或运行错误导致结果缺失，报告会标记为 `incomplete`，不会把缺失结果计为模型失败。
