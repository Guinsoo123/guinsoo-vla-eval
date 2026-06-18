# Guinsoo VLA Eval

本工程用于在本机统一评测三组已下载的开源 VLA 模型：

- WALL-OSS：`/home/qj00433/.wall-oss/models/wall-oss-0.5`
- LingBot-VLA：`/home/qj00433/.lingbot-vla/models/lingbot-vla-4b`
- OpenVLA：`/home/qj00433/.openvla/models/openvla-7b`

默认评测采用 LIBERO 闭环仿真，跑 `libero_10` 的前 3 个任务，每任务 5 个 episode。输出包含 CSV/JSON 指标、PNG 图表和 Markdown 报告，适合快速制作横向评测材料。

## 快速开始

先检查环境和资源：

```bash
cd /home/qj00433/guinsoo-vla-eval
conda run -n openvla-eval python scripts/doctor.py
```

先做 dry-run，确认三组模型会执行哪些命令：

```bash
conda run -n openvla-eval python scripts/eval_all.py --config configs/eval.default.yaml --dry-run
```

开始真实评测：

```bash
conda run -n openvla-eval python scripts/eval_all.py --config configs/eval.default.yaml
```

评测完成后查看：

```text
outputs/<run_id>/
  manifest.json
  raw_logs/
  metrics/episodes.csv
  metrics/task_summary.csv
  metrics/model_summary.csv
  figures/*.png
  report.md
```

如果只想验证图表和报告链路，不启动仿真：

```bash
conda run -n openvla-eval python scripts/parse_logs.py --sample
conda run -n openvla-eval python scripts/visualize.py --sample
conda run -n openvla-eval python scripts/make_report.py --run-dir outputs/sample
```

## 常用命令

单模型 dry-run：

```bash
conda run -n openvla-eval python scripts/eval_one.py wall_oss --dry-run
conda run -n openvla-eval python scripts/eval_one.py lingbot_vla --dry-run
conda run -n openvla-eval python scripts/eval_one.py openvla --dry-run
```

重新生成图表：

```bash
conda run -n openvla-eval python scripts/visualize.py --run-dir outputs/<run_id>
```

重新生成报告：

```bash
conda run -n openvla-eval python scripts/make_report.py --run-dir outputs/<run_id>
```

## 配置

默认评测配置在 `configs/eval.default.yaml`：

- `task_suite`：LIBERO task suite，默认 `libero_10`。
- `task_ids`：默认 `[0, 1, 2]`。
- `num_trials_per_task`：每个任务 episode 数，默认 `5`。
- `models`：默认评测 `wall_oss`、`lingbot_vla`、`openvla`。
- `output_root`：输出目录，默认 `outputs`。

模型路径和上游入口在 `configs/models.yaml`。本工程只读取这些资源，不移动、不覆盖模型文件。

## 指标文件

- `metrics/episodes.csv`：episode 级明细。
- `metrics/task_summary.csv`：任务级汇总。
- `metrics/model_summary.csv`：模型级汇总。

核心指标：

- `success_rate`：成功 episode 数 / 总 episode 数。
- `runtime_seconds`：评测命令耗时。
- `seconds_per_episode`：平均单 episode 耗时。
- `mean_steps`：平均交互步数。当前上游日志不稳定输出步数时，该字段允许为空。
- `failure_reason`：失败原因，分为 `not_successful`、`exception:*`、`nonzero_exit:*`。

更详细的指标解释见 `docs/metrics_guide.md`。

## 注意事项

- OpenVLA 官方脚本当前没有原生 `--task_id` 参数；本工程会在 manifest 中标注 `task_filter_supported=false`，并从 suite 日志中解析任务结果。
- OpenVLA/LIBERO import 时可能打印 TensorFlow、cuDNN、protobuf 相关 warning；只要命令退出码为 0 且有结果日志，通常不影响评测。
- 完整 LIBERO 评测耗时很长。默认配置是小样本报告配置，不应直接当作排行榜结论。
