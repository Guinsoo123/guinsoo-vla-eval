# Guinsoo VLA Eval 设计文档

## 评测目标

本工程的目标是在同一台机器、同一组 LIBERO 仿真任务、同一统计口径下，对 WALL-OSS、LingBot-VLA、OpenVLA 三组开源 VLA 模型进行横向比较。输出结果要便于制作评测报告，而不是只停留在上游脚本的原始日志。

默认评测规模是 `libero_10` 前 3 个任务、每任务 5 个 episode。这个规模用于快速出图和验证评测链路；如果要形成严肃结论，应把任务数和 episode 数调大。

## 评测原则

- 任务一致：三组模型都使用 LIBERO benchmark。
- 配置一致：默认使用相同 task suite、task ids、episode 数和随机种子。
- 入口克制：不修改三组上游模型仓库源码，只通过 subprocess 调用官方评测入口。
- 结果标准化：把不同上游脚本的日志统一解析为 `episodes.csv`、`task_summary.csv`、`model_summary.csv`。
- 失败隔离：单个模型或任务失败时，继续执行后续模型，并在 `manifest.json` 和 raw log 中保留失败信息。

## 系统架构

配置层由 `configs/eval.default.yaml` 和 `configs/models.yaml` 组成。前者描述本次评测的 benchmark、任务范围、episode 数和输出目录；后者描述模型路径、conda 环境、上游仓库路径和 runner 类型。

Runner 层位于 `src/guinsoo_vla_eval/runners.py`。它负责为每个模型生成真实命令，设置必要的环境变量，调用 conda 环境，保存 stdout/stderr，并记录运行耗时。

解析层位于 `src/guinsoo_vla_eval/parsers.py`。它从上游日志中提取 `Task:`、`Success:`、异常信息和退出码，转换为 episode 级记录。

指标层位于 `src/guinsoo_vla_eval/metrics.py`。它把 episode 记录聚合成任务级和模型级指标。

可视化层位于 `src/guinsoo_vla_eval/plots.py`。它从 CSV 读取指标，生成静态 PNG 图表。

报告层位于 `src/guinsoo_vla_eval/report.py`。它把模型总览、任务明细、图表和指标说明写入 `report.md`。

## 三模型适配策略

WALL-OSS 使用 `/home/qj00433/.wall-oss/repos/wall-x/scripts/infer_libero.py`。该入口支持 `--task-indices`，因此可以按配置中的 task id 逐个执行。

LingBot-VLA 使用 `/home/qj00433/.lingbot-vla/repos/lingbot-vla/experiment/libero/libero/run_libero_eval.py`。该入口支持 `--task_id`，因此也可以按 task id 逐个执行。

OpenVLA 使用 `/home/qj00433/.openvla/repos/openvla/experiments/robot/libero/run_libero_eval.py`。官方入口当前没有 `--task_id` 参数，因此 v1 不修改上游源码，而是运行 suite 命令并从日志中解析任务结果。`manifest.json` 会记录 `task_filter_supported=false`。

## 数据流

```text
YAML 配置
  -> runner 生成上游命令
  -> conda run 调用模型评测脚本
  -> raw_logs/<model>/*.txt
  -> parsers 提取 episode 级记录
  -> metrics/*.csv
  -> figures/*.png
  -> report.md
```

## 输出目录

每次运行会创建一个新的 run 目录：

```text
outputs/<timestamp>-<task_suite>/
  manifest.json
  raw_logs/
  videos/
  metrics/
  figures/
  report.md
```

`manifest.json` 是可复现实验的索引文件，包含评测配置、模型路径、命令、工作目录、raw log 路径、退出码和运行耗时。

## 失败处理

runner 不会因为单个命令失败而中断整次评测。失败命令的 stdout/stderr 会写入 raw log，退出码写入 manifest。解析层会尽量从日志中提取 episode 结果；如果模型在 episode 前就失败，会生成一条 `run_failed_before_episode` 记录，`failure_reason` 标记为 `nonzero_exit:<code>` 或 `exception:*`。

## 已知限制

当前上游日志不稳定输出每个 episode 的真实步数，因此 `mean_steps` 在 v1 中可能为空。这个字段已经保留在 schema 中，后续可以通过本地 adapter 或上游日志增强补齐。

OpenVLA 的 task 过滤不是官方参数，v1 采用非侵入式解析策略。若后续需要严格只运行指定 task id，可以增加本工程内的 OpenVLA adapter，但仍不建议直接改上游仓库文件。
