# VLA 评测指标说明

## 指标总览

本工程把原始评测日志分成三层数据：

- episode 级：每一次环境 rollout 的成败和归因。
- task 级：同一模型在同一 LIBERO task 上的汇总。
- model 级：同一模型在本次评测范围内的总体汇总。

## episode 级字段

`model_name` 表示模型标识，取值为 `wall_oss`、`lingbot_vla`、`openvla`。

`task_suite` 表示 LIBERO 任务集合，例如 `libero_10`、`libero_spatial`、`libero_object`、`libero_goal`、`libero_90`。

`task_id` 表示 task suite 内部任务编号。默认评测 `[0, 1, 2]`。

`task_description` 是 LIBERO 给出的自然语言任务描述，适合在报告中解释任务内容。

`episode_index` 是日志解析得到的 episode 序号。

`success` 表示该 episode 是否成功完成任务。它来自上游日志中的 `Success: True` 或 `Success: False`。

`steps` 表示 episode 使用的交互步数。当前三组上游日志不稳定输出该字段，因此 v1 允许为空。

`runtime_seconds` 表示对应评测命令的 wall-clock 耗时。它衡量的是执行成本，不等同于机器人动作步数。

`failure_reason` 表示失败原因。成功 episode 为空；普通未完成任务记为 `not_successful`；异常记为 `exception:*`；命令非零退出记为 `nonzero_exit:*`。

`raw_log` 指向原始日志文件，便于复查。

## task 级字段

`num_episodes` 是该模型在该任务上实际完成或记录到的 episode 数。它用于判断样本量是否足够。

`num_successes` 是成功次数。

`num_failures` 是失败次数，计算公式为：

```text
num_failures = num_episodes - num_successes
```

`success_rate` 是任务成功率，计算公式为：

```text
success_rate = num_successes / num_episodes
```

`mean_steps` 是平均交互步数。若为空，表示上游日志未提供稳定步数，不能据此比较动作效率。

`runtime_seconds` 是该任务对应命令的耗时。对于单任务 runner，它近似代表该模型在该 task 上的评测耗时。

`seconds_per_episode` 是平均单 episode 耗时，计算公式为：

```text
seconds_per_episode = runtime_seconds / num_episodes
```

## model 级字段

model 级指标把本次评测范围内的全部 episode 聚合到模型维度。

`success_rate` 是最重要的横向比较指标。它回答的问题是：在同一组任务和样本量下，这个 VLA 模型完成任务的比例是多少。

`runtime_seconds` 用于估算评测成本。它会受 GPU、仿真渲染、模型大小和上游实现影响，不应直接解释为模型能力。

`seconds_per_episode` 适合规划完整评测耗时。例如默认 15 episodes/model，如果完整评测扩展到数百 episodes，可以用该指标粗略估算时间。

## LIBERO task suite 含义

`libero_spatial` 主要考察空间关系理解与操作，例如把物体移动到指定位置。

`libero_object` 主要考察物体识别与操作泛化。

`libero_goal` 主要考察目标条件下的长程操作。

`libero_10` 是常用的小型任务集合，适合快速比较和报告演示。

`libero_90` 覆盖更多任务，适合更完整的泛化评测，但耗时显著更长。

## 报告解读建议

主图建议使用 `success_rate_by_model.png`，它最直观地展示三组模型的总体成功率。

任务维度建议使用 `success_rate_by_task.png`，它能显示某个模型是否只在特定任务上表现好。

效率维度建议使用 `runtime_by_model.png` 和 `seconds_per_episode`。这些指标用于说明评测成本，不直接等同于模型能力。

默认配置是小样本评测，适合工程验收和制作报告草稿。正式报告应注明任务数、episode 数、随机种子和硬件环境。
