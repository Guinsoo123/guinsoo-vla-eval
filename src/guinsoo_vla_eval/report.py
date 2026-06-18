from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def alignment_section(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return "_No manifest found._"
    alignment = json.loads(manifest_path.read_text(encoding="utf-8")).get("alignment")
    if not alignment:
        return "_未执行对齐校验（dry-run 或单步解析）。_"

    source = "LIBERO 规范任务清单" if alignment.get("used_canonical_libero") else "日志描述派生（LIBERO 不可用，回退方案）"
    lines = [
        f"- 任务身份来源：{source}",
        f"- 期望任务 ids：{alignment.get('expected_task_ids')}",
        f"- 每任务期望 episode 数：{alignment.get('num_trials_per_task')}",
    ]
    if alignment.get("aligned"):
        lines.append("- 对齐结果：**通过** —— 三模型覆盖同一任务集且 episode 数一致，横向对比有效。")
    else:
        lines.append("- 对齐结果：**未通过** —— 以下问题会导致对比不公平：")
        lines.extend(f"  - {issue}" for issue in alignment.get("issues", []))
    if alignment.get("unmatched_descriptions"):
        lines.append("- 未匹配到规范任务的描述：")
        lines.extend(f"  - {desc}" for desc in alignment["unmatched_descriptions"])
    return "\n".join(lines)


def markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    model_rows = read_csv(run_dir / "metrics" / "model_summary.csv")
    task_rows = read_csv(run_dir / "metrics" / "task_summary.csv")
    content = f"""# VLA Evaluation Report

生成时间：{datetime.now().isoformat(timespec="seconds")}

运行目录：`{run_dir}`

## 任务对齐校验

{alignment_section(run_dir)}

## 模型总览

{markdown_table(model_rows, ["model_name", "num_episodes", "num_successes", "num_failures", "success_rate", "runtime_seconds", "seconds_per_episode"])}

## 任务明细

{markdown_table(task_rows, ["model_name", "task_id", "task_description", "num_episodes", "num_successes", "success_rate", "runtime_seconds"])}

## 图表

![Success rate by model](figures/success_rate_by_model.png)

![Success rate by task](figures/success_rate_by_task.png)

![Mean steps by model](figures/mean_steps_by_model.png)

![Runtime by model](figures/runtime_by_model.png)

![Success and failure counts](figures/success_failure_stack.png)

## 指标解释

- `success_rate`：成功 episode 数除以总 episode 数，是闭环仿真评测的核心指标。
- `runtime_seconds`：对应模型评测命令的 wall-clock 耗时。
- `seconds_per_episode`：平均每个 episode 的评测耗时，用于估算完整评测成本。
- `mean_steps`：平均交互步数。当前上游日志未稳定输出每个 episode 步数时，该字段为空。
- `failure_reason`：失败归因见 `metrics/episodes.csv`，主要区分未成功、异常和非零退出。

## 说明

默认配置是报告友好小样本，适合快速制作横向比较素材；若用于正式结论，请增加任务数和每任务 episode 数。
"""
    path = run_dir / "report.md"
    path.write_text(content, encoding="utf-8")
    return path
