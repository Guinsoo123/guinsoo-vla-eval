from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Iterable

from .parsers import EpisodeRecord
from .paths import ensure_dir


EPISODE_FIELDS = [
    "model_name",
    "task_suite",
    "task_id",
    "raw_task_id",
    "task_description",
    "episode_index",
    "success",
    "steps",
    "runtime_seconds",
    "failure_reason",
    "raw_log",
]


SUMMARY_FIELDS = [
    "model_name",
    "task_suite",
    "task_id",
    "task_description",
    "num_episodes",
    "num_successes",
    "num_failures",
    "success_rate",
    "mean_steps",
    "runtime_seconds",
    "seconds_per_episode",
]


MODEL_FIELDS = [
    "model_name",
    "num_episodes",
    "num_successes",
    "num_failures",
    "success_rate",
    "mean_steps",
    "runtime_seconds",
    "seconds_per_episode",
]


def write_episode_csv(records: Iterable[EpisodeRecord], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["success"] = int(record.success)
            row["task_id"] = "" if record.task_id is None else record.task_id
            row["raw_task_id"] = "" if record.raw_task_id is None else record.raw_task_id
            row["steps"] = "" if record.steps is None else record.steps
            row["runtime_seconds"] = "" if record.runtime_seconds is None else f"{record.runtime_seconds:.3f}"
            writer.writerow(row)


def read_episode_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _num(value: str) -> float | None:
    if value in ("", "None", None):
        return None
    return float(value)


def _task_sort_key(item: tuple[tuple[str, str, str, str], object]) -> tuple[str, int, str]:
    (model_name, _task_suite, task_id, task_description), _group = item
    try:
        numeric = int(task_id)
    except (ValueError, TypeError):
        numeric = 10**9
    return (model_name, numeric, task_description)


def summarize_tasks(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_name"], row["task_suite"], row["task_id"], row["task_description"])].append(row)

    output: list[dict[str, object]] = []
    for (model_name, task_suite, task_id, task_description), group in sorted(groups.items(), key=_task_sort_key):
        episodes = len(group)
        successes = sum(int(row["success"]) for row in group)
        steps = [_num(row["steps"]) for row in group if _num(row["steps"]) is not None]
        runtime_by_log: dict[str, float] = {}
        for row in group:
            runtime_value = _num(row["runtime_seconds"])
            if runtime_value is not None:
                runtime_by_log[row["raw_log"]] = runtime_value
        runtime = sum(runtime_by_log.values()) if runtime_by_log else None
        output.append(
            {
                "model_name": model_name,
                "task_suite": task_suite,
                "task_id": task_id,
                "task_description": task_description,
                "num_episodes": episodes,
                "num_successes": successes,
                "num_failures": episodes - successes,
                "success_rate": successes / episodes if episodes else 0.0,
                "mean_steps": mean(steps) if steps else "",
                "runtime_seconds": runtime if runtime is not None else "",
                "seconds_per_episode": runtime / episodes if runtime is not None and episodes else "",
            }
        )
    return output


def summarize_models(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["model_name"]].append(row)

    output: list[dict[str, object]] = []
    for model_name, group in sorted(groups.items()):
        episodes = len(group)
        successes = sum(int(row["success"]) for row in group)
        steps = [_num(row["steps"]) for row in group if _num(row["steps"]) is not None]
        runtime_by_log: dict[str, float] = {}
        for row in group:
            runtime_value = _num(row["runtime_seconds"])
            if runtime_value is not None:
                runtime_by_log[row["raw_log"]] = runtime_value
        runtime = sum(runtime_by_log.values()) if runtime_by_log else None
        output.append(
            {
                "model_name": model_name,
                "num_episodes": episodes,
                "num_successes": successes,
                "num_failures": episodes - successes,
                "success_rate": successes / episodes if episodes else 0.0,
                "mean_steps": mean(steps) if steps else "",
                "runtime_seconds": runtime if runtime is not None else "",
                "seconds_per_episode": runtime / episodes if runtime is not None and episodes else "",
            }
        )
    return output


def write_summary_csv(rows: list[dict[str, object]], path: str | Path, fields: list[str]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_metrics(run_dir: str | Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    episodes = read_episode_csv(run_dir / "metrics" / "episodes.csv")
    task_summary = summarize_tasks(episodes)
    model_summary = summarize_models(episodes)
    task_path = run_dir / "metrics" / "task_summary.csv"
    model_path = run_dir / "metrics" / "model_summary.csv"
    write_summary_csv(task_summary, task_path, SUMMARY_FIELDS)
    write_summary_csv(model_summary, model_path, MODEL_FIELDS)
    return task_path, model_path
