from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, expand_path


@dataclass(frozen=True)
class EvalConfig:
    benchmark: str
    task_suite: str
    task_ids: list[int]
    num_trials_per_task: int
    seed: int
    save_videos: bool
    models: list[str]
    output_root: Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    display_name: str
    conda_env: str
    repo_path: Path
    model_path: Path
    libero_path: Path
    runner: str
    task_filter_supported: bool
    train_config_path: Path | None = None


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_eval_config(path: str | Path) -> EvalConfig:
    raw = load_yaml(path)
    output_root = Path(raw.get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return EvalConfig(
        benchmark=str(raw.get("benchmark", "libero")),
        task_suite=str(raw.get("task_suite", "libero_10")),
        task_ids=[int(x) for x in raw.get("task_ids", [0, 1, 2])],
        num_trials_per_task=int(raw.get("num_trials_per_task", 5)),
        seed=int(raw.get("seed", 42)),
        save_videos=bool(raw.get("save_videos", True)),
        models=[str(x) for x in raw.get("models", [])],
        output_root=output_root.resolve(),
    )


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    raw = load_yaml(path)
    configs: dict[str, ModelConfig] = {}
    for name, item in raw.items():
        if not isinstance(item, dict):
            raise ValueError(f"Model config for {name} must be a mapping")
        configs[name] = ModelConfig(
            name=name,
            display_name=str(item.get("display_name", name)),
            conda_env=str(item["conda_env"]),
            repo_path=expand_path(item["repo_path"]),
            model_path=expand_path(item["model_path"]),
            libero_path=expand_path(item["libero_path"]),
            runner=str(item["runner"]),
            task_filter_supported=bool(item.get("task_filter_supported", False)),
            train_config_path=expand_path(item["train_config_path"])
            if item.get("train_config_path")
            else None,
        )
    return configs
