from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .config import EvalConfig, ModelConfig
from .parsers import EpisodeRecord, parse_libero_log
from .paths import ensure_dir, find_conda, libero_pythonpath, resolve_libero_repo_root


@dataclass
class CommandSpec:
    model_name: str
    task_suite: str
    task_id: int | None
    command: list[str]
    cwd: Path
    env: dict[str, str]
    raw_log: Path


@dataclass
class RunResult:
    model_name: str
    task_id: int | None
    command: list[str]
    cwd: str
    raw_log: str
    returncode: int | None
    runtime_seconds: float | None
    dry_run: bool
    task_filter_supported: bool


def create_run_dir(eval_cfg: EvalConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = eval_cfg.output_root / f"{stamp}-{eval_cfg.task_suite}"
    for child in ["raw_logs", "videos", "metrics", "figures"]:
        ensure_dir(run_dir / child)
    return run_dir


def base_env(model_cfg: ModelConfig, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = libero_pythonpath(model_cfg.libero_path, env.get("PYTHONPATH", ""))
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["WANDB_MODE"] = env.get("WANDB_MODE", "disabled")
    env["GUINSOO_VLA_RUN_DIR"] = str(run_dir)
    return env


def conda_python(model_cfg: ModelConfig) -> list[str]:
    return [
        find_conda(),
        "run",
        "--no-capture-output",
        "-n",
        model_cfg.conda_env,
        "env",
        f"PYTHONPATH={libero_pythonpath(model_cfg.libero_path, os.environ.get('PYTHONPATH', ''))}",
        f"LIBERO_PATH={resolve_libero_repo_root(model_cfg.libero_path)}",
        "python",
    ]


def build_wall_command(model_cfg: ModelConfig, eval_cfg: EvalConfig, run_dir: Path, task_id: int | None) -> CommandSpec:
    raw_log = run_dir / "raw_logs" / model_cfg.name / f"task_{task_id}.txt"
    ensure_dir(raw_log.parent)
    command = conda_python(model_cfg) + [
        str(model_cfg.repo_path / "scripts" / "infer_libero.py"),
        "--checkpoint-path",
        str(model_cfg.model_path),
        "--train-config-path",
        str(model_cfg.train_config_path),
        "--norm-key",
        "libero_all",
        "--task-suite-name",
        eval_cfg.task_suite,
        "--task-indices",
        str(task_id),
        "--num-trials-per-task",
        str(eval_cfg.num_trials_per_task),
        "--seed",
        str(eval_cfg.seed),
        "--num-workers",
        "1",
        "--max-batch-size",
        "1",
        "--log-dir",
        str(run_dir / "videos" / model_cfg.name),
    ]
    env = base_env(model_cfg, run_dir)
    env["LIBERO_PATH"] = str(model_cfg.libero_path)
    return CommandSpec(model_cfg.name, eval_cfg.task_suite, task_id, command, model_cfg.repo_path, env, raw_log)


def build_lingbot_command(model_cfg: ModelConfig, eval_cfg: EvalConfig, run_dir: Path, task_id: int | None) -> CommandSpec:
    raw_log = run_dir / "raw_logs" / model_cfg.name / f"task_{task_id}.txt"
    ensure_dir(raw_log.parent)
    command = conda_python(model_cfg) + [
        str(model_cfg.repo_path / "experiment" / "libero" / "libero" / "run_libero_eval.py"),
        "--model_family",
        "instruct_vla",
        "--pretrained_checkpoint",
        str(model_cfg.model_path),
        "--task_suite_name",
        eval_cfg.task_suite,
        "--task_id",
        str(task_id),
        "--num_trials_per_task",
        str(eval_cfg.num_trials_per_task),
        "--seed",
        str(eval_cfg.seed),
        "--local_log_dir",
        str(run_dir / "raw_logs" / model_cfg.name / "upstream"),
        "--use_wandb",
        "False",
    ]
    return CommandSpec(model_cfg.name, eval_cfg.task_suite, task_id, command, model_cfg.repo_path, base_env(model_cfg, run_dir), raw_log)


def build_openvla_command(model_cfg: ModelConfig, eval_cfg: EvalConfig, run_dir: Path, task_id: int | None) -> CommandSpec:
    raw_log = run_dir / "raw_logs" / model_cfg.name / "suite.txt"
    ensure_dir(raw_log.parent)
    command = conda_python(model_cfg) + [
        str(model_cfg.repo_path / "experiments" / "robot" / "libero" / "run_libero_eval.py"),
        "--model_family",
        "openvla",
        "--pretrained_checkpoint",
        str(model_cfg.model_path),
        "--task_suite_name",
        eval_cfg.task_suite,
        "--num_trials_per_task",
        str(eval_cfg.num_trials_per_task),
        "--seed",
        str(eval_cfg.seed),
        "--center_crop",
        "True",
        "--local_log_dir",
        str(run_dir / "raw_logs" / model_cfg.name / "upstream"),
        "--use_wandb",
        "False",
    ]
    return CommandSpec(model_cfg.name, eval_cfg.task_suite, task_id, command, model_cfg.repo_path, base_env(model_cfg, run_dir), raw_log)


def build_command_specs(model_cfg: ModelConfig, eval_cfg: EvalConfig, run_dir: Path) -> list[CommandSpec]:
    if model_cfg.runner == "wall_oss":
        return [build_wall_command(model_cfg, eval_cfg, run_dir, task_id) for task_id in eval_cfg.task_ids]
    if model_cfg.runner == "lingbot_vla":
        return [build_lingbot_command(model_cfg, eval_cfg, run_dir, task_id) for task_id in eval_cfg.task_ids]
    if model_cfg.runner == "openvla":
        return [build_openvla_command(model_cfg, eval_cfg, run_dir, None)]
    raise ValueError(f"Unknown runner: {model_cfg.runner}")


def run_command(spec: CommandSpec, dry_run: bool, task_filter_supported: bool) -> tuple[RunResult, list[EpisodeRecord]]:
    started = time.perf_counter()
    if dry_run:
        result = RunResult(
            model_name=spec.model_name,
            task_id=spec.task_id,
            command=spec.command,
            cwd=str(spec.cwd),
            raw_log=str(spec.raw_log),
            returncode=None,
            runtime_seconds=None,
            dry_run=True,
            task_filter_supported=task_filter_supported,
        )
        return result, []

    ensure_dir(spec.raw_log.parent)
    with open(spec.raw_log, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(spec.command) + "\n\n")
        log.flush()
        proc = subprocess.run(
            spec.command,
            cwd=spec.cwd,
            env=spec.env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    runtime = time.perf_counter() - started
    text = spec.raw_log.read_text(encoding="utf-8", errors="replace")
    records = parse_libero_log(
        text=text,
        model_name=spec.model_name,
        task_suite=spec.task_suite,
        task_id_hint=spec.task_id,
        runtime_seconds=runtime,
        raw_log=str(spec.raw_log),
        returncode=proc.returncode,
    )
    result = RunResult(
        model_name=spec.model_name,
        task_id=spec.task_id,
        command=spec.command,
        cwd=str(spec.cwd),
        raw_log=str(spec.raw_log),
        returncode=proc.returncode,
        runtime_seconds=runtime,
        dry_run=False,
        task_filter_supported=task_filter_supported,
    )
    return result, records


def write_manifest(
    run_dir: Path,
    eval_cfg: EvalConfig,
    results: list[RunResult],
    model_configs: dict[str, ModelConfig],
    alignment: dict | None = None,
) -> None:
    payload = {
        "run_dir": str(run_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "alignment": alignment,
        "eval_config": {
            "benchmark": eval_cfg.benchmark,
            "task_suite": eval_cfg.task_suite,
            "task_ids": eval_cfg.task_ids,
            "num_trials_per_task": eval_cfg.num_trials_per_task,
            "seed": eval_cfg.seed,
            "save_videos": eval_cfg.save_videos,
            "models": eval_cfg.models,
        },
        "models": {
            name: {
                "display_name": cfg.display_name,
                "conda_env": cfg.conda_env,
                "repo_path": str(cfg.repo_path),
                "model_path": str(cfg.model_path),
                "libero_path": str(cfg.libero_path),
                "runner": cfg.runner,
                "task_filter_supported": cfg.task_filter_supported,
            }
            for name, cfg in model_configs.items()
        },
        "commands": [asdict(result) for result in results],
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
