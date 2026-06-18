#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.config import load_eval_config, load_model_configs
from guinsoo_vla_eval.paths import find_conda


def check_path(label: str, path: Path) -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local VLA eval resources.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.default.yaml"))
    parser.add_argument("--models", default=str(ROOT / "configs" / "models.yaml"))
    args = parser.parse_args()

    eval_cfg = load_eval_config(args.config)
    models = load_model_configs(args.models)
    ok = True

    print(f"Project root: {ROOT}")
    print(f"Benchmark: {eval_cfg.benchmark}/{eval_cfg.task_suite}, task_ids={eval_cfg.task_ids}")
    print(f"Conda: {find_conda()}")

    try:
        envs = subprocess.check_output([find_conda(), "env", "list"], text=True)
    except Exception as exc:
        envs = ""
        ok = False
        print(f"[ERROR] failed to list conda envs: {exc}")

    for name in eval_cfg.models:
        cfg = models[name]
        print(f"\n== {name} ({cfg.display_name}) ==")
        ok = check_path("repo", cfg.repo_path) and ok
        ok = check_path("model", cfg.model_path) and ok
        ok = check_path("LIBERO", cfg.libero_path) and ok
        if cfg.train_config_path:
            ok = check_path("train_config", cfg.train_config_path) and ok
        env_ok = cfg.conda_env in envs
        print(f"[{'OK' if env_ok else 'MISSING'}] conda env: {cfg.conda_env}")
        ok = env_ok and ok

    for pkg in ["yaml", "matplotlib"]:
        pkg_ok = importlib.util.find_spec(pkg) is not None
        print(f"[{'OK' if pkg_ok else 'MISSING'}] python package in current env: {pkg}")
        ok = pkg_ok and ok

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
