#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.config import load_eval_config, load_model_configs
from guinsoo_vla_eval.metrics import build_metrics, write_episode_csv
from guinsoo_vla_eval.plots import generate_figures
from guinsoo_vla_eval.report import build_report
from guinsoo_vla_eval.runners import build_command_specs, create_run_dir, run_command, write_manifest
from guinsoo_vla_eval.tasks import align_and_filter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all configured VLA evaluations.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.default.yaml"))
    parser.add_argument("--models", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifest without running simulation.")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    eval_cfg = load_eval_config(args.config)
    all_models = load_model_configs(args.models)
    selected = {name: all_models[name] for name in eval_cfg.models}
    run_dir = create_run_dir(eval_cfg)
    print(f"Run dir: {run_dir}")

    results = []
    records = []
    for name in eval_cfg.models:
        model_cfg = selected[name]
        specs = build_command_specs(model_cfg, eval_cfg, run_dir)
        if not model_cfg.task_filter_supported:
            print(f"[WARN] {name} does not expose native task-id filtering; wrapper will parse selected tasks from suite logs.")
        for spec in specs:
            print("\n$ " + " ".join(spec.command))
            result, parsed = run_command(spec, dry_run=args.dry_run, task_filter_supported=model_cfg.task_filter_supported)
            results.append(result)
            records.extend(parsed)
            if result.returncode not in (0, None):
                print(f"[WARN] {name} task={spec.task_id} exited with {result.returncode}; continuing.")

    alignment = None
    if not args.dry_run:
        records, alignment = align_and_filter(records, eval_cfg, selected, run_dir)
        if alignment["aligned"]:
            print("\n[OK] task alignment verified: all models cover the same tasks with equal episode counts.")
        else:
            print("\n[WARN] task alignment issues detected:")
            for issue in alignment["issues"]:
                print(f"       - {issue}")

    write_manifest(run_dir, eval_cfg, results, selected, alignment=alignment)
    if args.dry_run:
        print(f"\nDry run complete. Manifest: {run_dir / 'manifest.json'}")
        return 0

    episode_path = run_dir / "metrics" / "episodes.csv"
    write_episode_csv(records, episode_path)
    build_metrics(run_dir)
    if not args.skip_figures:
        generate_figures(run_dir)
    build_report(run_dir)
    print(f"\nEvaluation complete: {run_dir}")
    print(f"Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
