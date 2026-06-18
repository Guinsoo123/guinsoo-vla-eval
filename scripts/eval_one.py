#!/usr/bin/env python
from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.config import load_eval_config, load_model_configs
from guinsoo_vla_eval.metrics import build_metrics, write_episode_csv
from guinsoo_vla_eval.plots import generate_figures
from guinsoo_vla_eval.report import build_report
from guinsoo_vla_eval.runners import build_command_specs, create_run_dir, run_command, write_manifest
from guinsoo_vla_eval.tasks import align_and_filter


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one configured VLA model.")
    parser.add_argument("model", choices=["wall_oss", "lingbot_vla", "openvla"])
    parser.add_argument("--config", default=str(ROOT / "configs" / "eval.default.yaml"))
    parser.add_argument("--models", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    eval_cfg = load_eval_config(args.config)
    model_cfg = load_model_configs(args.models)[args.model]
    eval_cfg = type(eval_cfg)(
        benchmark=eval_cfg.benchmark,
        task_suite=eval_cfg.task_suite,
        task_ids=eval_cfg.task_ids,
        num_trials_per_task=eval_cfg.num_trials_per_task,
        seed=eval_cfg.seed,
        save_videos=eval_cfg.save_videos,
        models=[args.model],
        output_root=eval_cfg.output_root,
    )
    run_dir = create_run_dir(eval_cfg)
    results = []
    records = []
    for spec in build_command_specs(model_cfg, eval_cfg, run_dir):
        print("\n$ " + " ".join(spec.command))
        result, parsed = run_command(spec, dry_run=args.dry_run, task_filter_supported=model_cfg.task_filter_supported)
        results.append(result)
        records.extend(parsed)
    alignment = None
    if not args.dry_run:
        records, alignment = align_and_filter(records, eval_cfg, {args.model: model_cfg}, run_dir)
    write_manifest(run_dir, eval_cfg, results, {args.model: model_cfg}, alignment=alignment)
    if not args.dry_run:
        write_episode_csv(records, run_dir / "metrics" / "episodes.csv")
        build_metrics(run_dir)
        generate_figures(run_dir)
        build_report(run_dir)
    print(f"Run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
