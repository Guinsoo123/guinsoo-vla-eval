#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.metrics import build_metrics, write_episode_csv
from guinsoo_vla_eval.parsers import parse_libero_log
from guinsoo_vla_eval.paths import ensure_dir


SAMPLE_LOGS = {
    "wall_oss": """Task: put the bowl on the plate\nSuccess: True\n# episodes completed so far: 1\nTask: put the bowl on the plate\nSuccess: False\n# episodes completed so far: 2\n""",
    "lingbot_vla": """Task: open the middle drawer\nSuccess: True\n# episodes completed so far: 1\nTask: open the middle drawer\nSuccess: True\n# episodes completed so far: 2\n""",
    "openvla": """Task: move the mug to the left\nSuccess: False\n# episodes completed so far: 1\nTask: move the mug to the left\nSuccess: True\n# episodes completed so far: 2\n""",
}


def create_sample(run_dir: Path) -> None:
    records = []
    for idx, (model, text) in enumerate(SAMPLE_LOGS.items()):
        raw = run_dir / "raw_logs" / model / f"task_{idx}.txt"
        ensure_dir(raw.parent)
        raw.write_text(text, encoding="utf-8")
        records.extend(
            parse_libero_log(
                text=text,
                model_name=model,
                task_suite="libero_10",
                task_id_hint=idx,
                runtime_seconds=12.5 + idx,
                raw_log=str(raw),
            )
        )
    write_episode_csv(records, run_dir / "metrics" / "episodes.csv")
    build_metrics(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse raw LIBERO logs into metrics CSV files.")
    parser.add_argument("--run-dir")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()

    if args.sample:
        run_dir = ROOT / "outputs" / "sample"
        ensure_dir(run_dir / "metrics")
        create_sample(run_dir)
        print(f"Sample metrics written to {run_dir / 'metrics'}")
        return 0

    if not args.run_dir:
        parser.error("--run-dir is required unless --sample is used")
    run_dir = Path(args.run_dir)
    build_metrics(run_dir)
    print(f"Metrics summarized under {run_dir / 'metrics'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
