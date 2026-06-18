#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.plots import generate_figures


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static PNG figures from metrics CSV files.")
    parser.add_argument("--run-dir")
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    if args.sample:
        run_dir = ROOT / "outputs" / "sample"
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        parser.error("--run-dir is required unless --sample is used")
    figures = generate_figures(run_dir)
    for path in figures:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
