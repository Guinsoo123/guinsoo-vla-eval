#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from guinsoo_vla_eval.report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Markdown report from metrics and figures.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    path = build_report(Path(args.run_dir))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
