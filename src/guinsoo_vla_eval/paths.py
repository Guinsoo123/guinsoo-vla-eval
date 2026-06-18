from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expand_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def ensure_dir(path: str | Path) -> Path:
    resolved = expand_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def find_conda() -> str:
    found = shutil.which("conda")
    if found:
        return found
    fallback = Path("/home/qj00433/miniconda3/condabin/conda")
    if fallback.exists():
        return str(fallback)
    return "conda"


def rel_to_project(path: str | Path) -> str:
    resolved = expand_path(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
