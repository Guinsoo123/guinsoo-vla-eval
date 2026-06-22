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


def resolve_libero_repo_root(path: str | Path) -> Path:
    """Return the PYTHONPATH root that exposes ``libero.libero``.

    LIBERO's repository layout is unusual: the configured repo root contains a
    ``libero/`` directory, and that directory contains the actual
    ``libero/__init__.py`` package.  Adding the inner directory to PYTHONPATH
    makes ``import libero`` work but breaks upstream imports that expect
    ``from libero.libero import benchmark``.
    """
    resolved = expand_path(path)
    candidates = [resolved, resolved.parent, resolved.parent.parent]
    for candidate in candidates:
        if (candidate / "libero" / "libero" / "__init__.py").exists():
            return candidate
    return resolved


def libero_pythonpath(libero_path: str | Path, existing: str = "") -> str:
    root = resolve_libero_repo_root(libero_path)
    blocked = {
        str(root / "libero"),
        str(root / "libero" / "libero"),
    }
    kept = []
    for entry in existing.split(":"):
        if not entry:
            continue
        try:
            resolved = str(expand_path(entry))
        except Exception:  # noqa: BLE001 - keep unusual PYTHONPATH entries intact
            resolved = entry
        if resolved not in blocked:
            kept.append(entry)
    return ":".join([str(root)] + kept)


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
