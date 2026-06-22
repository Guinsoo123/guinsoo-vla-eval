"""Canonical LIBERO task registry and cross-model task alignment.

OpenVLA runs the whole suite while WALL-OSS / LingBot run per task, so the
positional ``task_id`` extracted from logs is not comparable across models.
This module derives the canonical ``id -> description`` map straight from the
LIBERO suite definition and re-keys every parsed episode by its
``task_description`` so all models share one task identity.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .config import EvalConfig, ModelConfig
from .parsers import EpisodeRecord
from .paths import ensure_dir, find_conda, libero_pythonpath, resolve_libero_repo_root

_WS_RE = re.compile(r"\s+")

_FAILED_DESC = "run_failed_before_episode"

# Dumps the canonical task list of a LIBERO suite to a JSON file. Kept as a
# string so it can run inside each model's own conda env via ``conda run``.
_DUMP_SNIPPET = """
import json, sys
from libero.libero import benchmark

suite_name = sys.argv[1]
out_path = sys.argv[2]
suite = benchmark.get_benchmark_dict()[suite_name]()
tasks = []
for i in range(suite.n_tasks):
    task = suite.get_task(i)
    tasks.append({"id": i, "language": task.language})
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"task_suite": suite_name, "tasks": tasks}, f, ensure_ascii=False)
"""


def normalize_desc(text: str) -> str:
    """Normalize a task description for robust matching across upstreams."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def _libero_env(libero_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = libero_pythonpath(libero_path, env.get("PYTHONPATH", ""))
    env["LIBERO_PATH"] = str(libero_path)
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    return env


def fetch_canonical_tasks(libero_path: str | Path, task_suite: str, conda_env: str) -> dict[int, str]:
    """Run LIBERO inside ``conda_env`` to obtain the canonical task list."""
    libero_path = Path(libero_path)
    out_dir = ensure_dir(Path("/tmp") / "guinsoo_vla_eval")
    out_path = out_dir / f"canonical_{task_suite}_{conda_env}.json"
    cmd = [
        find_conda(),
        "run",
        "--no-capture-output",
        "-n",
        conda_env,
        "env",
        f"PYTHONPATH={libero_pythonpath(libero_path, os.environ.get('PYTHONPATH', ''))}",
        f"LIBERO_PATH={resolve_libero_repo_root(libero_path)}",
        "python",
        "-c",
        _DUMP_SNIPPET,
        task_suite,
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        env=_libero_env(libero_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or "").strip()[-600:]
        raise RuntimeError(f"failed to load canonical tasks for '{task_suite}' in env '{conda_env}': {tail}")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    return {int(item["id"]): str(item["language"]) for item in data["tasks"]}


def load_canonical_tasks(
    libero_path: str | Path,
    task_suite: str,
    conda_env: str,
    cache_path: str | Path | None = None,
) -> dict[int, str]:
    """Load the canonical task map, using ``cache_path`` if it already exists."""
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("task_suite") == task_suite:
            return {int(k): str(v) for k, v in cached["tasks"].items()}
    canonical = fetch_canonical_tasks(libero_path, task_suite, conda_env)
    if cache_path:
        ensure_dir(cache_path.parent)
        payload = {"task_suite": task_suite, "tasks": {str(k): v for k, v in canonical.items()}}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return canonical


def build_reverse_map(canonical: dict[int, str]) -> dict[str, int]:
    """Map normalized description -> canonical task id."""
    return {normalize_desc(desc): task_id for task_id, desc in canonical.items()}


def canonical_from_records(records: list[EpisodeRecord]) -> tuple[dict[int, str], dict[str, int]]:
    """Fallback registry built from the descriptions actually seen in logs.

    Used only when the LIBERO suite cannot be queried. Ids are assigned by
    sorted normalized description so they are at least identical across models.
    """
    seen: dict[str, str] = {}
    for record in records:
        key = normalize_desc(record.task_description)
        if key and record.task_description != _FAILED_DESC and key not in seen:
            seen[key] = record.task_description
    ordered = sorted(seen.keys())
    canonical = {idx: seen[key] for idx, key in enumerate(ordered)}
    reverse = {key: idx for idx, key in enumerate(ordered)}
    return canonical, reverse


def remap_records(
    records: list[EpisodeRecord],
    reverse_map: dict[str, int],
) -> tuple[list[EpisodeRecord], list[str]]:
    """Re-key episodes to canonical task ids by their description.

    Returns the remapped records and the sorted list of descriptions that could
    not be matched against the canonical registry.
    """
    remapped: list[EpisodeRecord] = []
    unmatched: set[str] = set()
    for record in records:
        key = normalize_desc(record.task_description)
        canonical_id = reverse_map.get(key)
        if canonical_id is None and record.task_description != _FAILED_DESC:
            unmatched.add(record.task_description)
        remapped.append(replace(record, task_id=canonical_id))
    return remapped, sorted(unmatched)


def intersect_covered_ids(records: list[EpisodeRecord], models: list[str]) -> set[int]:
    """Canonical ids that every model produced at least one episode for."""
    per_model: dict[str, set[int]] = defaultdict(set)
    for record in records:
        if record.task_id is not None:
            per_model[record.model_name].add(record.task_id)
    covered = [per_model.get(model, set()) for model in models]
    if not covered:
        return set()
    return set.intersection(*covered) if all(covered) else set()


def check_alignment(
    records: list[EpisodeRecord],
    expected_task_ids: list[int],
    num_trials_per_task: int,
    models: list[str],
) -> dict:
    """Verify every model covers the same task set with the same episode count."""
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if record.task_id is not None:
            counts[record.model_name][record.task_id] += 1

    expected = set(expected_task_ids)
    issues: list[str] = []
    coverage: dict[str, dict[str, object]] = {}
    for model in models:
        model_counts = counts.get(model, {})
        covered = set(model_counts)
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        bad_counts = {
            task_id: model_counts[task_id]
            for task_id in sorted(covered & expected)
            if model_counts[task_id] != num_trials_per_task
        }
        if missing:
            issues.append(f"{model}: missing task ids {missing}")
        if extra:
            issues.append(f"{model}: unexpected task ids {extra}")
        if bad_counts:
            issues.append(f"{model}: episode count != {num_trials_per_task} for {bad_counts}")
        coverage[model] = {
            "covered_task_ids": sorted(covered),
            "episodes_per_task": {str(k): model_counts[k] for k in sorted(covered)},
        }

    return {
        "aligned": not issues,
        "expected_task_ids": sorted(expected),
        "num_trials_per_task": num_trials_per_task,
        "issues": issues,
        "coverage": coverage,
    }


def align_and_filter(
    records: list[EpisodeRecord],
    eval_cfg: EvalConfig,
    model_configs: dict[str, ModelConfig],
    run_dir: Path,
) -> tuple[list[EpisodeRecord], dict]:
    """Re-key episodes to canonical LIBERO ids, filter, and validate alignment.

    This is the single source of truth for cross-model task alignment used by
    both ``eval_all.py`` and ``eval_one.py``. Returns the filtered records and a
    serializable alignment report.
    """
    cache_path = run_dir / "metrics" / "canonical_tasks.json"
    reference = model_configs.get("openvla") or next(iter(model_configs.values()))
    canonical: dict[int, str] | None = None
    try:
        canonical = load_canonical_tasks(
            reference.libero_path, eval_cfg.task_suite, reference.conda_env, cache_path
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, warn loudly
        print(f"[WARN] could not load canonical LIBERO task list: {exc}")

    if canonical:
        reverse_map = build_reverse_map(canonical)
        use_id_filter = True
    else:
        print("[WARN] falling back to description-derived task ids; --task_ids filter is ignored,")
        print("       comparison is restricted to tasks shared by all models.")
        canonical, reverse_map = canonical_from_records(records)
        use_id_filter = False

    records, unmatched = remap_records(records, reverse_map)
    if unmatched:
        print(f"[WARN] {len(unmatched)} task description(s) not found in canonical registry:")
        for desc in unmatched:
            print(f"       - {desc}")

    if use_id_filter:
        records = [r for r in records if r.task_id in eval_cfg.task_ids]
    else:
        shared = intersect_covered_ids(records, eval_cfg.models)
        records = [r for r in records if r.task_id in shared]

    alignment = check_alignment(
        records, eval_cfg.task_ids, eval_cfg.num_trials_per_task, eval_cfg.models
    )
    alignment["used_canonical_libero"] = use_id_filter
    alignment["unmatched_descriptions"] = unmatched
    return records, alignment
