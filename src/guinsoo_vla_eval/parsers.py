from __future__ import annotations

import re
from dataclasses import dataclass


TASK_RE = re.compile(r"^Task:\s*(?P<task>.+?)\s*$", re.MULTILINE)
SUCCESS_RE = re.compile(r"^Success:\s*(?P<success>True|False)\s*$", re.MULTILINE)
EXCEPTION_RE = re.compile(r"(Caught exception:|Traceback \(most recent call last\):|Error:|Exception:)(?P<msg>.*)")


@dataclass(frozen=True)
class EpisodeRecord:
    model_name: str
    task_suite: str
    task_id: int | None
    raw_task_id: int | None
    task_description: str
    episode_index: int
    success: bool
    steps: int | None
    runtime_seconds: float | None
    failure_reason: str
    raw_log: str


def infer_failure_reason(success: bool, text: str, returncode: int = 0) -> str:
    if success:
        return ""
    match = EXCEPTION_RE.search(text)
    if match:
        msg = match.group("msg").strip()
        return f"exception:{msg[:120]}" if msg else "exception"
    if returncode != 0:
        return f"nonzero_exit:{returncode}"
    return "not_successful"


def parse_libero_log(
    *,
    text: str,
    model_name: str,
    task_suite: str,
    task_id_hint: int | None,
    runtime_seconds: float | None,
    raw_log: str,
    returncode: int = 0,
) -> list[EpisodeRecord]:
    task_matches = list(TASK_RE.finditer(text))
    successes = list(SUCCESS_RE.finditer(text))
    records: list[EpisodeRecord] = []
    task_by_pos: list[tuple[int, str]] = [(m.start(), m.group("task").strip()) for m in task_matches]
    seen_tasks: list[str] = []

    for episode_index, match in enumerate(successes, start=1):
        success = match.group("success") == "True"
        description = "unknown_task"
        for pos, task in task_by_pos:
            if pos < match.start():
                description = task
            else:
                break
        if description not in seen_tasks:
            seen_tasks.append(description)
        # Provisional id from the CLI hint or log order. The orchestrator
        # remaps this to a canonical LIBERO id by task_description, so it is
        # only kept as raw_task_id for debugging.
        raw_task_id = task_id_hint if task_id_hint is not None else seen_tasks.index(description)
        records.append(
            EpisodeRecord(
                model_name=model_name,
                task_suite=task_suite,
                task_id=raw_task_id,
                raw_task_id=raw_task_id,
                task_description=description,
                episode_index=episode_index,
                success=success,
                steps=None,
                runtime_seconds=runtime_seconds,
                failure_reason=infer_failure_reason(success, text, returncode),
                raw_log=raw_log,
            )
        )

    if not records and returncode != 0:
        records.append(
            EpisodeRecord(
                model_name=model_name,
                task_suite=task_suite,
                task_id=task_id_hint,
                raw_task_id=task_id_hint,
                task_description="run_failed_before_episode",
                episode_index=0,
                success=False,
                steps=None,
                runtime_seconds=runtime_seconds,
                failure_reason=infer_failure_reason(False, text, returncode),
                raw_log=raw_log,
            )
        )
    return records
