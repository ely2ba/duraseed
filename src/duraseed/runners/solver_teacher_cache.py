"""Bounded reuse of deterministic exact-solver teacher completions."""

from __future__ import annotations

from functools import lru_cache

from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.runners import RunnerGateError
from duraseed.schemas import TCESTask
from duraseed.tasks.tces import enumerate_task, generate_teacher_trace


_CACHE_SIZE = 4_096


@lru_cache(maxsize=_CACHE_SIZE)
def _completion(task_json: str, intended_family: str) -> str:
    task = TCESTask.model_validate_json(task_json)
    enumeration = enumerate_task(task)
    expression = enumeration.family_representatives.get(intended_family)
    if not enumeration.complete or expression is None:
        raise RunnerGateError("teacher task lacks its intended-family solution")
    return generate_teacher_trace(expression)


def solver_teacher_completion(record: TCESTaskManifestRecord) -> str:
    """Return one verified-by-construction trace, retaining only bounded text."""

    if not isinstance(record, TCESTaskManifestRecord):
        raise RunnerGateError("teacher source is not TCES")
    return _completion(record.to_task().model_dump_json(), record.intended_family)


__all__ = ["solver_teacher_completion"]
