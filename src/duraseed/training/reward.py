"""Pure exact-verifier dispatch for training samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from duraseed.schemas import MAPSTask, TCESTask, VerificationResult
from duraseed.tasks.maps.verifier import verify_maps
from duraseed.tasks.tces.verifier import verify_completion as verify_tces


VerifiableTask: TypeAlias = TCESTask | MAPSTask


def verify_task_completion(
    completion: str,
    task: VerifiableTask,
) -> VerificationResult:
    """Run the authoritative exact verifier for one TCES or MAPS completion."""

    if not isinstance(completion, str):
        raise TypeError("completion must be text")
    if isinstance(task, TCESTask):
        return verify_tces(completion, task)
    if isinstance(task, MAPSTask):
        return verify_maps(completion, task).to_shared_result()
    raise TypeError(f"unsupported verifier task type: {type(task).__name__}")


def verify_exact_completions(
    completions: Sequence[str],
    tasks: Sequence[VerifiableTask],
) -> tuple[VerificationResult, ...]:
    """Verify aligned completion/task pairs without runtime-specific adapters."""

    if isinstance(completions, (str, bytes, bytearray)) or not isinstance(
        completions, Sequence
    ):
        raise TypeError("completions must be an explicit sequence")
    if isinstance(tasks, (str, bytes, bytearray)) or not isinstance(tasks, Sequence):
        raise TypeError("tasks must be an explicit sequence")
    if len(completions) != len(tasks):
        raise ValueError("completions and tasks must align one-to-one")
    return tuple(
        verify_task_completion(completion, task)
        for completion, task in zip(completions, tasks, strict=True)
    )


__all__ = [
    "VerifiableTask",
    "verify_exact_completions",
    "verify_task_completion",
]
