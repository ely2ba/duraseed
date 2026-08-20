"""Exact local solver corpus for the six-replay capability-dose schedule."""

from __future__ import annotations

from typing import Any

from duraseed.capability_dose_budget import DOSE_TRAIN_TOKEN_CEILING
from duraseed.runners import RunnerGateError
from duraseed.runners.solver_teacher_cache import solver_teacher_completion
from duraseed.runners.stage_a_evidence import ordered_pools, scheduled_records
from duraseed.runtime import sft_datum
from duraseed.training.capability_dose_evidence import EPOCH_UPDATES
from duraseed.training.sft import VerifiedSourceRecord, build_solver_teacher_record


def capability_dose_cycle(inputs: Any) -> tuple[tuple[Any, ...], ...]:
    pools = ordered_pools(inputs.prompt_pools)
    return tuple(
        scheduled_records(pools, inputs.prompt_pools.artifact.bs_slot_order, step)
        for step in range(1, EPOCH_UPDATES + 1)
    )


def prepare_capability_dose_sources(
    inputs: Any,
) -> dict[str, VerifiedSourceRecord]:
    """Solve the distinct records in the frozen 1,568-presentation epoch."""

    cycle = capability_dose_cycle(inputs)
    manifest = inputs.prompt_pools.a_rl_train_manifest
    by_id = {row.task_id: row for rows in cycle for row in rows}
    sources = {
        task_id: build_solver_teacher_record(
            source_manifest=manifest,
            source_record=row,
            completion=solver_teacher_completion(row),
        )
        for task_id, row in sorted(by_id.items())
    }
    cycle_tokens = sum(
        int(sft_datum(inputs.runtime, sources[row.task_id]).model_input.length)
        for rows in cycle
        for row in rows
    )
    presentation_count = sum(len(rows) for rows in cycle)
    if (
        presentation_count != 1_568
        or len(sources) != 1_552
        or cycle_tokens * 6 != DOSE_TRAIN_TOKEN_CEILING
    ):
        raise RunnerGateError("capability-dose solver corpus differs from the freeze")
    return sources


__all__ = ["capability_dose_cycle", "prepare_capability_dose_sources"]
