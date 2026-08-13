"""Authenticated teacher and fixed data schedules for Pilot 0."""

from __future__ import annotations

from collections import Counter, defaultdict

from duraseed.data.manifests import MAPSTaskManifestRecord, TCESTaskManifestRecord
from duraseed.data.stage_a_prompt_pools import PromptPoolStratum
from duraseed.pilot0_contract import Pilot0Inputs, PilotSeedSources
from duraseed.runners import RunnerGateError
from duraseed.tasks.tces import enumerate_task, generate_teacher_trace
from duraseed.training.sft import (
    VerifiedSourceRecord,
    build_solver_teacher_record,
    build_stage_b_maps_record,
    build_teacher_dose_records,
)


def _tces_completion(record: TCESTaskManifestRecord) -> str:
    enumeration = enumerate_task(record.to_task())
    expression = enumeration.family_representatives.get(record.intended_family)
    if not enumeration.complete or expression is None:
        raise RunnerGateError("Pilot-0 TCES source lacks its intended-family solution")
    return generate_teacher_trace(expression)


def boundary_teacher_sources(
    inputs: Pilot0Inputs, source: PilotSeedSources
) -> tuple[VerifiedSourceRecord, ...]:
    families = source.prompt_pools.artifact.boundary_family_ids
    rows = tuple(
        (record, _tces_completion(record))
        for record in source.teacher_train.records
        if isinstance(record, TCESTaskManifestRecord)
        and record.intended_family in families
    )
    dose = inputs.teacher_recipe.decision.selected_dose
    assert dose is not None
    return build_teacher_dose_records(
        source_manifest=source.teacher_train,
        solver_completions=rows,
        selected_families=families,
        demonstrations_per_family=dose,
    )


def stage_a_solver_sources(source: PilotSeedSources) -> dict[str, VerifiedSourceRecord]:
    result = {}
    for record in source.prompt_pools.a_rl_train_manifest.records:
        if not isinstance(record, TCESTaskManifestRecord):
            raise RunnerGateError("Pilot-0 Stage-A training manifest is not TCES")
        result[record.task_id] = build_solver_teacher_record(
            source_manifest=source.prompt_pools.a_rl_train_manifest,
            source_record=record,
            completion=_tces_completion(record),
        )
    return result


def ordered_stage_a_pools(
    source: PilotSeedSources,
) -> dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]]:
    artifact = source.prompt_pools.artifact
    by_stratum = {
        PromptPoolStratum.BOUNDARY: artifact.boundary_family_ids,
        PromptPoolStratum.INTERMEDIATE: artifact.intermediate_family_ids,
        PromptPoolStratum.BROAD_RANDOM: artifact.broad_random_family_ids,
    }
    result = {}
    for stratum, families in by_stratum.items():
        by_family = {
            family: tuple(
                sorted(
                    (
                        row
                        for row in source.prompt_pools.a_rl_train_manifest.records
                        if isinstance(row, TCESTaskManifestRecord)
                        and row.intended_family == family
                    ),
                    key=lambda row: row.task_id,
                )
            )
            for family in families
        }
        if {len(rows) for rows in by_family.values()} != {64}:
            raise RunnerGateError("Pilot-0 Stage-A pool changed its 64-item strata")
        result[stratum] = tuple(
            by_family[family][item] for item in range(64) for family in families
        )
    return result


def scheduled_stage_a_records(
    pools: dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]],
    order: tuple[PromptPoolStratum, ...],
    step: int,
) -> tuple[TCESTaskManifestRecord, ...]:
    counts: dict[PromptPoolStratum, int] = defaultdict(int)
    per_step = Counter(order)
    selected = []
    for stratum in order:
        offset = (step - 1) * per_step[stratum] + counts[stratum]
        selected.append(pools[stratum][offset % len(pools[stratum])])
        counts[stratum] += 1
    return tuple(selected)


def stage_b_sources(source: PilotSeedSources) -> tuple[VerifiedSourceRecord, ...]:
    values = []
    for record in source.b_train.records:
        if not isinstance(record, MAPSTaskManifestRecord):
            raise RunnerGateError("Pilot-0 Stage-B training manifest is not MAPS")
        values.append(
            build_stage_b_maps_record(
                source_manifest=source.b_train,
                source_record=record,
                completion=f"<answer>{record.shortest_programs[0]}</answer>",
            )
        )
    return tuple(values)


__all__ = [
    "boundary_teacher_sources",
    "ordered_stage_a_pools",
    "scheduled_stage_a_records",
    "stage_a_solver_sources",
    "stage_b_sources",
]
