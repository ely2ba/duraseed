"""Concrete data schedules and paired monitor evidence for Stage-A calibration."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.stage_a_prompt_pools import (
    PromptPoolStratum,
    StageAPromptPoolBundle,
)
from duraseed.run_records import append_jsonl
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import (
    SampleObservation,
    SamplingCoordinates,
    SamplingTask,
    sample_seeded,
)
from duraseed.tasks.tces import enumerate_task, generate_teacher_trace, render_prompt
from duraseed.training.sft import (
    VerifiedSourceRecord,
    build_solver_teacher_record,
    build_teacher_dose_records,
)
from duraseed.training.stage_a_calibration import StageAPairedItemEvidence


CALIBRATION_SEED = 17


def _completion(record: TCESTaskManifestRecord) -> str:
    enumeration = enumerate_task(record.to_task())
    expression = enumeration.family_representatives.get(record.intended_family)
    if not enumeration.complete or expression is None:
        raise RunnerGateError("Stage-A task lacks its intended-family solution")
    return generate_teacher_trace(expression)


def solver_sources(manifest: DatasetManifest) -> tuple[VerifiedSourceRecord, ...]:
    values = []
    for record in manifest.records:
        if not isinstance(record, TCESTaskManifestRecord):
            raise RunnerGateError("Stage-A training manifest is not TCES")
        values.append(
            build_solver_teacher_record(
                source_manifest=manifest,
                source_record=record,
                completion=_completion(record),
            )
        )
    return tuple(values)


def boundary_sources(
    inputs: CalibrationLiveInputs, dose: int
) -> tuple[VerifiedSourceRecord, ...]:
    panel = inputs.teacher_sources.panel
    assignment = next(
        (
            row
            for row in panel.seed_block_assignments
            if row.training_seed == CALIBRATION_SEED
        ),
        None,
    )
    if assignment is None:
        raise RunnerGateError("Stage-A calibration seed is absent from panel schedule")
    families = (
        panel.panel_a_family_ids
        if assignment.targeted_panel.value == "A"
        else panel.panel_b_family_ids
    )
    manifest = inputs.teacher_sources.target_train_manifest
    completions = tuple(
        (record, _completion(record))
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
        and record.intended_family in families
    )
    return build_teacher_dose_records(
        source_manifest=manifest,
        solver_completions=completions,
        selected_families=families,
        demonstrations_per_family=dose,
    )


def ordered_pools(
    bundle: StageAPromptPoolBundle,
) -> dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]]:
    artifact = bundle.artifact
    family_sets = {
        PromptPoolStratum.BOUNDARY: artifact.boundary_family_ids,
        PromptPoolStratum.INTERMEDIATE: artifact.intermediate_family_ids,
        PromptPoolStratum.BROAD_RANDOM: artifact.broad_random_family_ids,
    }
    result = {}
    for stratum, families in family_sets.items():
        by_family = {
            family: tuple(
                sorted(
                    (
                        row
                        for row in bundle.a_rl_train_manifest.records
                        if isinstance(row, TCESTaskManifestRecord)
                        and row.intended_family == family
                    ),
                    key=lambda row: row.task_id,
                )
            )
            for family in families
        }
        if {len(rows) for rows in by_family.values()} != {64}:
            raise RunnerGateError("Stage-A prompt pool changed its 64-item strata")
        result[stratum] = tuple(
            by_family[family][item] for item in range(64) for family in families
        )
    return result


def scheduled_records(
    pools: dict[PromptPoolStratum, tuple[TCESTaskManifestRecord, ...]],
    order: tuple[PromptPoolStratum, ...],
    step: int,
) -> tuple[TCESTaskManifestRecord, ...]:
    counts: dict[PromptPoolStratum, int] = defaultdict(int)
    per_step = {stratum: order.count(stratum) for stratum in PromptPoolStratum}
    selected = []
    for stratum in order:
        offset = (step - 1) * per_step[stratum] + counts[stratum]
        selected.append(pools[stratum][offset % len(pools[stratum])])
        counts[stratum] += 1
    return tuple(selected)


def _monitor_records(
    bundle: StageAPromptPoolBundle, role: Literal["targeted", "sentinel"]
) -> tuple[TCESTaskManifestRecord, ...]:
    families = (
        bundle.artifact.boundary_family_ids
        if role == "targeted"
        else bundle.artifact.sentinel_family_ids
    )
    records = []
    for family in families:
        rows = sorted(
            (
                row
                for row in bundle.a_monitor_manifest.records
                if isinstance(row, TCESTaskManifestRecord)
                and row.intended_family == family
            ),
            key=lambda row: (row.item_index, row.task_id),
        )[:8]
        if len(rows) != 8:
            raise RunnerGateError("Stage-A monitor no longer has eight items/family")
        records.extend(rows)
    if len(records) != 96:
        raise RunnerGateError("Stage-A monitor must contain 96 items")
    return tuple(records)


async def evaluate_panel(
    inputs: CalibrationLiveInputs,
    sampler: object,
    output_directory: Path,
    *,
    role: Literal["targeted", "sentinel"],
    samples_per_item: int,
    sampler_path: str,
    training_step: int,
    label: str,
    origin_sampler_path: str,
    journal: RemoteJournal,
    method: Literal["B-S", "B-G"] | None = None,
) -> tuple[SampleObservation, ...]:
    observations = []
    for record in _monitor_records(inputs.prompt_pools, role):
        prompt_text = render_prompt(record.to_task())
        prompt = inputs.runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": prompt_text}], role="assistant"
        )
        journal.begin(
            "stage-a-monitor-group",
            {"label": label, "task_id": record.task_id},
            {
                "prefill_tokens": int(prompt.length) * samples_per_item,
                "sample_tokens": inputs.max_tokens.selected_max_tokens
                * samples_per_item,
                "train_tokens": 0,
            },
        )
        rows = await sample_seeded(
            inputs.runtime,
            sampler,
            SamplingTask(
                inputs.prompt_pools.a_monitor_manifest.manifest_id,
                record.task_id,
                "tces",
                "a_monitor",
                prompt_text,
                record.to_task(),
                record.item_index,
                record.intended_family,
                role,
            ),
            SamplingCoordinates(
                inputs.run_id,
                label,
                "evaluation",
                "stage_a",
                training_step,
                sampler_path,
                origin_sampler_path,
                CALIBRATION_SEED,
                "tinker.stage_a.seed-17.monitor",
                method,
            ),
            group_size=samples_per_item,
            max_tokens=inputs.max_tokens.selected_max_tokens,
            temperature=float(inputs.config.evaluation["temperature"]),
            top_p=float(inputs.config.evaluation["top_p"]),
            ledger=inputs.stage_a_ledger,
        )
        observations.extend(rows)
        for row in rows:
            append_jsonl(output_directory / "generations.jsonl", row.generation)
            append_jsonl(output_directory / "rewards.jsonl", row.reward)
        journal.complete({"operation": "stage-a-monitor-group", "row_count": len(rows)})
    return tuple(observations)


def paired_items(
    origin: tuple[SampleObservation, ...], current: tuple[SampleObservation, ...]
) -> tuple[StageAPairedItemEvidence, ...]:
    origin_by_coordinate = {
        (row.generation.task_id, row.generation.sampling_seed): row for row in origin
    }
    current_by_task: dict[str, list[SampleObservation]] = defaultdict(list)
    for row in current:
        current_by_task[row.generation.task_id].append(row)
    items = []
    for task_id in sorted(current_by_task):
        after = sorted(
            current_by_task[task_id], key=lambda row: row.generation.sample_index
        )
        before = []
        for row in after:
            key = (task_id, row.generation.sampling_seed)
            if key not in origin_by_coordinate:
                raise RunnerGateError("Stage-A monitor sample has no paired origin")
            before.append(origin_by_coordinate[key])
        family = after[0].generation.assigned_family_id
        if family is None:
            raise RunnerGateError("Stage-A monitor omitted family identity")
        items.append(
            StageAPairedItemEvidence(
                task_id,
                family,
                tuple(int(row.generation.sampling_seed) for row in after),
                tuple(bool(row.reward.reward) for row in before),
                tuple(bool(row.reward.reward) for row in after),
                tuple(
                    _wrapper_compliant(row.generation.completion_text) for row in before
                ),
                tuple(
                    _wrapper_compliant(row.generation.completion_text) for row in after
                ),
                tuple(row.generation.stop_reason == "length" for row in before),
                tuple(row.generation.stop_reason == "length" for row in after),
            )
        )
    return tuple(items)


def _wrapper_compliant(completion: str) -> bool:
    """Reproduce the archived outer-whitespace canonical wrapper estimand."""

    canonical = completion.strip()
    opening, closing = "<answer>", "</answer>"
    return (
        canonical.count(opening) == canonical.count(closing) == 1
        and canonical.startswith(opening)
        and canonical.endswith(closing)
    )


__all__ = [
    "boundary_sources",
    "evaluate_panel",
    "ordered_pools",
    "paired_items",
    "scheduled_records",
    "solver_sources",
]
