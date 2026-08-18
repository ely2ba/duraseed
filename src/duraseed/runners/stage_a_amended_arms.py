"""Continuous fixed-coordinate Stage-A arms for the valid-tag amendment."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners import RunnerGateError
from duraseed.runners.stage_a_evidence import evaluate_panel
from duraseed.runners.stage_a_result import FinalArmEvidence, final_envelope
from duraseed.runners.stage_a_setup import (
    StageAOriginEvidence,
    create_recorded_sampler,
    run_update,
    save_candidate,
)
from duraseed.runners.stage_a_arms import observations
from duraseed.runners.stage_a_updates import Branch, restore_branch
from duraseed.runtime import SampleObservation
from duraseed.training.stage_a_amended import (
    AmendedStageAFinalEvidence,
    AmendedStageAScreenEvidence,
    StageAAnswerTagPairedItemEvidence,
)


def paired_answer_tag_items(
    origin: tuple[SampleObservation, ...], current: tuple[SampleObservation, ...]
) -> tuple[StageAAnswerTagPairedItemEvidence, ...]:
    """Pair the verifier's valid-answer-tag flag at M0 and the current arm."""

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
            StageAAnswerTagPairedItemEvidence(
                task_id,
                family,
                tuple(int(row.generation.sampling_seed) for row in after),
                tuple(bool(row.reward.reward) for row in before),
                tuple(bool(row.reward.reward) for row in after),
                tuple(
                    bool(row.reward.exact_verification.valid_answer_tag)
                    for row in before
                ),
                tuple(
                    bool(row.reward.exact_verification.valid_answer_tag)
                    for row in after
                ),
                tuple(row.generation.stop_reason == "length" for row in before),
                tuple(row.generation.stop_reason == "length" for row in after),
            )
        )
    return tuple(items)


async def run_amended_screen(
    inputs: CalibrationLiveInputs,
    origin: StageAOriginEvidence,
    output,
    journal: RemoteJournal,
    *,
    method: Literal["B-S", "B-G"],
    learning_rate: float,
    pools: dict,
    sources: dict,
) -> tuple[AmendedStageAScreenEvidence, Branch]:
    """Train M0->10 and retain both target and sentinel health evidence."""

    branch = await restore_branch(
        inputs,
        origin.boundary_state_path,
        method,
        learning_rate,
        output,
        journal,
    )
    for step in range(1, 11):
        await run_update(
            inputs,
            branch,
            step,
            pools,
            sources,
            origin.boundary_sampler_path,
            output,
            journal,
        )
    candidate = await save_candidate(
        inputs,
        branch,
        output,
        journal,
        10,
        checkpoint_suffix=f"-{output.name}",
    )
    sampler = await create_recorded_sampler(inputs, candidate, output, journal)
    target = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="targeted",
        samples_per_item=1,
        sampler_path=candidate,
        training_step=10,
        label=f"amended-screen-{method}-{learning_rate:.0e}-target",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    sentinel = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="sentinel",
        samples_per_item=1,
        sampler_path=candidate,
        training_step=10,
        label=f"amended-screen-{method}-{learning_rate:.0e}-sentinel",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    return (
        AmendedStageAScreenEvidence(
            method,
            learning_rate,
            inputs.prompt_pools.artifact.targeted_panel.value,
            inputs.prompt_pools.artifact.sentinel_panel.value,
            origin.boundary_sampler_path,
            candidate,
            paired_answer_tag_items(
                observations(origin.target_generations, origin.target_rewards), target
            ),
            paired_answer_tag_items(
                observations(origin.sentinel_generations, origin.sentinel_rewards),
                sentinel,
            ),
            tuple(branch.metrics),
            True,
        ),
        branch,
    )


async def run_amended_final(
    inputs: CalibrationLiveInputs,
    origin: StageAOriginEvidence,
    output,
    journal: RemoteJournal,
    *,
    method: Literal["B-S", "B-G"],
    learning_rate: float,
    branch: Branch,
    pools: dict,
    sources: dict,
) -> FinalArmEvidence:
    """Continue the exact step-10 branch in place through step 50."""

    for step in range(11, 51):
        await run_update(
            inputs,
            branch,
            step,
            pools,
            sources,
            origin.boundary_sampler_path,
            output,
            journal,
        )
    final_path = await save_candidate(
        inputs,
        branch,
        output,
        journal,
        50,
        checkpoint_suffix=f"-{output.name}",
    )
    sampler = await create_recorded_sampler(inputs, final_path, output, journal)
    target = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="targeted",
        samples_per_item=2,
        sampler_path=final_path,
        training_step=50,
        label=f"amended-final-{method}-target",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    sentinel = await evaluate_panel(
        inputs,
        sampler,
        output,
        role="sentinel",
        samples_per_item=1,
        sampler_path=final_path,
        training_step=50,
        label=f"amended-final-{method}-sentinel",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    evidence = AmendedStageAFinalEvidence(
        method,
        learning_rate,
        inputs.prompt_pools.artifact.targeted_panel.value,
        inputs.prompt_pools.artifact.sentinel_panel.value,
        origin.boundary_sampler_path,
        final_path,
        paired_answer_tag_items(
            observations(origin.target_generations, origin.target_rewards), target
        ),
        paired_answer_tag_items(
            observations(origin.sentinel_generations, origin.sentinel_rewards),
            sentinel,
        ),
        tuple(branch.metrics),
        True,
    )
    return final_envelope(branch, evidence)  # type: ignore[arg-type]


__all__ = ["paired_answer_tag_items", "run_amended_final", "run_amended_screen"]
