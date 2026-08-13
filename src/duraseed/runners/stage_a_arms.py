"""One bounded Stage-A screen or final branch attempt."""

from __future__ import annotations

from typing import Literal

from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.stage_a_evidence import evaluate_panel, paired_items
from duraseed.runners.stage_a_setup import (
    StageAOriginEvidence,
    create_recorded_sampler,
    run_update,
    save_candidate,
)
from duraseed.runners.stage_a_result import FinalArmEvidence, final_envelope
from duraseed.runners.stage_a_updates import Branch, restore_branch
from duraseed.runtime import SampleObservation
from duraseed.training.stage_a_calibration import (
    StageAFinalEvidence,
    StageAScreenEvidence,
)


def observations(generations, rewards) -> tuple[SampleObservation, ...]:
    by_sample = {row.sample_id: row for row in rewards}
    if set(by_sample) != {row.sample_id for row in generations}:
        raise ValueError("persisted Stage-A generation/reward join differs")
    return tuple(
        SampleObservation(row, by_sample[row.sample_id], None, (), ())
        for row in generations
    )


async def run_screen(
    inputs: CalibrationLiveInputs,
    origin: StageAOriginEvidence,
    output,
    journal: RemoteJournal,
    *,
    method: Literal["B-S", "B-G"],
    learning_rate: float,
    pools: dict,
    sources: dict,
) -> tuple[StageAScreenEvidence, Branch]:
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
        label=f"screen-{method}-{learning_rate:.0e}-target",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    await evaluate_panel(
        inputs,
        sampler,
        output,
        role="sentinel",
        samples_per_item=1,
        sampler_path=candidate,
        training_step=10,
        label=f"screen-{method}-{learning_rate:.0e}-sentinel",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    result = StageAScreenEvidence(
        method,
        learning_rate,
        inputs.prompt_pools.artifact.targeted_panel.value,
        origin.boundary_sampler_path,
        candidate,
        paired_items(
            observations(origin.target_generations, origin.target_rewards), target
        ),
        tuple(branch.metrics),
        True,
    )
    return result, branch


async def run_final(
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
        label=f"final-{method}-target",
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
        label=f"final-{method}-sentinel",
        origin_sampler_path=origin.boundary_sampler_path,
        journal=journal,
        method=method,
    )
    evidence = StageAFinalEvidence(
        method,
        learning_rate,
        inputs.prompt_pools.artifact.targeted_panel.value,
        inputs.prompt_pools.artifact.sentinel_panel.value,
        origin.boundary_sampler_path,
        final_path,
        paired_items(
            observations(origin.target_generations, origin.target_rewards), target
        ),
        paired_items(
            observations(origin.sentinel_generations, origin.sentinel_rewards),
            sentinel,
        ),
        tuple(branch.metrics),
        True,
    )
    return final_envelope(branch, evidence)


__all__ = ["observations", "run_final", "run_screen"]
