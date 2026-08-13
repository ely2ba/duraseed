"""Paid, restartable execution of the fixed boundary-extension action chain."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from duraseed.boundary_live_sampling import (
    ACTION_CAPS,
    action_limits,
    collect_groups,
    summarize,
)
from duraseed.boundary_live_setup import open_boundary_artifacts
from duraseed.boundary_live_sources import (
    BoundaryLiveSource,
    capacity_cleared_confirmation,
    confirmed_family_summaries,
    load_frozen_extension1_confirmation,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary_confirmation import (
    BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
    BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM,
    choose_refinement_family_ids,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_BROAD_SAMPLES_PER_ITEM,
    audit_new_broad_cohort,
)
from duraseed.runners import RunnerGateError
from duraseed.runners.boundary_extension import (
    BoundaryBlockInputs,
    BoundaryExtensionResult,
    build_extension2_manifest,
    reduce_block,
)
from duraseed.runtime import (
    RuntimeBundle,
    TokenLedger,
)
from duraseed.run_records import RunStatus
from duraseed.tasks.tces import TCESGeneratorConfig


async def execute_boundary_live(
    runtime: RuntimeBundle,
    sampler: Any,
    *,
    source: BoundaryLiveSource,
    config: PilotConfig,
    output_root: str | Path,
    run_id: str,
    git_commit: str,
    extension1_confirmation_path: str | Path | None = None,
) -> BoundaryExtensionResult:
    """Run four fixed paid actions; restart skips validated complete task groups."""

    generator = TCESGeneratorConfig(**config.tasks.tces.generator_kwargs())
    extension2 = build_extension2_manifest(generator)
    artifacts = open_boundary_artifacts(
        output_root,
        run_id,
        git_commit=git_commit,
        runtime=runtime,
        config=config,
        source_contract=source.contract,
        extension2=extension2,
        action_caps=ACTION_CAPS,
    )
    ledgers: dict[str, TokenLedger] = {}
    action = "extension1-confirm"
    try:
        extension1_confirmation = (
            load_frozen_extension1_confirmation(
                extension1_confirmation_path, source.extension1_broad_manifest
            )
            if extension1_confirmation_path is not None
            else capacity_cleared_confirmation(
                generator,
                source.extension1_broad_manifest,
                source.extension1_refinement_summaries,
            )
        )
        audit_new_broad_cohort(
            extension2,
            (source.initial_broad_manifest, source.extension1_broad_manifest),
            (source.initial_confirmation_manifest, extension1_confirmation),
        )
        artifacts.write_manifest(
            "extension1_confirmation_manifest.json", extension1_confirmation
        )
        artifacts.add_manifest_identity(
            "boundary_extension1_confirmation", extension1_confirmation.manifest_id
        )
        ledgers[action] = artifacts.restore_ledger(
            action,
            action_limits(
                action,
                extension1_confirmation,
                BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
                config.tinker.max_sampled_tokens,
            ),
            ACTION_CAPS[action],
        )
        e1_g, e1_r = await collect_groups(
            artifacts,
            runtime,
            sampler,
            extension1_confirmation,
            action=action,
            run_id=run_id,
            source=source,
            samples=BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
            sample_start=0,
            config=config,
            ledger=ledgers[action],
        )
        e1_summaries = confirmed_family_summaries(
            source.extension1_broad_manifest,
            extension1_confirmation,
            (
                *source.extension1_broad_generations,
                *source.extension1_refinement_generations,
                *e1_g,
            ),
            (
                *source.extension1_broad_rewards,
                *source.extension1_refinement_rewards,
                *e1_r,
            ),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                *source.contract.prior_run_ids,
                f"{run_id}:extension1-confirm",
            ),
        )
        extension1 = reduce_block(
            generator,
            BoundaryBlockInputs(
                source.contract.cohort_id,
                source.extension1_broad_manifest,
                source.extension1_family_successes,
                source.extension1_refinement_summaries,
                e1_summaries,
                (source.initial_broad_manifest,),
                (source.initial_confirmation_manifest,),
            ),
        )
        if extension1.confirmation_manifest != extension1_confirmation:
            raise RunnerGateError("Extension-1 confirmation manifest changed")
        action = "extension2-broad"
        ledgers[action] = artifacts.restore_ledger(
            action,
            action_limits(
                action,
                extension2,
                BOUNDARY_BROAD_SAMPLES_PER_ITEM,
                config.tinker.max_sampled_tokens,
            ),
            ACTION_CAPS[action],
        )
        broad_g, broad_r = await collect_groups(
            artifacts,
            runtime,
            sampler,
            extension2,
            action=action,
            run_id=run_id,
            source=source,
            samples=BOUNDARY_BROAD_SAMPLES_PER_ITEM,
            sample_start=0,
            config=config,
            ledger=ledgers[action],
        )
        broad_summaries = summarize(extension2, broad_g, broad_r, config)
        successes = {
            row.intended_family_id: row.total_successes for row in broad_summaries
        }
        candidates, audit = choose_refinement_family_ids(successes)
        refined = frozenset((*candidates, *audit))
        refine_task_ids = frozenset(
            row.task_id for row in extension2.records if row.intended_family in refined
        )
        action = "extension2-refine"
        extra = (
            BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM - BOUNDARY_BROAD_SAMPLES_PER_ITEM
        )
        ledgers[action] = artifacts.restore_ledger(
            action,
            action_limits(
                action,
                extension2,
                extra,
                config.tinker.max_sampled_tokens,
                task_count=len(refine_task_ids),
            ),
            ACTION_CAPS[action],
        )
        refine_g, refine_r = await collect_groups(
            artifacts,
            runtime,
            sampler,
            extension2,
            action=action,
            run_id=run_id,
            source=source,
            samples=extra,
            sample_start=BOUNDARY_BROAD_SAMPLES_PER_ITEM,
            config=config,
            ledger=ledgers[action],
            task_ids=refine_task_ids,
        )
        refine_summaries = summarize(
            extension2,
            (*broad_g, *refine_g),
            (*broad_r, *refine_r),
            config,
            expected=(f"{run_id}:extension2-broad", f"{run_id}:extension2-refine"),
        )
        refine_summaries = tuple(
            row for row in refine_summaries if row.intended_family_id in refined
        )
        confirmation = capacity_cleared_confirmation(
            generator, extension2, refine_summaries
        )
        artifacts.write_manifest("extension2_confirmation_manifest.json", confirmation)
        artifacts.add_manifest_identity(
            "boundary_extension2_confirmation", confirmation.manifest_id
        )
        action = "extension2-confirm"
        ledgers[action] = artifacts.restore_ledger(
            action,
            action_limits(
                action,
                confirmation,
                BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
                config.tinker.max_sampled_tokens,
            ),
            ACTION_CAPS[action],
        )
        confirm_g, confirm_r = await collect_groups(
            artifacts,
            runtime,
            sampler,
            confirmation,
            action=action,
            run_id=run_id,
            source=source,
            samples=BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
            sample_start=0,
            config=config,
            ledger=ledgers[action],
            allow_empty=True,
        )
        confirm_summaries = confirmed_family_summaries(
            extension2,
            confirmation,
            (*broad_g, *refine_g, *confirm_g),
            (*broad_r, *refine_r, *confirm_r),
            group_size=config.tinker.group_size,
            expected_run_ids=(
                f"{run_id}:extension2-broad",
                f"{run_id}:extension2-refine",
                f"{run_id}:extension2-confirm",
            ),
        )
        extension2_result = reduce_block(
            generator,
            BoundaryBlockInputs(
                BOUNDARY_BROAD_EXTENSION_2_COHORT,
                extension2,
                successes,
                refine_summaries,
                confirm_summaries,
                (source.initial_broad_manifest, source.extension1_broad_manifest),
                (
                    source.initial_confirmation_manifest,
                    extension1.confirmation_manifest,
                ),
            ),
        )
        result = BoundaryExtensionResult(
            extension1, extension2_result, "blocked_pending_three_cohort_equivalence"
        )
        artifacts.write_result(result)
        artifacts.finish(RunStatus.COMPLETED, ledgers)
        return result
    except BaseException as error:
        artifacts.record_error(action, error)
        artifacts.finish(
            RunStatus.INTERRUPTED
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else RunStatus.FAILED,
            ledgers,
        )
        raise
