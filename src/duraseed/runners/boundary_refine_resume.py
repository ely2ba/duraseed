"""Incident-locked refinement-only continuation for the live boundary run."""

from __future__ import annotations

from collections import Counter
import asyncio
from pathlib import Path
from typing import Any

from duraseed.boundary_live_fresh_resume import STALE_RUNTIME_INCIDENT
from duraseed.boundary_live_retry import INCIDENT
from duraseed.boundary_live_sampling import (
    ACTION_CAPS,
    action_limits,
    collect_groups,
    summarize,
)
from duraseed.boundary_live_setup import open_boundary_artifacts
from duraseed.boundary_live_sources import (
    BoundaryLiveSource,
    load_frozen_extension1_confirmation,
)
from duraseed.config import PilotConfig
from duraseed.data.boundary_confirmation import (
    BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
    BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM,
    choose_refinement_family_ids,
)
from duraseed.data.boundary_protocol import BOUNDARY_BROAD_SAMPLES_PER_ITEM
from duraseed.data.manifests import DatasetManifest
from duraseed.runners import RunnerGateError
from duraseed.runtime import RuntimeBundle, TokenLedger
from duraseed.run_records import RunStatus


async def execute_boundary_refine_resume(
    runtime: RuntimeBundle,
    sampler: Any,
    *,
    source: BoundaryLiveSource,
    config: PilotConfig,
    output_root: str | Path,
    run_id: str,
    git_commit: str,
    extension1_confirmation_path: str | Path,
    refine_retry_trace: str | Path,
    extension2: DatasetManifest,
) -> None:
    """Complete only the exact 216-group refinement grid, then stop cleanly."""

    artifacts = open_boundary_artifacts(
        output_root,
        run_id,
        git_commit=git_commit,
        runtime=runtime,
        config=config,
        source_contract=source.contract,
        extension2=extension2,
        action_caps=ACTION_CAPS,
        refine_retry_trace=refine_retry_trace,
    )
    ledgers: dict[str, TokenLedger] = {}
    action = "extension1-confirm"
    extension1 = load_frozen_extension1_confirmation(
        extension1_confirmation_path, source.extension1_broad_manifest
    )
    artifacts.write_manifest("extension1_confirmation_manifest.json", extension1)
    ledgers[action] = artifacts.restore_ledger(
        action,
        action_limits(
            action,
            extension1,
            BOUNDARY_CONFIRMATION_SAMPLES_PER_ITEM,
            config.tinker.max_sampled_tokens,
        ),
        ACTION_CAPS[action],
    )
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
    summaries = summarize(extension2, broad_g, broad_r, config)
    successes = {row.intended_family_id: row.total_successes for row in summaries}
    candidates, audit = choose_refinement_family_ids(successes)
    selected = frozenset((*candidates, *audit))
    selected_records = tuple(
        row for row in extension2.records if row.intended_family in selected
    )
    expected = STALE_RUNTIME_INCIDENT["refinement"]
    if (
        len(candidates) != expected["positive_families"]
        or len(audit) != expected["audit_families"]
        or len(selected_records) != expected["tasks"]
        or selected_records[0].task_id != INCIDENT["pending"]["task_id"]
    ):
        raise RunnerGateError("fresh refinement selection changed")
    action = "extension2-refine"
    extra = BOUNDARY_REFINEMENT_TOTAL_SAMPLES_PER_ITEM - BOUNDARY_BROAD_SAMPLES_PER_ITEM
    task_ids = frozenset(row.task_id for row in selected_records)
    ledgers[action] = artifacts.restore_ledger(
        action,
        action_limits(
            action,
            extension2,
            extra,
            config.tinker.max_sampled_tokens,
            task_count=len(task_ids),
        ),
        ACTION_CAPS[action],
    )
    try:
        await collect_groups(
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
            task_ids=task_ids,
        )
        counts = Counter(group_action for group_action, _ in artifacts.groups)
        expected_counts = {
            **STALE_RUNTIME_INCIDENT["journal_groups"],
            action: expected["tasks"],
        }
        if (
            counts != expected_counts
            or len(artifacts.groups) != sum(expected_counts.values())
            or artifacts.pending.exists()
        ):
            raise RunnerGateError("refinement-only stop boundary changed")
        artifacts.finish(RunStatus.INTERRUPTED, ledgers)
    except BaseException as error:
        artifacts.record_error(action, error)
        artifacts.finish(
            RunStatus.INTERRUPTED
            if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError))
            else RunStatus.FAILED,
            ledgers,
        )
        raise


__all__ = ["execute_boundary_refine_resume"]
