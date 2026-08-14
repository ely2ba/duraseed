"""Concrete orchestration for the fixed two-block boundary continuation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from duraseed.boundary_capacity import audit_family_split_capacities
from duraseed.data.boundary import (
    BoundaryFamilySummary,
    assess_refinement_finalist_gate,
    summarize_m0_boundary,
)
from duraseed.data.boundary_confirmation import (
    ConfirmationEvidence,
    broad_cohort,
    build_confirmation_manifest,
    choose_refinement_family_ids,
    reduce_confirmation_evidence,
    regenerate_family_templates,
)
from duraseed.data.boundary_freeze import (
    BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS,
    BoundaryFreezeUnverifiedError,
    freeze_three_cohort_panels,
)
from duraseed.data.boundary_protocol import (
    BOUNDARY_BROAD_EXTENSION_1_COHORT,
    BOUNDARY_BROAD_EXTENSION_2_COHORT,
    BOUNDARY_ENGINEERING_SEED,
    audit_new_broad_cohort,
    build_broad_manifest,
)
from duraseed.data.manifests import DatasetManifest
from duraseed.data.panel_capacity import FamilyCapacityAudit
from duraseed.runners import (
    Action,
    RunPlan,
    RunnerGateError,
    render_preflight,
    validate_mock_output_root,
)
from duraseed.runtime import (
    RuntimeBundle,
    SamplingCoordinates,
    SamplingTask,
    TokenLedger,
    sample_seeded,
)
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.tasks.tces import TCESGeneratorConfig, render_prompt


@dataclass(frozen=True, slots=True)
class BoundaryBlockInputs:
    cohort_id: str
    broad_manifest: DatasetManifest
    family_successes: dict[str, int]
    refinement_summaries: tuple[BoundaryFamilySummary, ...]
    confirmation_summaries: tuple[BoundaryFamilySummary, ...]
    prior_broad_manifests: tuple[DatasetManifest, ...]
    prior_confirmation_manifests: tuple[DatasetManifest, ...]
    capacity_audits: tuple[FamilyCapacityAudit, ...] | None = None


@dataclass(frozen=True, slots=True)
class BoundaryBlockResult:
    cohort_id: str
    refined_family_ids: tuple[str, ...]
    confirmation_manifest: DatasetManifest
    evidence: ConfirmationEvidence
    capacity_audits: tuple[FamilyCapacityAudit, ...]


@dataclass(frozen=True, slots=True)
class BoundaryExtensionResult:
    extension1: BoundaryBlockResult
    extension2: BoundaryBlockResult
    composite_status: str


async def sample_and_summarize(
    runtime: RuntimeBundle,
    sampler: Any,
    sample_manifest: DatasetManifest,
    coordinates: SamplingCoordinates,
    *,
    samples_per_item: int,
    informative_group_size: int,
    sample_index_start: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    ledger: TokenLedger,
    existing_generations: tuple[GenerationRecord, ...] = (),
    existing_rewards: tuple[RewardRecord, ...] = (),
    expected_run_ids: tuple[str, ...] | None = None,
    summary_manifest: DatasetManifest | None = None,
    task_ids: frozenset[str] | None = None,
) -> tuple[BoundaryFamilySummary, ...]:
    generations = list(existing_generations)
    rewards = list(existing_rewards)
    records = tuple(
        row
        for row in sample_manifest.records
        if task_ids is None or row.task_id in task_ids
    )
    if not records or (
        task_ids is not None and {row.task_id for row in records} != task_ids
    ):
        raise RunnerGateError("sampling task IDs are empty or absent from the manifest")
    for record in records:
        rows = await sample_seeded(
            runtime,
            sampler,
            SamplingTask(
                sample_manifest.manifest_id,
                record.task_id,
                "tces",
                record.split,
                render_prompt(record.to_task()),
                record.to_task(),
                record.item_index,
                record.intended_family,
                "calibration-only",
            ),
            coordinates,
            group_size=samples_per_item,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            ledger=ledger,
            sample_index_start=sample_index_start,
        )
        generations.extend(row.generation for row in rows)
        rewards.extend(row.reward for row in rows)
    return summarize_m0_boundary(
        summary_manifest or sample_manifest,
        generations,
        rewards,
        group_size=informative_group_size,
        expected_run_ids=expected_run_ids or (coordinates.run_id,),
        additional_manifests=(sample_manifest,) if summary_manifest else (),
    )


def build_extension2_manifest(generator_config: TCESGeneratorConfig) -> DatasetManifest:
    return build_broad_manifest(
        generator_config, cohort=BOUNDARY_BROAD_EXTENSION_2_COHORT
    )


def build_plan() -> RunPlan:
    return RunPlan(
        name="boundary-extension",
        actions=(
            Action("freeze-extension2-manifest", Decimal("0"), remote=False),
            Action(
                "extension1-confirm", Decimal("40"), ("freeze-extension2-manifest",)
            ),
            Action("extension2-broad", Decimal("10"), ("freeze-extension2-manifest",)),
            Action("extension2-refine", Decimal("30"), ("extension2-broad",)),
            Action("extension2-confirm", Decimal("40"), ("extension2-refine",)),
            Action(
                "three-cohort-composite",
                Decimal("0"),
                ("extension1-confirm", "extension2-confirm"),
                remote=False,
            ),
        ),
        launch_preconditions=(
            "live_smoke_passed",
            "boundary_extension_human_approval",
            "extension1_source_authenticated",
            "remaining_balance_verified",
        ),
        dry_run_command="uv run duraseed boundary-extension --dry-run",
        mock_command="uv run pytest tests/unit/test_boundary_extension_flow.py",
        authorization_command=(
            "uv run duraseed boundary-extension --authorize "
            "--authorized-cost-usd 120 --confirm-live-smoke "
            "--confirm-boundary-extension-approval --confirm-source-authenticated "
            "--confirm-remaining-balance"
        ),
    )


def reduce_block(
    generator_config: TCESGeneratorConfig,
    inputs: BoundaryBlockInputs,
) -> BoundaryBlockResult:
    observed_cohort, _ = broad_cohort(inputs.broad_manifest)
    if observed_cohort != inputs.cohort_id or inputs.cohort_id not in {
        BOUNDARY_BROAD_EXTENSION_1_COHORT,
        BOUNDARY_BROAD_EXTENSION_2_COHORT,
    }:
        raise RunnerGateError("broad manifest differs from the requested extension")
    audit_new_broad_cohort(
        inputs.broad_manifest,
        inputs.prior_broad_manifests,
        inputs.prior_confirmation_manifests,
    )
    broad_families = {row.intended_family for row in inputs.broad_manifest.records}
    if set(inputs.family_successes) != broad_families:
        raise RunnerGateError("broad family-success coverage is incomplete")
    candidates, audit = choose_refinement_family_ids(inputs.family_successes)
    refined = tuple(sorted((*candidates, *audit)))
    summaries = {row.intended_family_id: row for row in inputs.refinement_summaries}
    if set(summaries) != set(refined):
        raise RunnerGateError("refinement summaries differ from the frozen selection")
    finalists = tuple(
        sorted(
            family_id
            for family_id in refined
            if assess_refinement_finalist_gate(summaries[family_id]).eligible
        )
    )
    carried = inputs.capacity_audits
    if carried is not None and tuple(row.family_id for row in carried) != finalists:
        raise RunnerGateError("carried capacity audits differ from Stage-2 finalists")
    provisional = build_confirmation_manifest(
        generator_config, inputs.broad_manifest, finalists
    )
    templates = (
        regenerate_family_templates(generator_config, inputs.broad_manifest, finalists)
        if finalists
        else {}
    )
    forbidden = (*inputs.broad_manifest.records, *provisional.records)
    if carried is None:
        carried = audit_family_split_capacities(
            templates,
            finalists,
            generator_config,
            root_seed=BOUNDARY_ENGINEERING_SEED,
            forbidden_records=forbidden,
            protected_family_ids=finalists,
        )
    cleared = tuple(row.family_id for row in carried if row.passed)
    confirmation = build_confirmation_manifest(
        generator_config, inputs.broad_manifest, cleared
    )
    evidence = reduce_confirmation_evidence(
        inputs.refinement_summaries,
        inputs.confirmation_summaries,
        refined,
        carried,
    )
    return BoundaryBlockResult(
        inputs.cohort_id, refined, confirmation, evidence, carried
    )


def run_mock(
    generator_config: TCESGeneratorConfig,
    extension1: BoundaryBlockInputs,
    extension2: BoundaryBlockInputs,
    *,
    output_root: str | Path | None = None,
) -> BoundaryExtensionResult:
    validate_mock_output_root(output_root)
    if extension2.broad_manifest != build_extension2_manifest(generator_config):
        raise RunnerGateError(
            "Extension-2 differs from its frozen deterministic manifest"
        )
    first = reduce_block(generator_config, extension1)
    if (
        extension2.prior_broad_manifests[-1] != extension1.broad_manifest
        or extension2.prior_confirmation_manifests[-1] != first.confirmation_manifest
    ):
        raise RunnerGateError("Extension-2 does not consume Extension-1 confirmation")
    second = reduce_block(generator_config, extension2)
    try:
        freeze_three_cohort_panels(
            (),
            (),
            panel_size=12,
            allocation_seed=6448342238137851489,
            training_seeds=(17, 37),
            m0_checkpoint_path="mock://m0",
        )
    except BoundaryFreezeUnverifiedError:
        status = "blocked_pending_three_cohort_equivalence"
    else:  # pragma: no cover
        raise RunnerGateError("three-cohort composite unexpectedly became available")
    return BoundaryExtensionResult(first, second, status)


def preflight_text() -> str:
    plan = build_plan()
    return render_preflight(
        "Boundary extension plan",
        plan,
        suffix=(f"Composite status: {BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS}",),
    )
