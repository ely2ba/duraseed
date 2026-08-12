"""Pure Phase-4 teacher-dose summaries, gates, and two-seed decision."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import fsum, isfinite, sqrt

from duraseed.config import TeacherDoseConfig
from duraseed.data.manifests import DatasetManifest, TCESTaskManifestRecord
from duraseed.data.panels import FamilyPanelArtifact, PanelLabel
from duraseed.evaluation.analysis import jeffreys_posterior_mean
from duraseed.provenance import (
    MAX_ROOT_SEED,
    canonical_json_hash,
    validate_sha256_id,
)
from duraseed.run_records import GenerationRecord, RewardRecord
from duraseed.tasks.tces import render_prompt as render_tces_prompt
from duraseed.training.reward import verify_task_completion


class TeacherDoseEvidenceError(ValueError):
    """Raised when dose evidence is incomplete or internally inconsistent."""


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


class TeacherDoseDecisionStatus(StrEnum):
    CALIBRATION_INCOMPLETE = "calibration_incomplete"
    CALIBRATION_UNRESOLVED = "calibration_unresolved"
    NO_EFFECTIVE_DOSE = "no_effective_dose"
    VERIFICATION_REQUIRED = "verification_required"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_UNRESOLVED = "verification_unresolved"
    SELECTED = "selected"


def _probability(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{name} must be a finite probability")
    return float(value)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class PairedControlChange:
    """Equal-item seeded-minus-M0 change with an approximate paired interval."""

    item_count: int
    mean_change: float
    standard_error: float
    approximate_95_interval: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.item_count, bool)
            or not isinstance(self.item_count, int)
            or self.item_count < 2
        ):
            raise ValueError("paired control change requires at least two items")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            for value in (
                self.mean_change,
                self.standard_error,
                *self.approximate_95_interval,
            )
        ):
            raise ValueError("paired control change values must be finite")
        if not -1 <= self.mean_change <= 1 or self.standard_error < 0:
            raise ValueError("invalid paired control change or standard error")
        lower, upper = self.approximate_95_interval
        if not -1 <= lower <= self.mean_change <= upper <= 1:
            raise ValueError("paired interval must contain the mean within [-1, 1]")


def summarize_paired_control_change(
    m0_item_success_rates: Sequence[float],
    seeded_item_success_rates: Sequence[float],
) -> PairedControlChange:
    """Summarize identical sentinel items without claiming noninferiority."""

    m0 = tuple(
        _probability(value, "m0_item_success_rate") for value in m0_item_success_rates
    )
    seeded = tuple(
        _probability(value, "seeded_item_success_rate")
        for value in seeded_item_success_rates
    )
    if len(m0) != len(seeded) or len(m0) < 2:
        raise TeacherDoseEvidenceError(
            "paired control rates must have equal length of at least two"
        )
    differences = tuple(
        after - before for before, after in zip(m0, seeded, strict=True)
    )
    mean = fsum(differences) / len(differences)
    variance = fsum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    standard_error = sqrt(variance / len(differences))
    radius = 1.96 * standard_error
    return PairedControlChange(
        item_count=len(differences),
        mean_change=mean,
        standard_error=standard_error,
        approximate_95_interval=(max(-1.0, mean - radius), min(1.0, mean + radius)),
    )


@dataclass(frozen=True, slots=True)
class TeacherDoseGateSummary:
    """Authenticated aggregate for one dose, checkpoint, and training seed."""

    demonstrations_per_family: int
    training_seed: int
    run_id: str
    m0_sampler_checkpoint_path: str
    seeded_sampler_checkpoint_path: str
    gate_manifest_id: str
    panel_artifact_id: str
    targeted_family_count: int
    targeted_item_count: int
    equal_item_posterior_success: float
    mixed_group_rate: float
    all_zero_group_rate: float
    all_one_group_rate: float
    family_reachability: float
    format_compliance: float
    sentinel_control_change: PairedControlChange
    sampled_token_mean_surprisal: float | None
    sampled_token_logprob_coverage: float

    def __post_init__(self) -> None:
        for name, value in (
            ("demonstrations_per_family", self.demonstrations_per_family),
            ("targeted_family_count", self.targeted_family_count),
            ("targeted_item_count", self.targeted_item_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.training_seed, bool)
            or not isinstance(self.training_seed, int)
            or not 0 <= self.training_seed <= MAX_ROOT_SEED
        ):
            raise ValueError("training_seed is outside the repository seed range")
        for name in (
            "run_id",
            "m0_sampler_checkpoint_path",
            "seeded_sampler_checkpoint_path",
        ):
            _nonempty(getattr(self, name), name)
        validate_sha256_id(self.gate_manifest_id)
        validate_sha256_id(self.panel_artifact_id)
        for name in (
            "equal_item_posterior_success",
            "mixed_group_rate",
            "all_zero_group_rate",
            "all_one_group_rate",
            "family_reachability",
            "format_compliance",
            "sampled_token_logprob_coverage",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if (
            abs(
                self.mixed_group_rate
                + self.all_zero_group_rate
                + self.all_one_group_rate
                - 1.0
            )
            > 1e-12
        ):
            raise ValueError("reward-group mode rates must sum to one")
        if self.sampled_token_mean_surprisal is not None and (
            isinstance(self.sampled_token_mean_surprisal, bool)
            or not isinstance(self.sampled_token_mean_surprisal, (int, float))
            or not isfinite(self.sampled_token_mean_surprisal)
            or self.sampled_token_mean_surprisal < 0
        ):
            raise ValueError(
                "sampled_token_mean_surprisal must be finite and nonnegative"
            )
        if (
            self.sampled_token_logprob_coverage == 0
            and self.sampled_token_mean_surprisal is not None
        ) or (
            self.sampled_token_logprob_coverage > 0
            and self.sampled_token_mean_surprisal is None
        ):
            raise ValueError(
                "surprisal must be present exactly when logprobs are observed"
            )
        if not isinstance(self.sentinel_control_change, PairedControlChange):
            raise TypeError("sentinel_control_change must be PairedControlChange")


def _panel_roles(
    artifact: FamilyPanelArtifact,
    training_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    assignments = {
        assignment.training_seed: assignment
        for assignment in artifact.seed_block_assignments
    }
    try:
        assignment = assignments[training_seed]
    except KeyError as error:
        raise TeacherDoseEvidenceError(
            "training seed is absent from the frozen panel schedule"
        ) from error
    panels = {
        PanelLabel.A: artifact.panel_a_family_ids,
        PanelLabel.B: artifact.panel_b_family_ids,
    }
    return panels[assignment.targeted_panel], panels[assignment.sentinel_panel]


def _authenticated_gate_rows(
    manifest: DatasetManifest,
    generations: Sequence[GenerationRecord],
    rewards: Sequence[RewardRecord],
    *,
    expected_family_ids: tuple[str, ...],
    expected_panel_role: str,
    expected_samples_per_item: int,
    expected_checkpoint_stage: str,
    expected_checkpoint_path: str,
    expected_origin_path: str,
    expected_training_seed: int,
) -> dict[str, tuple[tuple[GenerationRecord, RewardRecord], ...]]:
    records = {
        record.task_id: record
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
        and record.intended_family in expected_family_ids
    }
    if not records:
        raise TeacherDoseEvidenceError("gate manifest has no requested panel records")
    reward_by_sample = {reward.sample_id: reward for reward in rewards}
    if len(reward_by_sample) != len(tuple(rewards)):
        raise TeacherDoseEvidenceError("gate rewards contain duplicate sample IDs")
    by_task: dict[str, list[tuple[GenerationRecord, RewardRecord]]] = defaultdict(list)
    run_ids: set[str] = set()
    training_steps: set[int] = set()
    settings: set[tuple[float | None, float | None, int | None]] = set()
    coordinates: set[tuple[str, int]] = set()
    for generation in generations:
        record = records.get(generation.task_id)
        if record is None:
            raise TeacherDoseEvidenceError(
                "gate generation is absent from its requested panel manifest"
            )
        if generation.sample_id not in reward_by_sample:
            raise TeacherDoseEvidenceError("gate generation has no exact reward row")
        reward = reward_by_sample[generation.sample_id]
        if (
            generation.task_manifest_id != manifest.manifest_id
            or generation.task_family != "tces"
            or generation.source_split != "a_seed_gate"
            or generation.purpose != "evaluation"
            or generation.checkpoint_stage != expected_checkpoint_stage
            or generation.sampler_checkpoint_path != expected_checkpoint_path
            or generation.origin_sampler_checkpoint_path != expected_origin_path
            or generation.seed != expected_training_seed
            or generation.item_index != record.item_index
            or generation.assigned_family_id != record.intended_family
            or generation.panel_role != expected_panel_role
            or generation.sampling_seed is None
            or generation.stop_reason is None
            or generation.completion_token_ids is None
        ):
            raise TeacherDoseEvidenceError(
                "gate generation identity or raw sampling evidence is inconsistent"
            )
        if generation.run_id is None:
            raise TeacherDoseEvidenceError("gate generation omitted its run ID")
        if not 0 <= generation.sample_index < expected_samples_per_item:
            raise TeacherDoseEvidenceError("gate sample index is outside its grid")
        coordinate = (generation.task_id, generation.sample_index)
        if coordinate in coordinates:
            raise TeacherDoseEvidenceError("gate sample coordinate is duplicated")
        coordinates.add(coordinate)
        if (
            reward.task_id != generation.task_id
            or generation.reward != reward.reward
            or generation.family_id != reward.exact_verification.strategy_family_id
            or verify_task_completion(generation.completion_text, record.to_task())
            != reward.exact_verification
        ):
            raise TeacherDoseEvidenceError(
                "gate reward differs from authoritative exact verification"
            )
        if generation.prompt_text != render_tces_prompt(record.to_task()):
            raise TeacherDoseEvidenceError("gate prompt differs from its exact task")
        run_ids.add(generation.run_id)
        training_steps.add(generation.training_step)
        settings.add(
            (
                generation.sampling_temperature,
                generation.sampling_top_p,
                generation.sampling_max_tokens,
            )
        )
        by_task[generation.task_id].append((generation, reward))
    expected_count = len(records) * expected_samples_per_item
    if (
        len(generations) != expected_count
        or len(rewards) != expected_count
        or set(by_task) != set(records)
        or any(len(rows) != expected_samples_per_item for rows in by_task.values())
        or len(run_ids) != 1
        or len(training_steps) != 1
        or len(settings) != 1
    ):
        raise TeacherDoseEvidenceError("gate evidence grid is incomplete or mixed")
    for rows in by_task.values():
        if tuple(sorted(row[0].sample_index for row in rows)) != tuple(
            range(expected_samples_per_item)
        ):
            raise TeacherDoseEvidenceError("gate item sample grid is incomplete")
    return {
        task_id: tuple(sorted(rows, key=lambda row: row[0].sample_index))
        for task_id, rows in by_task.items()
    }


def summarize_teacher_dose_gate(
    *,
    demonstrations_per_family: int,
    training_seed: int,
    panel_artifact: FamilyPanelArtifact,
    gate_manifest: DatasetManifest,
    m0_sampler_checkpoint_path: str,
    seeded_sampler_checkpoint_path: str,
    targeted_generations: Sequence[GenerationRecord],
    targeted_rewards: Sequence[RewardRecord],
    sentinel_m0_generations: Sequence[GenerationRecord],
    sentinel_m0_rewards: Sequence[RewardRecord],
    sentinel_seeded_generations: Sequence[GenerationRecord],
    sentinel_seeded_rewards: Sequence[RewardRecord],
    group_size: int = 8,
) -> TeacherDoseGateSummary:
    """Authenticate and reduce one post-seed target gate plus paired controls."""

    if not isinstance(panel_artifact, FamilyPanelArtifact):
        raise TypeError("panel_artifact must be FamilyPanelArtifact")
    if not isinstance(gate_manifest, DatasetManifest):
        raise TypeError("gate_manifest must be DatasetManifest")
    if (
        gate_manifest.task_family != "tces"
        or gate_manifest.split != "a_seed_gate"
        or not gate_manifest.records
        or not all(
            isinstance(record, TCESTaskManifestRecord)
            for record in gate_manifest.records
        )
    ):
        raise TeacherDoseEvidenceError(
            "teacher-dose gate requires a nonempty TCES a_seed_gate manifest"
        )
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 2
    ):
        raise ValueError("group_size must be an integer of at least two")
    targeted_families, sentinel_families = _panel_roles(panel_artifact, training_seed)
    all_families = {record.intended_family for record in gate_manifest.records}
    if all_families != set(targeted_families).union(sentinel_families):
        raise TeacherDoseEvidenceError(
            "gate manifest family set differs from the frozen panels"
        )
    counts: dict[str, int] = defaultdict(int)
    for record in gate_manifest.records:
        counts[record.intended_family] += 1
    if set(counts.values()) != {8}:
        raise TeacherDoseEvidenceError(
            "teacher-dose gate requires exactly eight items per panel family"
        )
    if panel_artifact.m0_checkpoint_path != m0_sampler_checkpoint_path:
        raise TeacherDoseEvidenceError("teacher-dose evidence changed the frozen M0")

    targeted = _authenticated_gate_rows(
        gate_manifest,
        targeted_generations,
        targeted_rewards,
        expected_family_ids=targeted_families,
        expected_panel_role="targeted",
        expected_samples_per_item=group_size,
        expected_checkpoint_stage="stage_a",
        expected_checkpoint_path=seeded_sampler_checkpoint_path,
        expected_origin_path=m0_sampler_checkpoint_path,
        expected_training_seed=training_seed,
    )
    sentinel_m0 = _authenticated_gate_rows(
        gate_manifest,
        sentinel_m0_generations,
        sentinel_m0_rewards,
        expected_family_ids=sentinel_families,
        expected_panel_role="sentinel",
        expected_samples_per_item=1,
        expected_checkpoint_stage="m0",
        expected_checkpoint_path=m0_sampler_checkpoint_path,
        expected_origin_path=m0_sampler_checkpoint_path,
        expected_training_seed=training_seed,
    )
    sentinel_seeded = _authenticated_gate_rows(
        gate_manifest,
        sentinel_seeded_generations,
        sentinel_seeded_rewards,
        expected_family_ids=sentinel_families,
        expected_panel_role="sentinel",
        expected_samples_per_item=1,
        expected_checkpoint_stage="stage_a",
        expected_checkpoint_path=seeded_sampler_checkpoint_path,
        expected_origin_path=m0_sampler_checkpoint_path,
        expected_training_seed=training_seed,
    )
    if set(sentinel_m0) != set(sentinel_seeded):
        raise TeacherDoseEvidenceError("paired sentinel item sets differ")
    for task_id in sentinel_m0:
        before = sentinel_m0[task_id][0][0]
        after = sentinel_seeded[task_id][0][0]
        if (
            before.sample_index != after.sample_index
            or before.sampling_seed != after.sampling_seed
            or (
                before.sampling_temperature,
                before.sampling_top_p,
                before.sampling_max_tokens,
            )
            != (
                after.sampling_temperature,
                after.sampling_top_p,
                after.sampling_max_tokens,
            )
        ):
            raise TeacherDoseEvidenceError(
                "sentinel observations are not paired on sampling coordinates"
            )

    task_successes: list[int] = []
    mixed = all_zero = all_one = valid_tags = 0
    reached_families: set[str] = set()
    observed_logprobs: list[float] = []
    completion_token_count = 0
    record_by_task = {record.task_id: record for record in gate_manifest.records}
    for task_id, rows in targeted.items():
        successes = sum(int(reward.reward) for _, reward in rows)
        task_successes.append(successes)
        if successes == 0:
            all_zero += 1
        elif successes == group_size:
            all_one += 1
        else:
            mixed += 1
        if successes:
            reached_families.add(record_by_task[task_id].intended_family)
        valid_tags += sum(
            int(reward.exact_verification.valid_answer_tag) for _, reward in rows
        )
        for generation, _ in rows:
            assert generation.completion_token_ids is not None
            completion_token_count += len(generation.completion_token_ids)
            for value in generation.completion_logprobs or ():
                if value is not None:
                    if not isfinite(value):
                        raise TeacherDoseEvidenceError(
                            "teacher-dose log-probability is non-finite"
                        )
                    observed_logprobs.append(value)
    targeted_item_count = len(targeted)
    total_targeted_samples = targeted_item_count * group_size
    item_means = tuple(
        jeffreys_posterior_mean(successes, group_size) for successes in task_successes
    )
    m0_rates = [sentinel_m0[key][0][1].reward for key in sorted(sentinel_m0)]
    seeded_rates = [
        sentinel_seeded[key][0][1].reward for key in sorted(sentinel_seeded)
    ]
    run_ids = {rows[0][0].run_id for rows in targeted.values()}
    return TeacherDoseGateSummary(
        demonstrations_per_family=demonstrations_per_family,
        training_seed=training_seed,
        run_id=str(next(iter(run_ids))),
        m0_sampler_checkpoint_path=m0_sampler_checkpoint_path,
        seeded_sampler_checkpoint_path=seeded_sampler_checkpoint_path,
        gate_manifest_id=gate_manifest.manifest_id,
        panel_artifact_id=canonical_json_hash(panel_artifact.model_dump(mode="json")),
        targeted_family_count=len(targeted_families),
        targeted_item_count=targeted_item_count,
        equal_item_posterior_success=fsum(item_means) / len(item_means),
        mixed_group_rate=mixed / targeted_item_count,
        all_zero_group_rate=all_zero / targeted_item_count,
        all_one_group_rate=all_one / targeted_item_count,
        family_reachability=len(reached_families) / len(targeted_families),
        format_compliance=valid_tags / total_targeted_samples,
        sentinel_control_change=summarize_paired_control_change(m0_rates, seeded_rates),
        sampled_token_mean_surprisal=(
            -fsum(observed_logprobs) / len(observed_logprobs)
            if observed_logprobs
            else None
        ),
        sampled_token_logprob_coverage=(
            len(observed_logprobs) / completion_token_count
            if completion_token_count
            else 0.0
        ),
    )


@dataclass(frozen=True, slots=True)
class TeacherDoseCriterion:
    name: str
    status: GateStatus
    observed: float | bool | None
    threshold: float | bool
    reason: str


@dataclass(frozen=True, slots=True)
class TeacherDoseAssessment:
    summary: TeacherDoseGateSummary
    status: GateStatus
    criteria: tuple[TeacherDoseCriterion, ...]


def _criterion(
    name: str,
    observed: float,
    threshold: float,
    *,
    minimum: bool,
) -> TeacherDoseCriterion:
    passed = observed >= threshold if minimum else observed <= threshold
    comparator = ">=" if minimum else "<="
    return TeacherDoseCriterion(
        name=name,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        observed=observed,
        threshold=threshold,
        reason=f"observed {observed:.6g} {comparator} {threshold:.6g}"
        if passed
        else f"observed {observed:.6g} violates {comparator} {threshold:.6g}",
    )


def _evidence_criterion(
    name: str, observed: bool | None, *, required: bool = True
) -> TeacherDoseCriterion:
    status = (
        GateStatus.UNRESOLVED
        if observed is None
        else GateStatus.PASSED
        if observed is required
        else GateStatus.FAILED
    )
    return TeacherDoseCriterion(
        name=name,
        status=status,
        observed=observed,
        threshold=required,
        reason=(
            "evidence not yet supplied"
            if observed is None
            else "required evidence satisfied"
            if status is GateStatus.PASSED
            else "required evidence failed"
        ),
    )


def assess_teacher_dose(
    summary: TeacherDoseGateSummary,
    config: TeacherDoseConfig,
    *,
    optimization_stable: bool | None,
    leakage_clean: bool | None,
    maximum_catastrophic_control_drop: float | None = None,
) -> TeacherDoseAssessment:
    """Apply the declared gates without treating a broad safeguard as equivalence."""

    if not isinstance(summary, TeacherDoseGateSummary):
        raise TypeError("summary must be TeacherDoseGateSummary")
    if not isinstance(config, TeacherDoseConfig):
        raise TypeError("config must be TeacherDoseConfig")
    if summary.demonstrations_per_family not in config.demonstrations_per_family:
        raise ValueError("summary dose is absent from the configured grid")
    drop_limit = _probability(
        config.maximum_catastrophic_control_drop
        if maximum_catastrophic_control_drop is None
        else maximum_catastrophic_control_drop,
        "maximum_catastrophic_control_drop",
    )
    criteria = (
        _criterion(
            "mixed_group_rate",
            summary.mixed_group_rate,
            config.minimum_mixed_group_rate,
            minimum=True,
        ),
        _criterion(
            "family_reachability",
            summary.family_reachability,
            config.minimum_family_reachability,
            minimum=True,
        ),
        _criterion(
            "saturated_group_rate",
            summary.all_one_group_rate,
            config.maximum_saturated_group_rate,
            minimum=False,
        ),
        _criterion(
            "format_compliance",
            summary.format_compliance,
            config.required_format_compliance,
            minimum=True,
        ),
        _criterion(
            "catastrophic_sentinel_drop",
            -summary.sentinel_control_change.mean_change,
            drop_limit,
            minimum=False,
        ),
        _evidence_criterion("optimization_stable", optimization_stable),
        _evidence_criterion("leakage_clean", leakage_clean),
    )
    statuses = {criterion.status for criterion in criteria}
    status = (
        GateStatus.FAILED
        if GateStatus.FAILED in statuses
        else GateStatus.UNRESOLVED
        if GateStatus.UNRESOLVED in statuses
        else GateStatus.PASSED
    )
    return TeacherDoseAssessment(summary=summary, status=status, criteria=criteria)


@dataclass(frozen=True, slots=True)
class TeacherDoseDecision:
    status: TeacherDoseDecisionStatus
    calibration_seed: int | None
    verification_seed: int | None
    candidate_dose: int | None
    selected_dose: int | None
    reason: str


def decide_teacher_dose(
    dose_grid: Sequence[int],
    calibration_assessments: Sequence[TeacherDoseAssessment],
    *,
    verification_assessment: TeacherDoseAssessment | None = None,
) -> TeacherDoseDecision:
    """Choose the smallest passing dose, then require a distinct-seed verification."""

    grid = tuple(dose_grid)
    if (
        not grid
        or any(
            isinstance(dose, bool) or not isinstance(dose, int) or dose < 1
            for dose in grid
        )
        or grid != tuple(sorted(set(grid)))
    ):
        raise ValueError("dose_grid must contain unique increasing positive integers")
    assessments = tuple(calibration_assessments)
    if any(not isinstance(value, TeacherDoseAssessment) for value in assessments):
        raise TypeError("calibration_assessments must contain TeacherDoseAssessment")
    by_dose = {value.summary.demonstrations_per_family: value for value in assessments}
    if len(by_dose) != len(assessments) or not set(by_dose).issubset(grid):
        raise TeacherDoseEvidenceError("calibration doses are duplicate or off-grid")
    seeds = {value.summary.training_seed for value in assessments}
    if len(seeds) > 1:
        raise TeacherDoseEvidenceError("calibration assessments must share one seed")
    calibration_seed = next(iter(seeds), None)

    candidate: TeacherDoseAssessment | None = None
    for dose in grid:
        assessment = by_dose.get(dose)
        if assessment is None:
            return TeacherDoseDecision(
                status=TeacherDoseDecisionStatus.CALIBRATION_INCOMPLETE,
                calibration_seed=calibration_seed,
                verification_seed=None,
                candidate_dose=None,
                selected_dose=None,
                reason=f"dose {dose} has not been assessed",
            )
        if assessment.status is GateStatus.UNRESOLVED:
            return TeacherDoseDecision(
                status=TeacherDoseDecisionStatus.CALIBRATION_UNRESOLVED,
                calibration_seed=calibration_seed,
                verification_seed=None,
                candidate_dose=None,
                selected_dose=None,
                reason=f"dose {dose} has unresolved gate evidence",
            )
        if assessment.status is GateStatus.PASSED:
            candidate = assessment
            break
    if candidate is None:
        return TeacherDoseDecision(
            status=TeacherDoseDecisionStatus.NO_EFFECTIVE_DOSE,
            calibration_seed=calibration_seed,
            verification_seed=None,
            candidate_dose=None,
            selected_dose=None,
            reason="every configured dose failed at the calibration seed",
        )

    candidate_dose = candidate.summary.demonstrations_per_family
    if verification_assessment is None:
        return TeacherDoseDecision(
            status=TeacherDoseDecisionStatus.VERIFICATION_REQUIRED,
            calibration_seed=calibration_seed,
            verification_seed=None,
            candidate_dose=candidate_dose,
            selected_dose=None,
            reason="the smallest calibration-passing dose requires a second seed",
        )
    verification = verification_assessment
    if verification.summary.demonstrations_per_family != candidate_dose:
        raise TeacherDoseEvidenceError("verification must assess the candidate dose")
    if verification.summary.training_seed == calibration_seed:
        raise TeacherDoseEvidenceError("verification seed must differ from calibration")
    if verification.status is GateStatus.UNRESOLVED:
        status = TeacherDoseDecisionStatus.VERIFICATION_UNRESOLVED
        selected = None
        reason = "candidate-dose verification has unresolved evidence"
    elif verification.status is GateStatus.FAILED:
        status = TeacherDoseDecisionStatus.VERIFICATION_FAILED
        selected = None
        reason = "candidate dose failed verification; no automatic escalation"
    else:
        status = TeacherDoseDecisionStatus.SELECTED
        selected = candidate_dose
        reason = "smallest calibration-passing dose passed a distinct verification seed"
    return TeacherDoseDecision(
        status=status,
        calibration_seed=calibration_seed,
        verification_seed=verification.summary.training_seed,
        candidate_dose=candidate_dose,
        selected_dose=selected,
        reason=reason,
    )


__all__ = [
    "GateStatus",
    "PairedControlChange",
    "TeacherDoseAssessment",
    "TeacherDoseCriterion",
    "TeacherDoseDecision",
    "TeacherDoseDecisionStatus",
    "TeacherDoseEvidenceError",
    "TeacherDoseGateSummary",
    "assess_teacher_dose",
    "decide_teacher_dose",
    "summarize_paired_control_change",
    "summarize_teacher_dose_gate",
]
