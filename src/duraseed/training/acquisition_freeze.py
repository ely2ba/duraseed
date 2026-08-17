"""Pure final decisions for the common acquisition-calibration freeze."""

from __future__ import annotations

from dataclasses import dataclass

from duraseed.config import PilotConfig
from duraseed.provenance import validate_sha256_id
from duraseed.runners import RunnerGateError
from duraseed.training.stage_a_calibration import (
    StageADurationDecision,
    StageADurationDecisionStatus,
    StageAFinalEvidence,
    StageALearningRateDecision,
    StageALearningRateDecisionStatus,
    StageAScreenEvidence,
    decide_stage_a_duration,
)
from duraseed.training.stage_a_direct import select_direct_m0_learning_rate
from duraseed.training.teacher_dose import (
    GateStatus,
    TeacherDoseAssessment,
    TeacherDoseDecision,
    TeacherDoseDecisionStatus,
    decide_teacher_dose,
)


ALL_METHODS = ("G-U", "G-B", "R-G", "B-S", "B-O", "B-G")
COMMON_RL_METHODS = ("G-U", "G-B", "R-G", "B-G")
FROZEN_PROTOCOL_MAX_TOKENS = 4096
PROVISIONAL_ACQUISITION_MAX_TOKENS = 256
Scalar = str | int | float | bool


@dataclass(frozen=True, slots=True)
class TeacherDoseArmEvidence:
    learning_rate: float
    assessment: TeacherDoseAssessment


@dataclass(frozen=True, slots=True)
class TeacherDoseLiveEvidence:
    calibration_arms: tuple[TeacherDoseArmEvidence, ...]
    verification_arm: TeacherDoseArmEvidence


@dataclass(frozen=True, slots=True)
class TeacherDoseRecipe:
    decision: TeacherDoseDecision
    selected_learning_rate: float
    evidence: TeacherDoseLiveEvidence
    selected_updates: int = 16


@dataclass(frozen=True, slots=True)
class StageALiveEvidence:
    bs_screens: tuple[StageAScreenEvidence, ...]
    bg_screens: tuple[StageAScreenEvidence, ...]
    final_evidence: tuple[StageAFinalEvidence, ...]


@dataclass(frozen=True, slots=True)
class LiveSmokeCalibrationEvidence:
    artifact_sha256: str
    protocol_max_tokens: int
    sample_count: int
    truncated_count: int
    runtime_diagnostic_passed: bool

    def __post_init__(self) -> None:
        validate_sha256_id(self.artifact_sha256)
        valid_counts = (
            type(self.sample_count) is int
            and type(self.truncated_count) is int
            and 0 <= self.truncated_count <= self.sample_count
            and self.sample_count > 0
        )
        if (
            not valid_counts
            or type(self.protocol_max_tokens) is not int
            or self.protocol_max_tokens != FROZEN_PROTOCOL_MAX_TOKENS
        ):
            raise ValueError("invalid live-smoke max-token evidence")
        if self.runtime_diagnostic_passed is not True:
            raise RunnerGateError(
                "live-smoke runtime max-token diagnostic did not pass"
            )


MAX_TOKEN_CANDIDATE_LADDER = (256, 512, 1024, 2048, 4096)
MAX_TOKEN_OBSERVED_CAPS = (512, 1024, 2048, 4096)
MAX_TOKEN_REFERENCE_RUN_ID = "tces-cap-reference-20260809T204337Z"
MAX_TOKEN_SELECTION_RULE = (
    "smallest_cap_censoring_at_most_5_percent_of_all_items_and_no_exact_successes"
)
MAX_TOKEN_SUMMARY_SHA256 = (
    "sha256:d2505e16bec8dc4374e4a5917d80a71095018407ab5d95937c06dd7997e9ae76"
)
MAX_TOKEN_GENERATIONS_SHA256 = (
    "sha256:a3854d5cfad49a75bcdb0e95b3e0cdaddbc2cf9685f54d88eb7d5cd2cc95ff70"
)
MAX_TOKEN_REWARDS_SHA256 = (
    "sha256:3e91a65617e97495d8646802fdab0da5f94c4bb5ca0922b39af13b3e84b285de"
)


@dataclass(frozen=True, slots=True)
class MaxTokenCapObservation:
    max_tokens: int
    censored_answer_count: int
    censored_exact_success_count: int
    passed: bool

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not int
                for value in (
                    self.max_tokens,
                    self.censored_answer_count,
                    self.censored_exact_success_count,
                )
            )
            or type(self.passed) is not bool
        ):
            raise ValueError("max-token cap observation has invalid field types")


@dataclass(frozen=True, slots=True)
class MaxTokenReferenceEvidence:
    summary_sha256: str
    generations_sha256: str
    rewards_sha256: str
    run_id: str
    sample_count: int
    source_split: str
    seed: int
    training_step: int
    reference_max_tokens: int
    answer_termination_count: int
    exact_success_count: int
    observations: tuple[MaxTokenCapObservation, ...]

    def __post_init__(self) -> None:
        expected_observations = tuple(
            MaxTokenCapObservation(*row)
            for row in (
                (512, 15, 3, False),
                (1024, 9, 2, False),
                (2048, 4, 1, False),
                (4096, 0, 0, True),
            )
        )
        if (
            self.summary_sha256 != MAX_TOKEN_SUMMARY_SHA256
            or self.generations_sha256 != MAX_TOKEN_GENERATIONS_SHA256
            or self.rewards_sha256 != MAX_TOKEN_REWARDS_SHA256
            or self.run_id != MAX_TOKEN_REFERENCE_RUN_ID
            or self.sample_count != 128
            or self.source_split != "a_validation"
            or self.seed != 5
            or self.training_step != 2
            or self.reference_max_tokens != FROZEN_PROTOCOL_MAX_TOKENS
            or self.answer_termination_count != 73
            or self.exact_success_count != 8
            or self.observations != expected_observations
        ):
            raise RunnerGateError(
                "max-token evidence differs from the frozen M0 source"
            )


@dataclass(frozen=True, slots=True)
class MaxTokenFreezeEvidence:
    specification_sha256: str
    evidence_sha256: str
    specification_authorization_sha256: str
    reference: MaxTokenReferenceEvidence
    candidate_max_tokens: tuple[int, ...]
    maximum_reference_censor_rate: float
    minimum_answer_terminations: int
    require_zero_censored_exact_successes: bool
    provisional_failure_inferred_from_max_tokens: int
    selected_max_tokens: int
    apply_to_methods: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_sha256_id(self.specification_sha256)
        validate_sha256_id(self.evidence_sha256)
        validate_sha256_id(self.specification_authorization_sha256)
        if (
            self.evidence_sha256 != self.reference.summary_sha256
            or self.candidate_max_tokens != MAX_TOKEN_CANDIDATE_LADDER
            or self.maximum_reference_censor_rate != 0.05
            or self.minimum_answer_terminations != 64
            or self.require_zero_censored_exact_successes is not True
            or self.provisional_failure_inferred_from_max_tokens != 512
            or self.selected_max_tokens != FROZEN_PROTOCOL_MAX_TOKENS
            or self.apply_to_methods != ALL_METHODS
        ):
            raise RunnerGateError(
                "max-token ratification differs from its proposed rule"
            )


def _validate_settings(values: tuple[tuple[str, Scalar], ...], label: str) -> None:
    names = tuple(name for name, _ in values)
    if (
        not values
        or names != tuple(sorted(set(names)))
        or any(not name.strip() for name in names)
        or any(type(value) not in {str, int, float, bool} for _, value in values)
    ):
        raise ValueError(f"{label} must be nonempty, unique, and key-sorted")


@dataclass(frozen=True, slots=True)
class CommonRLFreezeEvidence:
    configuration: tuple[tuple[str, Scalar], ...]
    entropy_collapse_detected: bool
    entropy_collapse_gate_passed: bool
    final_ten_mixed_group_rate: float
    mean_sampled_token_surprisal: float
    unique_completion_count: int
    successful_completion_count: int
    valid_family_count: int
    loss_health_passed: bool
    fallback_configuration: tuple[tuple[str, Scalar], ...] | None = None

    def __post_init__(self) -> None:
        _validate_settings(self.configuration, "RL configuration")
        if not any(name == "objective" for name, _ in self.configuration):
            raise RunnerGateError("common RL configuration must name its objective")
        if self.entropy_collapse_gate_passed is not True:
            raise RunnerGateError("entropy-collapse gate did not pass")
        if (
            not 0 <= self.final_ten_mixed_group_rate <= 1
            or self.mean_sampled_token_surprisal < 0
            or type(self.unique_completion_count) is not int
            or type(self.successful_completion_count) is not int
            or type(self.valid_family_count) is not int
            or min(
                self.unique_completion_count,
                self.successful_completion_count,
                self.valid_family_count,
            )
            < 0
            or type(self.loss_health_passed) is not bool
        ):
            raise ValueError("common RL entropy diagnostics are out of range")
        if self.entropy_collapse_detected:
            if self.fallback_configuration is None:
                raise RunnerGateError("entropy collapse requires one common fallback")
            _validate_settings(self.fallback_configuration, "RL fallback")
        elif self.fallback_configuration is not None:
            raise RunnerGateError("fallback cannot be selected without collapse")


@dataclass(frozen=True, slots=True)
class AcquisitionFreeze:
    learning_rates: dict[str, float]
    duration: StageADurationDecision
    selected_max_tokens: int
    max_tokens_apply_to: tuple[str, ...]
    live_smoke: LiveSmokeCalibrationEvidence
    max_token_gate: MaxTokenFreezeEvidence
    common_rl: CommonRLFreezeEvidence
    common_rl_apply_to: tuple[str, ...]


def select_teacher_recipe(
    config: PilotConfig, evidence: TeacherDoseLiveEvidence
) -> TeacherDoseRecipe:
    """Apply dose order and the same-dose smallest-LR tie-break."""

    grid = config.teacher_dose.demonstrations_per_family
    rates = tuple(sorted(config.tinker.learning_rates.teacher_seed_sft.grid))
    by_dose: dict[int, list[TeacherDoseArmEvidence]] = {}
    for arm in evidence.calibration_arms:
        dose = arm.assessment.summary.demonstrations_per_family
        by_dose.setdefault(dose, []).append(arm)
    if tuple(sorted(by_dose)) != grid[: len(by_dose)]:
        raise RunnerGateError("teacher-dose arms are not a contiguous grid prefix")
    representatives: list[TeacherDoseAssessment] = []
    selected: TeacherDoseArmEvidence | None = None
    for dose in sorted(by_dose):
        arms = by_dose[dose]
        if tuple(sorted(arm.learning_rate for arm in arms)) != rates:
            raise RunnerGateError("teacher-dose action omitted a learning-rate arm")
        passing = tuple(
            arm for arm in arms if arm.assessment.status is GateStatus.PASSED
        )
        chosen = (
            min(passing, key=lambda arm: arm.learning_rate)
            if passing
            else min(arms, key=lambda arm: arm.learning_rate)
        )
        representatives.append(chosen.assessment)
        if passing:
            selected = chosen
            break
    if selected is None or len(by_dose) != len(representatives):
        raise RunnerGateError(
            "teacher-dose action did not stop at its first passing dose"
        )
    verification = evidence.verification_arm
    if verification.learning_rate != selected.learning_rate:
        raise RunnerGateError("teacher-dose verification changed learning rate")
    decision = decide_teacher_dose(
        grid, representatives, verification_assessment=verification.assessment
    )
    if decision.status is not TeacherDoseDecisionStatus.SELECTED:
        raise RunnerGateError(f"teacher dose did not freeze: {decision.status.value}")
    return TeacherDoseRecipe(decision, selected.learning_rate, evidence)


def freeze_acquisition(
    stage_a: StageALiveEvidence,
    smoke: LiveSmokeCalibrationEvidence,
    max_tokens: MaxTokenFreezeEvidence,
    common_rl: CommonRLFreezeEvidence,
) -> AcquisitionFreeze:
    decisions: tuple[StageALearningRateDecision, ...] = (
        select_direct_m0_learning_rate("B-S", stage_a.bs_screens),
        select_direct_m0_learning_rate("B-G", stage_a.bg_screens),
    )
    if any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        raise RunnerGateError("Stage-A learning rates did not freeze")
    duration = decide_stage_a_duration(decisions, stage_a.final_evidence)
    if duration.status is not StageADurationDecisionStatus.FROZEN:
        raise RunnerGateError(
            f"Stage-A duration did not freeze: {duration.status.value}"
        )
    bs_lr, bg_lr = (row.selected_learning_rate for row in decisions)
    assert bs_lr is not None and bg_lr is not None
    return AcquisitionFreeze(
        {"static_sft": bs_lr, "on_policy_sft": bs_lr, "group_relative_rl": bg_lr},
        duration,
        max_tokens.selected_max_tokens,
        ALL_METHODS,
        smoke,
        max_tokens,
        common_rl,
        COMMON_RL_METHODS,
    )


__all__: tuple[str, ...] = (
    "ALL_METHODS",
    "COMMON_RL_METHODS",
    "FROZEN_PROTOCOL_MAX_TOKENS",
    "AcquisitionFreeze",
    "CommonRLFreezeEvidence",
    "LiveSmokeCalibrationEvidence",
    "MAX_TOKEN_CANDIDATE_LADDER",
    "MAX_TOKEN_GENERATIONS_SHA256",
    "MAX_TOKEN_REFERENCE_RUN_ID",
    "MAX_TOKEN_REWARDS_SHA256",
    "MAX_TOKEN_SELECTION_RULE",
    "MAX_TOKEN_SUMMARY_SHA256",
    "MaxTokenCapObservation",
    "MaxTokenFreezeEvidence",
    "MaxTokenReferenceEvidence",
    "StageALiveEvidence",
    "TeacherDoseArmEvidence",
    "TeacherDoseLiveEvidence",
    "TeacherDoseRecipe",
    "freeze_acquisition",
    "select_teacher_recipe",
)
