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
    select_stage_a_learning_rate,
)
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


@dataclass(frozen=True, slots=True)
class MaxTokenPathEvidence:
    method: str
    max_tokens: int
    reference_sample_count: int
    reference_censored_count: int
    censored_success_count: int
    minimum_answer_terminated_count: int

    def __post_init__(self) -> None:
        if (
            not self.method.strip()
            or type(self.max_tokens) is not int
            or self.max_tokens < PROVISIONAL_ACQUISITION_MAX_TOKENS
            or type(self.reference_sample_count) is not int
            or self.reference_sample_count < 1
            or any(
                type(value) is not int or not 0 <= value <= self.reference_sample_count
                for value in (
                    self.reference_censored_count,
                    self.censored_success_count,
                    self.minimum_answer_terminated_count,
                )
            )
            or self.censored_success_count > self.reference_censored_count
        ):
            raise ValueError("invalid acquisition max-token path evidence")


@dataclass(frozen=True, slots=True)
class MaxTokenFreezeEvidence:
    specification_sha256: str
    evidence_sha256: str
    candidate_max_tokens: tuple[int, ...]
    maximum_reference_censor_rate: float
    rows: tuple[MaxTokenPathEvidence, ...]
    selected_max_tokens: int
    specification_authorization_sha256: str

    def __post_init__(self) -> None:
        validate_sha256_id(self.specification_sha256)
        validate_sha256_id(self.evidence_sha256)
        validate_sha256_id(self.specification_authorization_sha256)
        candidates = self.candidate_max_tokens
        if (
            not candidates
            or candidates[0] != PROVISIONAL_ACQUISITION_MAX_TOKENS
            or candidates != tuple(sorted(set(candidates)))
            or candidates[-1] > FROZEN_PROTOCOL_MAX_TOKENS
            or self.selected_max_tokens not in candidates
            or not 0 <= self.maximum_reference_censor_rate < 1
        ):
            raise ValueError("invalid prospective acquisition max-token rule")
        by_cap = {
            cap: tuple(row for row in self.rows if row.max_tokens == cap)
            for cap in candidates
        }
        tested = candidates[: candidates.index(self.selected_max_tokens) + 1]
        methods = tuple(row.method for row in by_cap[tested[0]])
        if (
            any(tuple(row.method for row in by_cap[cap]) != methods for cap in tested)
            or len(methods) < 2
            or len(set(methods)) != len(methods)
            or any(by_cap[cap] for cap in candidates[len(tested) :])
        ):
            raise RunnerGateError("max-token evidence is not a common ordered gate")
        passes = {
            cap: all(
                row.reference_censored_count / row.reference_sample_count
                <= self.maximum_reference_censor_rate
                and row.censored_success_count == 0
                and row.minimum_answer_terminated_count > 0
                for row in by_cap[cap]
            )
            for cap in tested
        }
        if not passes[self.selected_max_tokens] or any(
            passes[cap] for cap in tested[:-1]
        ):
            raise RunnerGateError("max-token selection does not follow its frozen rule")


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
        select_stage_a_learning_rate("B-S", stage_a.bs_screens),
        select_stage_a_learning_rate("B-G", stage_a.bg_screens),
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
    "MaxTokenFreezeEvidence",
    "MaxTokenPathEvidence",
    "StageALiveEvidence",
    "TeacherDoseArmEvidence",
    "TeacherDoseLiveEvidence",
    "TeacherDoseRecipe",
    "freeze_acquisition",
    "select_teacher_recipe",
)
