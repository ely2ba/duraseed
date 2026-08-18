"""Final acquisition freeze for the verifier-valid Stage-A amendment."""

from __future__ import annotations

from duraseed.runners import RunnerGateError
from duraseed.training.acquisition_freeze import (
    ALL_METHODS,
    COMMON_RL_METHODS,
    AcquisitionFreeze,
    CommonRLFreezeEvidence,
    LiveSmokeCalibrationEvidence,
    MaxTokenFreezeEvidence,
)
from duraseed.training.stage_a_amended import (
    AmendedStageALiveEvidence,
    decide_amended_stage_a_duration,
    decide_amended_stage_a_screen,
)
from duraseed.training.stage_a_calibration import (
    StageADurationDecisionStatus,
    StageALearningRateDecisionStatus,
)


def freeze_amended_acquisition(
    stage_a: AmendedStageALiveEvidence,
    smoke: LiveSmokeCalibrationEvidence,
    max_tokens: MaxTokenFreezeEvidence,
    common_rl: CommonRLFreezeEvidence,
) -> AcquisitionFreeze:
    """Freeze the prospective fixed coordinates after corrected evidence gates."""

    decisions = tuple(
        decide_amended_stage_a_screen(method, screens)
        for method, screens in (
            ("B-S", stage_a.bs_screens),
            ("B-G", stage_a.bg_screens),
        )
    )
    if any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        raise RunnerGateError("amended Stage-A health gates did not pass")
    duration = decide_amended_stage_a_duration(decisions, stage_a.final_evidence)
    if duration.status is not StageADurationDecisionStatus.FROZEN:
        raise RunnerGateError(
            f"amended Stage-A duration did not freeze: {duration.status.value}"
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


__all__ = ["freeze_amended_acquisition"]
