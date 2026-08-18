"""Indivisible prospective two-arm Stage-A calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Never

from pydantic import TypeAdapter

from duraseed.calibration_attempts import ArmAttempt, ArmAttempts
from duraseed.calibration_budget import (
    persist_budget_preflight,
    require_remaining_budget,
    stage_a_budget,
)
from duraseed.calibration_stage_a_terminal import StageAScientificFailure
from duraseed.provenance import canonical_json_hash
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.stage_a_amended_arms import (
    run_amended_final,
    run_amended_screen,
)
from duraseed.runners.stage_a_evidence import ordered_pools, solver_sources
from duraseed.runners.stage_a_result import common_rl_freeze
from duraseed.runners.stage_a_setup import build_origin
from duraseed.training.acquisition_freeze import CommonRLFreezeEvidence
from duraseed.training.stage_a_amended import (
    AMENDED_STAGE_A_LEARNING_RATES,
    AmendedStageALiveEvidence,
    decide_amended_stage_a_duration,
    decide_amended_stage_a_screen,
)
from duraseed.training.stage_a_calibration import (
    StageADurationDecisionStatus,
    StageALearningRateDecisionStatus,
)
from duraseed.training.stage_a_update_health import (
    StageAUpdateHealthFailure,
    StageAUpdateHealthFailureEvidence,
    parse_stage_a_update_health_failure,
)


_COMPLETED = TypeAdapter(
    tuple[AmendedStageALiveEvidence, CommonRLFreezeEvidence | None]
)


def _completed_update_health_failure(
    value: object,
) -> StageAUpdateHealthFailureEvidence | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or value[1] is not None:
        return None
    evidence = value[0]
    if not isinstance(evidence, dict) or evidence.get("schema_version") != (
        "duraseed-stage-a-update-health-failure-v1"
    ):
        return None
    return parse_stage_a_update_health_failure(evidence)


def _terminalize_update_health_failure(
    attempts: ArmAttempts, attempt: ArmAttempt, failure: StageAUpdateHealthFailure
) -> Never:
    attempts.complete(attempt, (failure.evidence, None))
    attempts.assert_no_unused_reconciliations()
    raise StageAScientificFailure(
        "update_health_failed", failure.evidence, (), None
    ) from failure


def _resolve_completed(
    evidence: AmendedStageALiveEvidence,
    common_rl: CommonRLFreezeEvidence | None,
) -> tuple[AmendedStageALiveEvidence, CommonRLFreezeEvidence]:
    decisions = tuple(
        decide_amended_stage_a_screen(method, screens)
        for method, screens in (
            ("B-S", evidence.bs_screens),
            ("B-G", evidence.bg_screens),
        )
    )
    if any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        raise StageAScientificFailure(
            "no_eligible_learning_rate", evidence, decisions, None
        )
    duration = decide_amended_stage_a_duration(decisions, evidence.final_evidence)
    if duration.status is not StageADurationDecisionStatus.FROZEN:
        raise StageAScientificFailure(
            "duration_gate_failed", evidence, decisions, duration
        )
    if common_rl is None:
        raise StageAScientificFailure(
            "common_rl_gate_failed", evidence, decisions, duration
        )
    return evidence, common_rl


async def collect_amended_stage_a(
    inputs: CalibrationLiveInputs,
    output: Path,
    *,
    preflight_sha256: str,
) -> tuple[AmendedStageALiveEvidence, CommonRLFreezeEvidence]:
    """Run only the selected arms continuously, with one hard step-10 gate."""

    if inputs.stage_a_ledger.authorized_usd <= 0:
        raise RunnerGateError("Stage-A collector requires its preflight ledger")
    if (
        inputs.prompt_pools.a_rl_train_manifest
        != inputs.teacher_sources.a_rl_train_manifest
        or inputs.prompt_pools.a_monitor_manifest
        != inputs.teacher_sources.a_monitor_manifest
        or inputs.prompt_pools.artifact.family_panel_artifact_id
        != canonical_json_hash(inputs.teacher_sources.panel)
    ):
        raise RunnerGateError("Stage-A prompt pools differ from authenticated sources")
    attempts = ArmAttempts(
        output,
        inputs.stage_a_ledger,
        run_id=inputs.run_id,
        action="stage-a",
        project_id=inputs.project_id,
        preflight_sha256=preflight_sha256,
        reconciliations=tuple(
            row for row in inputs.reconciled_restarts if row.action == "stage-a"
        ),
    )
    completed = "complete-bounded-stage-a" in attempts.completed_arm_ids
    budget_preflight = require_remaining_budget(
        stage_a_budget(inputs, completed=completed),
        inputs.stage_a_ledger,
        prior_billed_usd=attempts.prior_billed_usd,
    )
    persist_budget_preflight(
        output,
        {
            **budget_preflight,
            "run_id": inputs.run_id,
            "action": "stage-a",
            "preflight_sha256": preflight_sha256,
        },
    )
    attempt = attempts.open("complete-bounded-stage-a")
    if attempt.completed:
        failure = _completed_update_health_failure(attempt.completed_payload)
        if failure is not None:
            raise StageAScientificFailure("update_health_failed", failure, (), None)
        return _resolve_completed(
            *_COMPLETED.validate_python(attempt.completed_payload)
        )
    assert attempt.journal is not None
    origin = await build_origin(inputs, attempt.directory, attempt.journal)
    pools = ordered_pools(inputs.prompt_pools)
    sources = {
        row.task_id: row
        for row in solver_sources(inputs.prompt_pools.a_rl_train_manifest)
    }
    screens = {"B-S": [], "B-G": []}
    branches = {}
    for method in ("B-S", "B-G"):
        learning_rate = AMENDED_STAGE_A_LEARNING_RATES[method]
        try:
            evidence, branch = await run_amended_screen(
                inputs,
                origin,
                attempt.directory,
                attempt.journal,
                method=method,
                learning_rate=learning_rate,
                pools=pools,
                sources=sources,
            )
        except StageAUpdateHealthFailure as failure:
            _terminalize_update_health_failure(attempts, attempt, failure)
        screens[method].append(evidence)
        branches[method] = branch
    decisions = tuple(
        decide_amended_stage_a_screen(method, screens[method])
        for method in ("B-S", "B-G")
    )
    evidence = AmendedStageALiveEvidence(
        tuple(screens["B-S"]), tuple(screens["B-G"]), ()
    )
    if any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        attempts.complete(attempt, (evidence, None))
        attempts.assert_no_unused_reconciliations()
        return _resolve_completed(evidence, None)

    finals = []
    for decision in decisions:
        learning_rate = decision.selected_learning_rate
        assert learning_rate is not None
        try:
            finals.append(
                await run_amended_final(
                    inputs,
                    origin,
                    attempt.directory,
                    attempt.journal,
                    method=decision.method,
                    learning_rate=learning_rate,
                    branch=branches[decision.method],
                    pools=pools,
                    sources=sources,
                )
            )
        except StageAUpdateHealthFailure as failure:
            _terminalize_update_health_failure(attempts, attempt, failure)
    evidence = AmendedStageALiveEvidence(
        tuple(screens["B-S"]),
        tuple(screens["B-G"]),
        tuple(row.evidence for row in finals),  # type: ignore[arg-type]
    )
    duration = decide_amended_stage_a_duration(decisions, evidence.final_evidence)
    if duration.status is not StageADurationDecisionStatus.FROZEN:
        attempts.complete(attempt, (evidence, None))
        attempts.assert_no_unused_reconciliations()
        return _resolve_completed(evidence, None)
    bg = next(row for row in finals if row.evidence.method == "B-G")
    try:
        common_rl = common_rl_freeze(bg)
    except RunnerGateError as error:
        attempts.complete(attempt, (evidence, None))
        attempts.assert_no_unused_reconciliations()
        raise StageAScientificFailure(
            "common_rl_gate_failed", evidence, decisions, duration
        ) from error
    attempts.complete(attempt, (evidence, common_rl))
    attempts.assert_no_unused_reconciliations()
    return _resolve_completed(evidence, common_rl)


__all__ = ["collect_amended_stage_a"]
