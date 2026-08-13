"""Indivisible Stage-A calibration under its launch allocation."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from duraseed.calibration_attempts import ArmAttempts
from duraseed.calibration_budget import (
    persist_budget_preflight,
    require_remaining_budget,
    stage_a_budget,
)
from duraseed.provenance import canonical_json_hash
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration_live import CalibrationLiveInputs
from duraseed.runners.stage_a_arms import run_final, run_screen
from duraseed.runners.stage_a_evidence import ordered_pools, solver_sources
from duraseed.runners.stage_a_result import common_rl_freeze
from duraseed.runners.stage_a_setup import build_origin
from duraseed.training.acquisition_freeze import (
    CommonRLFreezeEvidence,
    StageALiveEvidence,
)
from duraseed.training.stage_a_calibration import (
    STAGE_A_LEARNING_RATE_GRIDS,
    StageALearningRateDecisionStatus,
    select_stage_a_learning_rate,
)


_COMPLETED = TypeAdapter(tuple[StageALiveEvidence, CommonRLFreezeEvidence])


async def collect_stage_a(
    inputs: CalibrationLiveInputs,
    output: Path,
    *,
    selected_dose: int,
    teacher_learning_rate: float,
    preflight_sha256: str,
) -> tuple[StageALiveEvidence, CommonRLFreezeEvidence]:
    """Never splice stochastic evidence across Stage-A training clients."""

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
        raise RunnerGateError("Stage-A prompt pools differ from teacher-dose sources")
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
        stage_a_budget(inputs, selected_dose, completed=completed),
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
        return _COMPLETED.validate_python(attempt.completed_payload)
    assert attempt.journal is not None
    origin = await build_origin(
        inputs,
        attempt.directory,
        attempt.journal,
        selected_dose=selected_dose,
        teacher_learning_rate=teacher_learning_rate,
        checkpoint_suffix=f"-{attempt.directory.name}",
    )
    pools = ordered_pools(inputs.prompt_pools)
    sources = {
        row.task_id: row
        for row in solver_sources(inputs.prompt_pools.a_rl_train_manifest)
    }
    branches = {}
    screens = {"B-S": [], "B-G": []}
    for method in ("B-S", "B-G"):
        for learning_rate in STAGE_A_LEARNING_RATE_GRIDS[method]:
            evidence, branch = await run_screen(
                inputs,
                origin,
                attempt.directory,
                attempt.journal,
                method=method,
                learning_rate=learning_rate,
                pools=pools,
                sources=sources,
            )
            screens[method].append(evidence)
            branches[(method, learning_rate)] = branch
    decisions = tuple(
        select_stage_a_learning_rate(method, screens[method])
        for method in ("B-S", "B-G")
    )
    if any(
        row.status is not StageALearningRateDecisionStatus.SELECTED for row in decisions
    ):
        raise RunnerGateError("Stage-A learning-rate screen did not select both arms")
    finals = []
    for decision in decisions:
        learning_rate = decision.selected_learning_rate
        assert learning_rate is not None
        finals.append(
            await run_final(
                inputs,
                origin,
                attempt.directory,
                attempt.journal,
                method=decision.method,
                learning_rate=learning_rate,
                branch=branches[(decision.method, learning_rate)],
                pools=pools,
                sources=sources,
            )
        )
    bg = next(row for row in finals if row.evidence.method == "B-G")
    common_rl = common_rl_freeze(bg)
    evidence = StageALiveEvidence(
        tuple(screens["B-S"]),
        tuple(screens["B-G"]),
        tuple(row.evidence for row in finals),
    )
    attempts.complete(attempt, (evidence, common_rl))
    attempts.assert_no_unused_reconciliations()
    return evidence, common_rl


__all__ = ["collect_stage_a"]
