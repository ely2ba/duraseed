#!/usr/bin/env python3
"""Locally seal the completed 2026-08 teacher-dose verification failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from duraseed.calibration_input_loader import load_calibration_source_objects
from duraseed.calibration_integrity import seal_calibration_action
from duraseed.calibration_state import SCHEMA_VERSION
from duraseed.config import load_pilot_config
from duraseed.data.io import atomic_write_bytes
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import RunStatus, read_run_record, write_run_record
from duraseed.runners import RunnerGateError
from duraseed.runners.teacher_dose_evidence import (
    RAW_TEACHER_ARM,
    TEACHER_BASELINE,
    assess_arm,
)
from duraseed.training.acquisition_freeze import TeacherDoseArmEvidence
from duraseed.training.teacher_dose import (
    GateStatus,
    TeacherDoseDecisionStatus,
    decide_teacher_dose,
)


TERMINAL_FILE = "teacher-dose-terminal.json"
TERMINAL_SCHEMA = "duraseed-teacher-dose-terminal-v1"


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid local finalization input: {path}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"local finalization input is not an object: {path}")
    return value


def _write_exact(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError(f"terminal calibration artifact changed: {path.name}")
    atomic_write_bytes(path, payload)
    return sha256_bytes(payload)


def _arm_id(seed: int, dose: int, learning_rate: float) -> str:
    return f"seed-{seed}-dose-{dose}-lr-{learning_rate:.0e}".replace(
        "+", "plus"
    ).replace(".", "p")


def _completed_evidence(path: Path, preflight_sha256: str, *, baseline: bool) -> Any:
    value = _object(path)
    arm_id = path.parent.name
    if (
        value.get("arm_id") != arm_id
        or value.get("preflight_sha256") != preflight_sha256
        or type(value.get("attempt")) is not int
        or value["attempt"] < 1
        or "evidence" not in value
    ):
        raise RunnerGateError(f"completed teacher arm identity differs: {arm_id}")
    adapter = TEACHER_BASELINE if baseline else RAW_TEACHER_ARM
    return adapter.validate_json(canonical_json_bytes(value["evidence"]))


def _teacher_evidence(
    run_directory: Path, config: Any, teacher_sources: Any, preflight_sha256: str
) -> tuple[tuple[TeacherDoseArmEvidence, ...], TeacherDoseArmEvidence]:
    arms_root = run_directory / "teacher-dose-arms"
    baseline_paths = sorted(arms_root.glob("baseline-seed-*/completed.json"))
    if tuple(path.parent.name for path in baseline_paths) != (
        "baseline-seed-17",
        "baseline-seed-37",
    ):
        raise RunnerGateError("terminal teacher-dose baselines differ")
    for path in baseline_paths:
        _completed_evidence(path, preflight_sha256, baseline=True)

    inputs = SimpleNamespace(
        config=config,
        teacher_sources=teacher_sources,
        m0_sampler_path=teacher_sources.panel.m0_checkpoint_path,
    )
    calibration: list[TeacherDoseArmEvidence] = []
    verification: list[TeacherDoseArmEvidence] = []
    for path in sorted(arms_root.glob("seed-*/completed.json")):
        raw = _completed_evidence(path, preflight_sha256, baseline=False)
        if path.parent.name != _arm_id(raw.seed, raw.dose, raw.learning_rate):
            raise RunnerGateError("terminal teacher-dose arm coordinate differs")
        assessment = assess_arm(inputs, raw)
        if assessment.summary.run_id != run_directory.name:
            raise RunnerGateError("terminal teacher-dose run identity differs")
        evidence = TeacherDoseArmEvidence(raw.learning_rate, assessment)
        (calibration if raw.seed == 17 else verification).append(evidence)
    if len(verification) != 1 or verification[0].assessment.summary.training_seed != 37:
        raise RunnerGateError("terminal teacher-dose verification arm differs")
    return tuple(calibration), verification[0]


def _decision(
    config: Any,
    calibration: tuple[TeacherDoseArmEvidence, ...],
    verification: TeacherDoseArmEvidence,
) -> tuple[Any, TeacherDoseArmEvidence]:
    grid = config.teacher_dose.demonstrations_per_family
    rates = tuple(sorted(config.tinker.learning_rates.teacher_seed_sft.grid))
    by_dose: dict[int, list[TeacherDoseArmEvidence]] = {}
    for arm in calibration:
        dose = arm.assessment.summary.demonstrations_per_family
        by_dose.setdefault(dose, []).append(arm)
    if tuple(sorted(by_dose)) != grid[: len(by_dose)]:
        raise RunnerGateError("terminal teacher-dose grid is not a prefix")

    representatives = []
    candidate = None
    for dose in sorted(by_dose):
        arms = by_dose[dose]
        if tuple(sorted(arm.learning_rate for arm in arms)) != rates:
            raise RunnerGateError("terminal teacher-dose learning-rate grid differs")
        passing = tuple(
            arm for arm in arms if arm.assessment.status is GateStatus.PASSED
        )
        chosen = min(passing or tuple(arms), key=lambda arm: arm.learning_rate)
        representatives.append(chosen.assessment)
        if passing:
            candidate = chosen
            break
    if candidate is None or len(representatives) != len(by_dose):
        raise RunnerGateError("terminal teacher-dose did not stop at its first pass")
    if verification.learning_rate != candidate.learning_rate:
        raise RunnerGateError("terminal verification learning rate differs")
    decision = decide_teacher_dose(
        grid,
        representatives,
        verification_assessment=verification.assessment,
    )
    if decision.status is not TeacherDoseDecisionStatus.VERIFICATION_FAILED:
        raise RunnerGateError(
            f"terminal decision is not verification_failed: {decision.status.value}"
        )
    return decision, candidate


def finalize_teacher_verification_failure(
    *,
    run_directory: Path,
    config_path: Path,
    boundary_directory: Path,
    source_directory: Path,
    panel_split_authorization_path: Path,
    panel_split_equivalence_path: Path,
) -> dict[str, Any]:
    """Finalize this completed failure using local evidence only."""

    run_directory = run_directory.resolve()
    state = _object(run_directory / "state.json")
    preflight_path = run_directory / "preflight.json"
    preflight_sha256 = sha256_bytes(preflight_path.read_bytes())
    preflight = _object(preflight_path)
    run = read_run_record(run_directory)
    if (
        preflight.get("run_id") != run_directory.name
        or state.get("schema_version") != SCHEMA_VERSION
        or state.get("preflight_sha256") != preflight_sha256
        or state.get("completed_actions") != []
        or state.get("artifact_sha256") != {}
        or state.get("status") not in {"interrupted", "failed"}
        or run.run_kind != "stage_a_calibration"
        or run.status not in {RunStatus.INTERRUPTED, RunStatus.FAILED}
    ):
        raise RunnerGateError("terminal calibration run prefix differs")
    forbidden = (
        run_directory / "teacher-dose.json",
        run_directory / "teacher-allocation.json",
        run_directory / "acquisition-freeze.json",
        run_directory / "stage-a-arms",
        run_directory / "teacher-dose-arms/checkpoint-ttl-audit.json",
        run_directory / "teacher-dose-arms/ttl-verification-pending.json",
    )
    if any(path.exists() for path in forbidden):
        raise RunnerGateError(
            "terminal calibration advanced beyond teacher verification"
        )

    config = load_pilot_config(config_path)
    loaded = load_calibration_source_objects(
        config=config,
        boundary_directory=boundary_directory,
        source_directory=source_directory,
        panel_split_authorization_path=panel_split_authorization_path,
        panel_split_equivalence_path=panel_split_equivalence_path,
    )
    expected_manifests = {
        "a_monitor": loaded.prompts.a_monitor_manifest.manifest_id,
        "a_rl_train": loaded.prompts.a_rl_train_manifest.manifest_id,
        "a_seed_gate": loaded.teacher.gate_manifest.manifest_id,
        "a_seed_train": loaded.teacher.target_train_manifest.manifest_id,
    }
    if (
        run.resolved_config_hash != config.resolved_config_hash()
        or run.task_manifest_ids != expected_manifests
    ):
        raise RunnerGateError("terminal calibration source identity differs")

    calibration, verification = _teacher_evidence(
        run_directory, config, loaded.teacher, preflight_sha256
    )
    decision, candidate = _decision(config, calibration, verification)
    integrity = seal_calibration_action(
        "teacher-dose",
        run_directory / "teacher-dose-arms",
        teacher_updates=config.teacher_dose.calibration_updates,
    )
    terminal_path = run_directory / TERMINAL_FILE
    prior_interruption = state.get("error")
    if terminal_path.exists():
        prior_interruption = _object(terminal_path).get("prior_interruption")
    terminal = {
        "schema_version": TERMINAL_SCHEMA,
        "action": "teacher-dose",
        "status": decision.status,
        "run_id": run_directory.name,
        "run_git_commit": run.git_commit,
        "preflight_sha256": preflight_sha256,
        "finalizer_source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "candidate_learning_rate": candidate.learning_rate,
        "decision": decision,
        "calibration_arms": calibration,
        "verification_arm": verification,
        "integrity": integrity,
        "prior_interruption": prior_interruption,
    }
    terminal_hash = _write_exact(terminal_path, terminal)
    terminal_error = (
        "teacher-dose verification_failed: candidate dose failed verification; "
        "no automatic escalation"
    )
    atomic_write_bytes(
        run_directory / "state.json",
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "completed_actions": [],
                "artifact_sha256": {},
                "preflight_sha256": preflight_sha256,
                "terminal_decision": {
                    "action": "teacher-dose",
                    "status": decision.status,
                    "artifact_sha256": terminal_hash,
                },
                "error": terminal_error,
            }
        ),
    )
    deviation = "teacher-dose verification_failed; no automatic escalation"
    deviations = list(run.deviations)
    if deviation not in deviations:
        deviations.append(deviation)
    write_run_record(
        run_directory,
        run.model_copy(update={"status": RunStatus.FAILED, "deviations": deviations}),
    )
    return {
        "status": decision.status.value,
        "candidate_dose": decision.candidate_dose,
        "candidate_learning_rate": candidate.learning_rate,
        "integrity_sha256": integrity["artifact_sha256"],
        "terminal_artifact_sha256": terminal_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--boundary-directory", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--panel-split-authorization", required=True, type=Path)
    parser.add_argument("--panel-split-equivalence", required=True, type=Path)
    args = parser.parse_args()
    result = finalize_teacher_verification_failure(
        run_directory=args.run_directory,
        config_path=args.config,
        boundary_directory=args.boundary_directory,
        source_directory=args.source_directory,
        panel_split_authorization_path=args.panel_split_authorization,
        panel_split_equivalence_path=args.panel_split_equivalence,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
