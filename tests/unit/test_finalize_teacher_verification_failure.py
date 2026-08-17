from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

from tools import finalize_teacher_verification_failure as finalizer

from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import (
    RunRecord,
    RunStatus,
    read_run_record,
    write_run_record,
)
from duraseed.training.acquisition_freeze import TeacherDoseArmEvidence
from duraseed.training.teacher_dose import TeacherDoseDecisionStatus
from tests.unit.teacher_dose_fixtures import _assessment


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _evidence():  # type: ignore[no-untyped-def]
    rates = sorted(CONFIG.tinker.learning_rates.teacher_seed_sft.grid)
    calibration = tuple(
        TeacherDoseArmEvidence(
            rate,
            _assessment(
                dose,
                passing=(dose == 2 and rate == rates[0]),
            ),
        )
        for dose in (1, 2)
        for rate in rates
    )
    verification = TeacherDoseArmEvidence(
        rates[0], _assessment(2, seed=37, passing=False)
    )
    return calibration, verification


def test_terminal_decision_preserves_smallest_lr_and_verification_failure() -> None:
    decision, candidate = finalizer._decision(CONFIG, *_evidence())

    assert decision.status is TeacherDoseDecisionStatus.VERIFICATION_FAILED
    assert decision.candidate_dose == 2
    assert decision.selected_dose is None
    assert candidate.learning_rate == min(
        CONFIG.tinker.learning_rates.teacher_seed_sft.grid
    )


def test_local_finalizer_writes_one_idempotent_failed_terminal_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "acquisition-calibration-test"
    run_directory = tmp_path / run_id
    run_directory.mkdir()
    preflight = {"run_id": run_id}
    (run_directory / "preflight.json").write_bytes(canonical_json_bytes(preflight))
    preflight_sha256 = sha256_bytes((run_directory / "preflight.json").read_bytes())
    (run_directory / "state.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "duraseed-acquisition-calibration-v1",
                "status": "interrupted",
                "completed_actions": [],
                "artifact_sha256": {},
                "preflight_sha256": preflight_sha256,
                "error": "ValidationError: JSON arrays were not parsed as tuples",
            }
        )
    )
    manifests = {
        "a_monitor": SHA_A,
        "a_rl_train": SHA_A,
        "a_seed_gate": SHA_A,
        "a_seed_train": SHA_A,
    }
    now = datetime(2026, 8, 17, 4, 46, tzinfo=UTC)
    write_run_record(
        run_directory,
        RunRecord(
            protocol_version=str(CONFIG.protocol["version"]),
            git_commit="incident-source-commit",
            resolved_config_hash=CONFIG.resolved_config_hash(),
            run_kind="stage_a_calibration",
            method=None,
            seed=17,
            model_id=CONFIG.tinker.model_id,
            renderer=CONFIG.tinker.renderer_name,
            lora_rank=CONFIG.tinker.lora_rank,
            task_manifest_ids=manifests,
            status=RunStatus.INTERRUPTED,
            started_at=now,
            updated_at=now,
            finished_at=now,
            deviations=["billing reconciliation required after remote completion"],
        ),
    )
    loaded = SimpleNamespace(
        prompts=SimpleNamespace(
            a_monitor_manifest=SimpleNamespace(manifest_id=SHA_A),
            a_rl_train_manifest=SimpleNamespace(manifest_id=SHA_A),
        ),
        teacher=SimpleNamespace(
            gate_manifest=SimpleNamespace(manifest_id=SHA_A),
            target_train_manifest=SimpleNamespace(manifest_id=SHA_A),
        ),
    )
    monkeypatch.setattr(finalizer, "load_pilot_config", lambda _path: CONFIG)
    monkeypatch.setattr(
        finalizer, "load_calibration_source_objects", lambda **_kwargs: loaded
    )
    monkeypatch.setattr(finalizer, "_teacher_evidence", lambda *_args: _evidence())
    monkeypatch.setattr(
        finalizer,
        "seal_calibration_action",
        lambda *_args, **_kwargs: {
            "schema_version": "duraseed-calibration-integrity-v1",
            "artifact_sha256": SHA_B,
        },
    )
    arguments = {
        "run_directory": run_directory,
        "config_path": tmp_path / "config.yaml",
        "boundary_directory": tmp_path / "boundary",
        "source_directory": tmp_path / "sources",
        "panel_split_authorization_path": tmp_path / "authorization.json",
        "panel_split_equivalence_path": tmp_path / "equivalence.json",
    }

    first = finalizer.finalize_teacher_verification_failure(**arguments)
    terminal_path = run_directory / finalizer.TERMINAL_FILE
    first_terminal = terminal_path.read_bytes()
    second = finalizer.finalize_teacher_verification_failure(**arguments)

    assert first == second
    assert terminal_path.read_bytes() == first_terminal
    assert first["status"] == "verification_failed"
    terminal = json.loads(first_terminal)
    assert terminal["prior_interruption"].startswith("ValidationError")
    state = json.loads((run_directory / "state.json").read_bytes())
    assert state["status"] == "failed"
    assert state["terminal_decision"]["status"] == "verification_failed"
    assert read_run_record(run_directory).status is RunStatus.FAILED
    assert not (run_directory / "stage-a-arms").exists()
