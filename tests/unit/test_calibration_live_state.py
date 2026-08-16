from __future__ import annotations

import json
from pathlib import Path

import pytest

from duraseed.calibration_input_loader import (
    ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256,
)
from duraseed.calibration_sources import (
    ACCEPTED_BOUNDARY_FREEZE_EQUIVALENCE_SHA256,
    load_max_token_evidence,
)
from duraseed.calibration_state import (
    artifact,
    checkpoint,
    commit_action,
    existing,
    write,
)
from duraseed.max_token_ratification import REFERENCE_DESCRIPTOR
from duraseed.provenance import sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.training.acquisition_freeze import (
    ALL_METHODS,
    FROZEN_PROTOCOL_MAX_TOKENS,
    LiveSmokeCalibrationEvidence,
    MAX_TOKEN_CANDIDATE_LADDER,
    MAX_TOKEN_GENERATIONS_SHA256,
    MAX_TOKEN_REFERENCE_RUN_ID,
    MAX_TOKEN_REWARDS_SHA256,
    MAX_TOKEN_SUMMARY_SHA256,
    MaxTokenCapObservation,
    MaxTokenFreezeEvidence,
    MaxTokenReferenceEvidence,
)


SHA_A = "sha256:" + "a" * 64


def test_completed_action_prefix_resumes_by_hash_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    preflight = SHA_A
    checkpoint(tmp_path, preflight)
    for action, cost in (
        ("teacher-dose", 75.5),
        ("teacher-allocation", 0),
        ("stage-a", 224.5),
    ):
        commit_action(
            tmp_path,
            action,
            artifact(action, cost, preflight_sha256=preflight, result=action),
            preflight,
        )

    state, values = existing(tmp_path, preflight)
    assert state["status"] == "completed"
    assert tuple(values) == ("teacher-dose", "teacher-allocation", "stage-a")
    changed = artifact(
        "teacher-dose", 149, preflight_sha256=preflight, result="teacher-dose"
    )
    write(tmp_path / "teacher-dose.json", changed)
    with pytest.raises(RunnerGateError, match="hash mismatch"):
        existing(tmp_path, preflight)


def test_action_artifacts_require_committed_state_and_dependency_prefix(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "teacher-dose.json",
        artifact("teacher-dose", 75.5, preflight_sha256=SHA_A),
    )
    with pytest.raises(RunnerGateError, match="without its committed state"):
        existing(tmp_path, SHA_A)
    write(tmp_path / "state.json", {"preflight_sha256": SHA_A, "artifact_sha256": {}})
    write(
        tmp_path / "acquisition-freeze.json",
        artifact("stage-a", 224.5, preflight_sha256=SHA_A),
    )
    with pytest.raises(RunnerGateError, match="dependency prefix"):
        existing(tmp_path, SHA_A)


def _max_token_reference(**changes: object) -> MaxTokenReferenceEvidence:
    values: dict[str, object] = {
        "summary_sha256": MAX_TOKEN_SUMMARY_SHA256,
        "generations_sha256": MAX_TOKEN_GENERATIONS_SHA256,
        "rewards_sha256": MAX_TOKEN_REWARDS_SHA256,
        "run_id": MAX_TOKEN_REFERENCE_RUN_ID,
        "sample_count": 128,
        "source_split": "a_validation",
        "seed": 5,
        "training_step": 2,
        "reference_max_tokens": 4096,
        "answer_termination_count": 73,
        "exact_success_count": 8,
        "observations": tuple(
            MaxTokenCapObservation(*row)
            for row in (
                (512, 15, 3, False),
                (1024, 9, 2, False),
                (2048, 4, 1, False),
                (4096, 0, 0, True),
            )
        ),
    }
    values.update(changes)
    return MaxTokenReferenceEvidence(**values)  # type: ignore[arg-type]


def _ratification_spec() -> dict[str, object]:
    return {
        "schema_version": "duraseed-acquisition-max-token-ratification-v1",
        "status": "ratified_from_preexisting_m0_evidence",
        "specified_at_utc": "2026-08-13T00:00:00Z",
        "reference": REFERENCE_DESCRIPTOR,
        "candidate_max_tokens": list(MAX_TOKEN_CANDIDATE_LADDER),
        "maximum_reference_censor_rate": 0.05,
        "minimum_answer_terminations": 64,
        "require_zero_censored_exact_successes": True,
        "provisional_failure_inferred_from_max_tokens": 512,
        "selected_max_tokens": 4096,
        "common_apply_to_methods": list(ALL_METHODS),
    }


def _write_ratification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, authorized_at: str
) -> tuple[Path, Path]:
    import duraseed.calibration_sources as sources

    spec = tmp_path / "spec.json"
    auth = tmp_path / "auth.json"
    spec.write_text(json.dumps(_ratification_spec()))
    spec_hash = sha256_bytes(spec.read_bytes())
    auth.write_text(
        json.dumps(
            {
                "schema_version": "duraseed-acquisition-max-token-authorization-v1",
                "status": "accepted",
                "specification_sha256": spec_hash,
                "authorizer": "Ely",
                "authorized_at_utc": authorized_at,
            }
        )
    )
    monkeypatch.setattr(sources, "ACCEPTED_MAX_TOKEN_SPECIFICATION_SHA256", spec_hash)
    monkeypatch.setattr(
        sources,
        "ACCEPTED_MAX_TOKEN_AUTHORIZATION_SHA256",
        sha256_bytes(auth.read_bytes()),
    )
    return spec, auth


def test_max_token_gate_rejects_missing_authorization_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        MaxTokenFreezeEvidence(
            SHA_A,
            MAX_TOKEN_SUMMARY_SHA256,
            "",
            _max_token_reference(),
            MAX_TOKEN_CANDIDATE_LADDER,
            0.05,
            64,
            True,
            512,
            4096,
            ALL_METHODS,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    (("sample_count", 127), ("summary_sha256", SHA_A)),
)
def test_max_token_reference_requires_exact_frozen_counts_and_hashes(
    field: str, invalid: object
) -> None:
    with pytest.raises(RunnerGateError, match="frozen M0 source"):
        _max_token_reference(**{field: invalid})


def test_accepted_max_token_artifacts_authenticate_frozen_source() -> None:
    root = Path(__file__).resolve().parents[2]
    summary = (
        root
        / "frozen/v0/runs/tinker-calibration/tces-cap"
        / MAX_TOKEN_REFERENCE_RUN_ID
        / "tces_cap_summary.json"
    )
    value = load_max_token_evidence(
        root / "provenance/acquisition-max-token-ratification.json",
        root / "provenance/acquisition-max-token-authorization.json",
        summary,
    )
    assert value.selected_max_tokens == 4096
    assert value.reference.sample_count == 128
    assert value.apply_to_methods == ALL_METHODS


def test_completed_panel_sources_are_pinned() -> None:
    assert ACCEPTED_BOUNDARY_FREEZE_EQUIVALENCE_SHA256 == (
        "sha256:e003fe85289f29915e25582b22ae582182b89185ea66f0d74fdcb4a202653f15"
    )
    assert ACCEPTED_PANEL_SPLIT_AUTHORIZATION_SHA256 == (
        "sha256:420421f8bf0d8fbac08791d72b908e627f6a4ed845834d91b818eac0ab064e12"
    )


def test_max_token_loader_authenticates_one_source_then_common_apply_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, auth = _write_ratification(
        tmp_path, monkeypatch, authorized_at="2026-08-13T00:00:01Z"
    )
    root = Path(__file__).resolve().parents[2]
    summary = (
        root
        / "frozen/v0/runs/tinker-calibration/tces-cap"
        / MAX_TOKEN_REFERENCE_RUN_ID
        / "tces_cap_summary.json"
    )
    value = load_max_token_evidence(spec, auth, summary)
    assert value.reference.sample_count == 128
    assert value.reference.reference_max_tokens == 4096
    assert value.apply_to_methods == ALL_METHODS
    assert not hasattr(value.reference, "method")


def test_max_token_loader_rejects_bad_ladder_and_chronology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, auth = _write_ratification(
        tmp_path, monkeypatch, authorized_at="2026-08-12T23:59:59Z"
    )
    root = Path(__file__).resolve().parents[2]
    summary = (
        root
        / "frozen/v0/runs/tinker-calibration/tces-cap"
        / MAX_TOKEN_REFERENCE_RUN_ID
        / "tces_cap_summary.json"
    )
    with pytest.raises(RunnerGateError, match="chronology"):
        load_max_token_evidence(spec, auth, summary)

    altered = _ratification_spec()
    altered["candidate_max_tokens"] = [256, 512, 4096]
    spec.write_text(json.dumps(altered))
    import duraseed.calibration_sources as sources

    spec_hash = sha256_bytes(spec.read_bytes())
    auth_payload = json.loads(auth.read_text())
    auth_payload["specification_sha256"] = spec_hash
    auth_payload["authorized_at_utc"] = "2026-08-13T00:00:01Z"
    auth.write_text(json.dumps(auth_payload))
    monkeypatch.setattr(sources, "ACCEPTED_MAX_TOKEN_SPECIFICATION_SHA256", spec_hash)
    monkeypatch.setattr(
        sources,
        "ACCEPTED_MAX_TOKEN_AUTHORIZATION_SHA256",
        sha256_bytes(auth.read_bytes()),
    )
    with pytest.raises(RunnerGateError, match="proposed rule"):
        load_max_token_evidence(spec, auth, summary)


def test_live_smoke_authenticates_protocol_transport_not_cap_selection() -> None:
    value = LiveSmokeCalibrationEvidence(SHA_A, 4096, 16, 4, True)
    assert value.protocol_max_tokens == FROZEN_PROTOCOL_MAX_TOKENS
    with pytest.raises(ValueError, match="max-token evidence"):
        LiveSmokeCalibrationEvidence(SHA_A, 2048, 16, 0, True)
