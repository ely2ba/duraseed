from __future__ import annotations

from pathlib import Path

import pytest

from duraseed.calibration_sources import load_max_token_evidence
from duraseed.calibration_state import (
    artifact,
    checkpoint,
    commit_action,
    existing,
    write,
)
from duraseed.runners import RunnerGateError
from duraseed.training.acquisition_freeze import (
    FROZEN_PROTOCOL_MAX_TOKENS,
    LiveSmokeCalibrationEvidence,
    MaxTokenFreezeEvidence,
    MaxTokenPathEvidence,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


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


def _max_token_rows(cap: int) -> tuple[MaxTokenPathEvidence, ...]:
    return tuple(
        MaxTokenPathEvidence(method, cap, 8, 0, 0, 8)
        for method in ("supervised", "on_policy", "group_relative")
    )


def test_max_token_gate_rejects_missing_authorization_hash() -> None:
    with pytest.raises(ValueError, match="sha256"):
        MaxTokenFreezeEvidence(SHA_A, SHA_B, (256,), 0.0, _max_token_rows(256), 256, "")


def test_max_token_loader_requires_ratified_authorization_bytes(
    tmp_path: Path,
) -> None:
    import json

    spec = tmp_path / "spec.json"
    auth = tmp_path / "auth.json"
    evidence = tmp_path / "evidence.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": "duraseed-acquisition-max-token-spec-v1",
                "status": "ratified_before_sampling",
                "representative_paths": ["supervised", "on_policy", "group_relative"],
                "candidate_max_tokens": [256],
                "maximum_reference_censor_rate": 0.0,
            }
        )
    )
    auth.write_text(
        json.dumps(
            {
                "schema_version": "duraseed-acquisition-max-token-authorization-v1",
                "status": "accepted",
                "specification_sha256": SHA_A,
                "authorizer": "Ely",
                "authorized_at_utc": "2026-08-13T00:00:00Z",
            }
        )
    )
    evidence.write_text("{}")
    with pytest.raises(RunnerGateError, match="no prospective"):
        load_max_token_evidence(spec, auth, evidence)


def test_live_smoke_authenticates_protocol_transport_not_cap_selection() -> None:
    value = LiveSmokeCalibrationEvidence(SHA_A, 4096, 16, 4, True)
    assert value.protocol_max_tokens == FROZEN_PROTOCOL_MAX_TOKENS
    with pytest.raises(ValueError, match="max-token evidence"):
        LiveSmokeCalibrationEvidence(SHA_A, 2048, 16, 0, True)
