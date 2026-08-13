from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed import pilot0_sources
from duraseed.pilot0_contract import EPHEMERAL_SAMPLER_FIXED_USD
from duraseed.pilot0_sources import load_pilot0_source_authentication
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import ephemeral_sampler
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import TokenBudget, TokenLedger


class _Model:
    async def save_weights_and_get_sampling_client_async(self):  # type: ignore[no-untyped-def]
        return object()


def test_ephemeral_sampler_uses_full_coordinate_identity_and_charges_fixed_cost(
    tmp_path: Path,
) -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 0), 1.0)
    inputs = SimpleNamespace(run_id="pilot", ledger=ledger)
    runtime = SimpleNamespace(model=_Model())
    journal = RemoteJournal(tmp_path)

    async def run() -> tuple[str, str]:
        _, first = await ephemeral_sampler(
            inputs,
            runtime,
            journal,
            coordinate={"seed": 11, "method": "B-G", "step": 1, "group": 0},
        )
        _, second = await ephemeral_sampler(
            inputs,
            runtime,
            journal,
            coordinate={"seed": 11, "method": "B-G", "step": 1, "group": 1},
        )
        return first, second

    first, second = asyncio.run(run())
    assert first != second
    assert first.startswith("ephemeral:pilot:")
    assert ledger.observed_fixed_usd == 2 * EPHEMERAL_SAMPLER_FIXED_USD


def _write(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def test_source_authentication_binds_smoke_m0_billing_and_sealed_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    smoke_hash = "sha256:" + "1" * 64
    selection_hash = "sha256:" + "2" * 64
    ttl_hash = "sha256:" + "3" * 64
    teacher_hash = "sha256:" + "4" * 64
    acquisition_hash = "sha256:" + "5" * 64
    smoke_finished = datetime(2026, 8, 13, 10, tzinfo=UTC)
    monkeypatch.setattr(
        pilot0_sources,
        "authenticate_live_smoke",
        lambda path, *, project_id: ("smoke-run", smoke_hash, smoke_finished),
    )
    monkeypatch.setattr(
        pilot0_sources,
        "load_m0_evidence",
        lambda selection, ttl: (
            "tinker://m0/sampler_weights/a",
            "tinker://m0/weights/a",
            2,
            selection_hash,
            ttl_hash,
        ),
    )
    raw_billing_path = tmp_path / "raw-billing.json"
    raw_billing_path.write_text("{}")
    raw_billing_hash = sha256_bytes(raw_billing_path.read_bytes())
    billing = {
        "schema_version": "duraseed-post-calibration-billing-v1",
        "status": "reconciled",
        "project_id": "project",
        "source_run_id": "smoke-run",
        "source_acceptance_sha256": smoke_hash,
        "raw_usage_sha256": raw_billing_hash,
        "raw_usage_cutoff_utc": "2026-08-13T12:00:00+00:00",
        "latest_calibration_finished_at_utc": "2026-08-13T11:00:00+00:00",
        "remaining_balance_usd": 1000.0,
        "protected_reserve_usd": 200.0,
        "remaining_balance_verified": True,
        "protected_reserve_survives": True,
        "pilot0_authorization_usd": 600.0,
        "raw_billing_entry_count": 1,
        "source_artifact_sha256s": [teacher_hash, acquisition_hash],
    }
    billing_path = tmp_path / "billing.json"
    billing_hash = _write(billing_path, billing)
    plaintext_hash = "6" * 64
    seal = {
        "ciphertext": base64.b64encode(bytes(16)).decode(),
        "declared_split": "b_test",
        "nonce": base64.b64encode(bytes(12)).decode(),
        "plaintext_sha256": plaintext_hash,
        "version": "duraseed-final-test-v1",
    }
    seal_path = tmp_path / "b-test.sealed"
    seal_hash = _write(seal_path, seal)
    dummy = "sha256:" + "7" * 64
    bundle = {
        "schema_version": "duraseed-pilot0-source-bundle-v1",
        "status": "ready_for_authorization",
        "project_id": "project",
        "git_commit": "commit",
        "resolved_config_hash": dummy,
        "completed_live_smoke_sha256": smoke_hash,
        "completed_live_smoke_run_id": "smoke-run",
        "completed_live_smoke_finished_at_utc": smoke_finished.isoformat(),
        "post_calibration_billing_sha256": billing_hash,
        "post_calibration_raw_billing_sha256": raw_billing_hash,
        "billing_cutoff_utc": billing["raw_usage_cutoff_utc"],
        "latest_calibration_finished_at_utc": billing[
            "latest_calibration_finished_at_utc"
        ],
        "uncommitted_grant_balance_usd": 1000.0,
        "protected_reserve_usd": 200.0,
        "m0_sampler_path": "tinker://m0/sampler_weights/a",
        "m0_state_path": "tinker://m0/weights/a",
        "m0_selection_sha256": selection_hash,
        "m0_ttl_sha256": ttl_hash,
        "panel_artifact_sha256": dummy,
        "teacher_recipe_artifact_sha256": teacher_hash,
        "acquisition_artifact_sha256": acquisition_hash,
        "stage_b_recipe_artifact_sha256": dummy,
        "visible_leakage_sha256": dummy,
        "sealed_b_test_envelope_sha256": seal_hash,
        "sealed_b_test_plaintext_sha256": f"sha256:{plaintext_hash}",
        "seed_sources": [],
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_hash = _write(bundle_path, bundle)
    authorization = {
        "schema_version": "duraseed-pilot0-launch-authorization-v1",
        "status": "accepted",
        "source_bundle_sha256": bundle_hash,
        "post_calibration_billing_sha256": billing_hash,
        "project_id": "project",
        "authorizer": "Ely",
        "authorized_at_utc": "2026-08-13T12:30:00+00:00",
        "authorized_usd": 600.0,
        "no_rerun_authorized": True,
    }
    authorization_path = tmp_path / "authorization.json"
    _write(authorization_path, authorization)
    kwargs = {
        "completed_live_smoke_path": tmp_path / "smoke.json",
        "post_calibration_billing_path": billing_path,
        "post_calibration_raw_billing_path": raw_billing_path,
        "m0_selection_path": tmp_path / "selection.json",
        "m0_ttl_path": tmp_path / "ttl.json",
    }
    auth = load_pilot0_source_authentication(
        bundle_path, authorization_path, seal_path, **kwargs
    )
    assert auth.bundle_sha256 == bundle_hash
    raw_billing_path.write_text('{"tampered":true}')
    with pytest.raises(RunnerGateError, match="unauthenticated"):
        load_pilot0_source_authentication(
            bundle_path, authorization_path, seal_path, **kwargs
        )
