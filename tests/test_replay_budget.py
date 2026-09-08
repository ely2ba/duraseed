"""Synthetic exact-token accounting and protected-balance preflight checks."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from duraseed.replay_budget import (
    build_preflight,
    storage_cost,
)


def inputs():
    block = {
        "per_arm_per_update_train_tokens": {"R-S": [64] * 294, "R-P": [96] * 294},
        "stage_b_per_update_train_tokens": [128] * 480,
        "prompt_lengths": {
            name: [10] * count
            for name, count in (
                ("a_cadence", 192),
                ("targeted_a_validation", 256),
                ("a_validation", 512),
                ("a_monitor", 384),
                ("b_validation", 512),
            )
        },
    }
    storage = dict(
        pairs=160,
        state_bytes=1_200_000_000,
        sampler_bytes=400_000_000,
        ttl_seconds=30 * 86400,
        backup_seconds=0,
        size_bound_verified=True,
        backup_policy_verified=True,
    )
    return {11: block, 29: block}, storage


def test_full_matrix_and_no_duplicate_origins():
    blocks, storage = inputs()
    result = build_preflight(blocks, storage=storage)
    assert result["token_budget"]["sample"] == 887_095_296
    assert result["token_budget"]["train"] == 2 * (294 * 160 + 2 * 480 * 128)
    candidates = [
        row
        for row in result["components"]
        if row["component"] == "candidate_assessments"
    ]
    assert sum(row["arms"] * row["points"] for row in candidates) == 12
    assert Decimal(result["storage"]["cost_usd"]) == Decimal("25.60")
    main = Decimal(result["main_usd"])
    assert Decimal(result["package_usd"]) == 20 + Decimal("1.10") * main
    assert result["blockers"] == ["LIVE_BALANCE_AND_OTHER_COMMITMENTS_UNVERIFIED"]


def test_actual_lengths_not_global_maximum():
    blocks, storage = inputs()
    first = build_preflight(blocks, storage=storage)
    blocks[11]["prompt_lengths"]["targeted_a_validation"][0] += 7
    second = build_preflight(blocks, storage=storage)
    # Shared test object changes both blocks; each has two arms, three candidates,16draws.
    assert (
        second["token_budget"]["prefill"] - first["token_budget"]["prefill"]
        == 7 * 2 * 2 * 3 * 16
    )


def test_package_overflow_never_cuts_schedule():
    blocks, storage = inputs()
    storage["backup_seconds"] = 10 * 365 * 86400
    result = build_preflight(blocks, storage=storage)
    assert result["status"] == "BUDGET_BLOCKED"
    assert result["token_budget"]["sample"] == 887_095_296


def test_storage_entire_ttl_and_backup():
    assert storage_cost(
        pairs=2,
        state_bytes=3_000_000_000,
        sampler_bytes=1_000_000_000,
        ttl_seconds=15 * 86400,
        backup_seconds=15 * 86400,
    ) == Decimal("0.8")
    with pytest.raises(ValueError, match="nonzero"):
        storage_cost(pairs=1, state_bytes=1, sampler_bytes=1, ttl_seconds=0)


def test_unverified_storage_and_balance_block_launch():
    blocks, storage = inputs()
    storage.update(size_bound_verified=False, backup_policy_verified=False)
    result = build_preflight(
        blocks, storage=storage, current_balance_usd="2000", other_committed_usd="1"
    )
    assert set(result["blockers"]) == {
        "CHECKPOINT_SIZE_BOUND_UNVERIFIED",
        "BACKUP_LIABILITY_UNVERIFIED",
        "PROTECTED_BALANCE_BLOCKED",
    }


@pytest.mark.parametrize("other", ["0", "51.27"])
def test_preflight_protected_balance_boundary_and_other_commitments(other):
    blocks, storage = inputs()
    initial = build_preflight(blocks, storage=storage)
    required = Decimal(initial["approval_ceiling_usd"]) + Decimal("1343.74")
    balance = required + Decimal(other)
    result = build_preflight(
        blocks,
        storage=storage,
        current_balance_usd=str(balance),
        other_committed_usd=other,
    )
    assert result["status"] == "budget_fits_not_authorized"
    assert result["blockers"] == []
    blocked = build_preflight(
        blocks,
        storage=storage,
        current_balance_usd=str(balance - Decimal("0.01")),
        other_committed_usd=other,
    )
    assert blocked["blockers"] == ["PROTECTED_BALANCE_BLOCKED"]


@pytest.mark.parametrize("field", ["current_balance_usd", "other_committed_usd"])
@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1"])
def test_preflight_rejects_nonfinite_or_negative_money(field, amount):
    blocks, storage = inputs()
    amounts = {"current_balance_usd": "4000", "other_committed_usd": "0"}
    amounts[field] = amount
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_preflight(blocks, storage=storage, **amounts)


def test_fixed_checkpoint_inventory():
    blocks, storage = inputs()
    storage["pairs"] -= 1
    with pytest.raises(ValueError, match="160"):
        build_preflight(blocks, storage=storage)


def test_panel_measurement_uses_family_ids_not_panel_label(monkeypatch):
    from tools import prepare_replay_budget as preparation

    rows = [
        SimpleNamespace(
            task_family="tces", intended_family=family, to_task=lambda: "prompt"
        )
        for family in ("target", "sentinel")
    ]
    manifest = SimpleNamespace(records=rows)
    source = SimpleNamespace(
        a_cadence=manifest,
        a_validation=manifest,
        b_validation=manifest,
        prompt_pools=SimpleNamespace(
            a_monitor_manifest=manifest,
            artifact=SimpleNamespace(
                boundary_family_ids=["target"], targeted_panel="PANEL_A"
            ),
        ),
    )
    runtime = SimpleNamespace(
        renderer=SimpleNamespace(
            build_generation_prompt=lambda *args, **kwargs: SimpleNamespace(length=10)
        )
    )
    monkeypatch.setattr(preparation, "tces_prompt", lambda task: task)
    monkeypatch.setattr(
        preparation, "stage_b_sources", lambda source: ["datum1", "datum2"]
    )
    monkeypatch.setattr(
        preparation, "measure_source", lambda runtime, row: {"train_tokens": 12}
    )
    result = preparation.measure_panels(runtime, source)
    assert result["prompt_lengths"]["targeted_a_validation"] == [10]
    assert result["stage_b_per_update_train_tokens"] == [32 * 12] * 480
