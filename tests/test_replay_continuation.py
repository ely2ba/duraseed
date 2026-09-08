"""Synthetic-only matching-revision, budget and continuation-path checks."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from duraseed import replay_continuation as continuation
from duraseed.replay_matching import ARMS, TARGETS, match
from duraseed.replay_remote import write_json


def original_matches():
    result = {}
    for seed, counts in (
        (11, ((1504, 1700, 1900), (1503, 1600, 1800))),
        (29, ((732, 731, 722), (1537, 1676, 1838))),
    ):
        nomination = {
            "seed": seed,
            "target": str(TARGETS[seed]),
            "nominees": {arm: [{"update": u} for u in (10, 20, 30)] for arm in ARMS},
        }
        assessments = {
            arm: [
                {"update": u, "successes": c, "trials": 4096}
                for u, c in zip((10, 20, 30), arm_counts)
            ]
            for arm, arm_counts in zip(ARMS, counts)
        }
        result[str(seed)] = match(nomination, assessments)
    return result


def test_only_anchor_is_removed_original_records_and_tie_breaks_survive():
    original = original_matches()
    before = deepcopy(original)
    revised = continuation.revise_matching(original)
    assert original == before
    assert all(row["status"] == "NO_MATCH" for row in original.values())
    assert revised["11"]["selected"]["R-S"] == 10
    assert revised["11"]["selected"]["R-P"] == 10
    assert revised["29"]["selected"] is None
    assert revised["29"]["stage_b_allowed"] is False
    for seed in original:
        for old, new in zip(
            original[seed]["combinations"], revised[seed]["combinations"]
        ):
            assert {k: v for k, v in old.items() if k != "eligible"} == {
                k: v for k, v in new.items() if k != "eligible"
            }


@pytest.mark.parametrize("gap, eligible", [(122, True), (123, False)])
def test_between_arm_three_point_boundary_remains_exact(gap, eligible):
    original = original_matches()
    row = original["11"]["combinations"][0]
    row["R-P_successes"] = row["R-S_successes"] + gap
    revised = continuation.revise_matching(original)
    assert revised["11"]["combinations"][0]["eligible"] is eligible


def budget_fixture(tmp_path):
    for directory in (tmp_path, tmp_path / "engineering-only"):
        write_json(
            directory / "billing.json",
            {
                "committed": {"prefill": 100, "sample": 1000, "train": 100},
                "observed": {"prefill": 100, "sample": 100, "train": 100},
                "committed_fixed_usd": 1,
                "observed_fixed_usd": 1,
            },
        )
        write_json(
            directory / "remote/remote-call-state.json",
            {
                "pending": None,
                "reserved_floor": {
                    "prefill_tokens": 100,
                    "sample_tokens": 1000,
                    "train_tokens": 100,
                    "fixed_usd": 1,
                },
            },
        )
    original = {
        "components": [
            {
                "component": name,
                "seed": seed,
                "prefill": 100,
                "sample": 1000,
                "train": 10,
            }
            for seed in (11, 29)
            for name in continuation.COMPONENTS
        ],
        "approval_ceiling_usd": "100",
        "recovery_reservation_usd": "10",
        "prices_per_million_usd": {},
        "blockers": ["BACKUP_LIABILITY_UNVERIFIED"],
    }
    return original, continuation.revise_matching(original_matches())


def test_budget_prices_only_matched_stage_b_and_counts_prior_reservations(tmp_path):
    original, matching = budget_fixture(tmp_path)
    result = continuation.continuation_preflight(
        tmp_path,
        original,
        matching,
        {"storage_per_pair_usd": 0.25},
    )
    assert len(result["components"]) == 4
    assert result["checkpoint_pairs"] == 20
    assert result["storage_allowance_usd"] == "5.00"
    assert result["token_budget"] == {"prefill": 400, "sample": 4000, "train": 40}
    original["approval_ceiling_usd"] = "17"
    with pytest.raises(ValueError, match="approved package ceiling"):
        continuation.continuation_preflight(
            tmp_path,
            original,
            matching,
            {"storage_per_pair_usd": 0.25},
        )
    original["approval_ceiling_usd"] = "100"
    path = tmp_path / "remote/remote-call-state.json"
    value = continuation.read_json(path)
    value["pending"] = {"operation": "synthetic-ambiguous"}
    write_json(path, value)
    with pytest.raises(ValueError, match="pending"):
        continuation.continuation_preflight(
            tmp_path,
            original,
            matching,
            {"storage_per_pair_usd": 0.25},
        )


def test_continuation_binds_renderer_and_runs_only_selected_stage_b(
    monkeypatch, tmp_path
):
    events = []

    class Remote:
        def __init__(self, *args):
            events.append("session")

        async def restore(self, path, *, full_state, coordinate):
            assert full_state is False and coordinate["stage"] == "stage_b_setup"
            events.append(("renderer", path))

        def snapshot(self):
            return {"synthetic": True}

    async def stage_b(remote, source, arm, update):
        events.append(("stage_b", source.seed, arm, update))

    matching = continuation.revise_matching(original_matches())
    write_json(
        tmp_path / "seed-11/R-S/stage_a/u10/checkpoint.json",
        {"state_path": "synthetic-selected-state"},
    )
    monkeypatch.setattr(continuation, "ReplayRemote", Remote)
    monkeypatch.setattr(continuation, "stage_b", stage_b)
    monkeypatch.setattr(
        continuation, "write_report", lambda root: events.append("report")
    )
    config = {"project_id": "synthetic", "continuation": {"synthetic": True}}
    asyncio.run(
        continuation.execute(tmp_path, tmp_path, config, matching, {11: NS(seed=11)})
    )
    assert events == [
        "session",
        ("renderer", "synthetic-selected-state"),
        ("stage_b", 11, "R-S", 10),
        ("stage_b", 11, "R-P", 10),
        "report",
    ]
    events.clear()
    asyncio.run(
        continuation.execute(tmp_path, tmp_path, config, matching, {11: NS(seed=11)})
    )
    assert events == ["report"]
