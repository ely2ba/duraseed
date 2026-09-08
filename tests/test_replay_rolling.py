"""Synthetic scheduling, reservation, and clean-handoff checks; no remote calls."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from duraseed import replay_rolling as rolling
from duraseed.replay_remote import write_json
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.runtime.ledger import ReservationError


def sampling_fixture(tmp_path, monkeypatch, *, budget=1000):
    inputs = NS(ledger=TokenLedger(TokenBudget(budget, budget, 0), 10.0))
    records = [NS(task_id=f"synthetic-{i}") for i in range(8)]
    monkeypatch.setattr(
        rolling.groups, "_reservation", lambda *a, **kw: TokenBudget(2, 12, 0)
    )
    arguments = dict(
        pending=list(enumerate(records)),
        output=tmp_path,
        contracts={r.task_id: {"prompt_text": r.task_id} for r in records},
        samples_per_item=4,
        max_tokens=128,
        manifest=object(),
        sampler=object(),
        coordinates=object(),
        sample_index_start=0,
        temperature=1.0,
        top_p=0.95,
    )
    return inputs, arguments


def test_refill_does_not_wait_for_slowest_and_preserves_requests(tmp_path, monkeypatch):
    inputs, arguments = sampling_fixture(tmp_path, monkeypatch)
    calls, completed, active, maximum = [], [], 0, 0

    async def run():
        fifth_started = asyncio.Event()

        async def sample(inputs, **kw):
            nonlocal active, maximum
            index = kw["index"]
            calls.append(index)
            active += 1
            maximum = max(maximum, active)
            for key in (
                "manifest",
                "sampler",
                "coordinates",
                "samples_per_item",
                "sample_index_start",
                "max_tokens",
                "temperature",
                "top_p",
            ):
                assert kw[key] == arguments[key]
            assert kw["contract"] == arguments["contracts"][kw["record"].task_id]
            kw["ledger"].reserve_call(TokenBudget(2, 12, 0))
            if index == 0:
                await fifth_started.wait()
            elif index == 4:
                fifth_started.set()
            await asyncio.sleep(0)
            kw["ledger"].settle_call(TokenBudget(2, 3, 0))
            active -= 1
            completed.append(index)
            return index, (kw["record"].task_id,)

        monkeypatch.setattr(rolling.groups, "_sample_group", sample)
        return await asyncio.wait_for(
            rolling.sample_manifest_groups(inputs, **arguments), timeout=1
        )

    result = asyncio.run(run())
    assert sorted(calls) == list(range(8)) and len(calls) == 8
    assert maximum == 4 and active == 0
    assert completed.index(1) < completed.index(0) and 4 in calls
    assert result == {i: (f"synthetic-{i}",) for i in range(8)}
    assert inputs.ledger.committed == TokenBudget(16, 96, 0)
    assert inputs.ledger.observed == TokenBudget(16, 24, 0)
    assert not inputs.ledger.has_pending_call


def test_failure_drains_submitted_groups_without_cancel_retry_or_new_work(
    tmp_path, monkeypatch
):
    inputs, arguments = sampling_fixture(tmp_path, monkeypatch)
    calls, finished = [], []

    async def run():
        submitted, failed = asyncio.Event(), asyncio.Event()

        async def sample(inputs, **kw):
            index = kw["index"]
            calls.append(index)
            if len(calls) == 4:
                submitted.set()
            await submitted.wait()
            if index == 0:
                failed.set()
                raise RuntimeError("synthetic ambiguous request")
            await failed.wait()
            await asyncio.sleep(0.01)
            kw["ledger"].reserve_call(TokenBudget(2, 12, 0))
            kw["ledger"].settle_call(TokenBudget(2, 3, 0))
            finished.append(index)
            return index, (index,)

        monkeypatch.setattr(rolling.groups, "_sample_group", sample)
        await rolling.sample_manifest_groups(inputs, **arguments)

    with pytest.raises(RuntimeError, match="ambiguous"):
        asyncio.run(run())
    assert calls == [0, 1, 2, 3] and sorted(finished) == [1, 2, 3]
    assert inputs.ledger.observed == inputs.ledger.committed == TokenBudget(16, 96, 0)
    assert not inputs.ledger.has_pending_call


def test_unaffordable_panel_submits_nothing(tmp_path, monkeypatch):
    inputs, arguments = sampling_fixture(tmp_path, monkeypatch, budget=8)

    async def forbidden(*args, **kwargs):
        pytest.fail("sampling started before the full panel was reserved")

    monkeypatch.setattr(rolling.groups, "_sample_group", forbidden)
    with pytest.raises(ReservationError):
        asyncio.run(rolling.sample_manifest_groups(inputs, **arguments))


def boundary_fixture(root, monkeypatch):
    def exited(pid, signal):
        raise ProcessLookupError

    monkeypatch.setattr(rolling.os, "kill", exited)
    coordinate = {"seed": 11, "replay_arm": "R-S", "update": 10, "purpose": "a_monitor"}
    write_json(
        root / "progress.json", {"phase": "evaluation", "pending": False, **coordinate}
    )
    stage = root / "seed-11/R-S/stage_b/u10"
    write_json(stage / "checkpoint.json", {"update": 10})
    write_json(stage / "update-10.json", {})
    write_json(
        root / "billing.json",
        {
            "committed": {"prefill": 1, "sample": 2, "train": 3},
            "committed_fixed_usd": 0.25,
        },
    )
    write_json(
        root / "remote/remote-call-state.json",
        {
            "pending": None,
            "completed_count": 40,
            "reserved_floor": {
                "prefill_tokens": 1,
                "sample_tokens": 2,
                "train_tokens": 3,
                "fixed_usd": 0.25,
            },
        },
    )
    return stage


def test_clean_handoff_rejects_live_owner_pending_calls_and_incomplete_training(
    tmp_path, monkeypatch
):
    stage = boundary_fixture(tmp_path, monkeypatch)
    result = rolling.clean_boundary(tmp_path, 123)
    assert result["boundary"]["update"] == 10 and result["completed_calls"] == 40
    with monkeypatch.context() as patch:
        patch.setattr(rolling.os, "kill", lambda *args: None)
        with pytest.raises(ValueError, match="still alive"):
            rolling.clean_boundary(tmp_path, 123)
    partial = stage / "a_monitor/remote-call-state.json"
    write_json(partial, {"pending": {"operation": "synthetic"}})
    with pytest.raises(ValueError, match="pending"):
        rolling.clean_boundary(tmp_path, 123)
    write_json(partial, {"pending": None})
    write_json(stage.parent / "u20/update-11.json", {})
    with pytest.raises(ValueError, match="incomplete training"):
        rolling.clean_boundary(tmp_path, 123)
