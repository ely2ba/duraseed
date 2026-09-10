"""Synthetic scheduling checks for the bounded dense follow-up; no model calls."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from duraseed.replay_remote import write_json

SPEC = importlib.util.spec_from_file_location(
    "dense_runner", Path(__file__).parents[1] / "tools/run_dense_retention.py"
)
dense = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense)


def fixture(tmp_path, monkeypatch, fail_at=None):
    calls = []
    source = NS(seed=11, prompt_pools=NS(a_monitor_manifest=object()))
    records = object()
    for arm, step in dense.SELECTED.items():
        write_json(
            tmp_path / f"seed-11/{arm}/stage_a/u{step}/checkpoint.json",
            {
                "state_path": f"synthetic-{arm}-initial-state",
                "sampler_path": f"synthetic-{arm}-initial-sampler",
            },
        )

    async def train(
        remote, passed_source, arm, stage, passed_records, previous, start, stop
    ):
        assert passed_source is source and passed_records is records
        assert stage == "stage_b" and stop == start + 1
        assert previous["state_path"] == (
            f"synthetic-{arm}-initial-state"
            if start == 0
            else f"synthetic-{arm}-{start}-state"
        )
        calls.append(("train", arm, stop))
        return {
            "state_path": f"synthetic-{arm}-{stop}-state",
            "sampler_path": f"synthetic-{arm}-{stop}-sampler",
        }

    async def evaluate(remote, passed_source, checkpoint, manifest, **kw):
        arm, stop = kw["arm"], kw["update"]
        assert (
            passed_source is source
            and manifest is source.prompt_pools.a_monitor_manifest
        )
        assert checkpoint["sampler_path"] == f"synthetic-{arm}-{stop}-sampler"
        assert checkpoint["origin_sampler_path"] == f"synthetic-{arm}-initial-sampler"
        assert (kw["draws"], kw["cap"], kw["purpose"]) == (4, 4096, "a_monitor")
        calls.append(("eval", arm, stop))
        if (arm, stop) == fail_at:
            raise RuntimeError("synthetic pending evaluation")
        return {"row_count": 1536}

    monkeypatch.setattr(dense, "train_segment", train)
    monkeypatch.setattr(dense, "evaluate", evaluate)
    return NS(root=tmp_path), source, records, calls


def test_every_update_evaluated_in_order_with_correct_origin(tmp_path, monkeypatch):
    remote, source, records, calls = fixture(tmp_path, monkeypatch)
    asyncio.run(dense.run_grid(remote, source, records))
    assert calls == [
        (operation, arm, update)
        for arm in ("R-S", "R-P")
        for update in range(1, 21)
        for operation in ("train", "eval")
    ]


def test_failed_evaluation_does_not_repeat_or_dispatch_next_update(
    tmp_path, monkeypatch
):
    remote, source, records, calls = fixture(tmp_path, monkeypatch, fail_at=("R-S", 2))
    with pytest.raises(RuntimeError, match="pending evaluation"):
        asyncio.run(dense.run_grid(remote, source, records))
    assert calls == [
        ("train", "R-S", 1),
        ("eval", "R-S", 1),
        ("train", "R-S", 2),
        ("eval", "R-S", 2),
    ]


def test_second_dispatch_is_refused_before_creating_a_session(tmp_path):
    (tmp_path / "remote").mkdir()
    with pytest.raises(RuntimeError, match="already dispatched"):
        asyncio.run(dense.execute(tmp_path, tmp_path))
