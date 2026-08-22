from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.pilot0_contract import BG_STAGE_A_GRID, BS_STAGE_A_GRID, STAGE_B_GRID
from duraseed.runners import pilot0_stage_a, pilot0_stage_b, pilot0_updates
from duraseed.runners.remote_journal import RemoteJournal


@pytest.mark.parametrize(
    ("method", "grid"), (("B-S", BS_STAGE_A_GRID), ("B-G", BG_STAGE_A_GRID))
)
def test_stage_a_branch_executes_every_adjacent_grid_segment(
    monkeypatch, tmp_path: Path, method: str, grid: tuple[int, ...]
) -> None:  # type: ignore[no-untyped-def]
    calls = []

    async def segment(*args, start: int, stop: int, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((start, stop))
        return {"sampler_path": f"sampler-{stop}", "state_path": f"state-{stop}"}

    monkeypatch.setattr(pilot0_stage_a, "ordered_stage_a_pools", lambda source: {})
    monkeypatch.setattr(pilot0_stage_a, "stage_a_solver_sources", lambda source: {})
    monkeypatch.setattr(pilot0_stage_a, "_branch_segment", segment)
    result = asyncio.run(
        pilot0_stage_a._branch(
            SimpleNamespace(),
            SimpleNamespace(seed=11),
            {"sampler_path": "origin-sampler", "state_path": "origin-state"},
            method=method,
            output=tmp_path,
            preflight_sha256="sha256:" + "1" * 64,
        )
    )

    expected = list(zip(grid[:-1], grid[1:], strict=True))
    assert calls == expected
    assert tuple(result["segments"]) == tuple(str(stop) for _, stop in expected)


def test_stage_b_probe_executes_every_adjacent_grid_segment(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    calls = []

    async def stage_zero(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"sampler_path": "sampler-0", "state_path": "state-0"}

    async def segment(*args, start: int, stop: int, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((start, stop))
        return {"sampler_path": f"sampler-{stop}", "state_path": f"state-{stop}"}

    monkeypatch.setattr(pilot0_stage_b, "stage_b_sources", lambda source: ())
    monkeypatch.setattr(pilot0_stage_b, "_stage_zero", stage_zero)
    monkeypatch.setattr(pilot0_stage_b, "_train_segment", segment)
    result = asyncio.run(
        pilot0_stage_b.run_stage_b(
            SimpleNamespace(
                runtime=object(),
                config=SimpleNamespace(
                    stage_b=SimpleNamespace(selected_profile="shortest2_cap2")
                ),
            ),
            SimpleNamespace(seed=11),
            {
                "selected_sampler_path": "stage-a-sampler",
                "selected_state_path": "stage-a-state",
            },
            method="B-S",
            output=tmp_path,
            preflight_sha256="sha256:" + "1" * 64,
        )
    )

    expected = list(zip(STAGE_B_GRID[:-1], STAGE_B_GRID[1:], strict=True))
    assert calls == expected
    assert tuple(result["segments"]) == ("0", *(str(stop) for _, stop in expected))


def test_b_s_update_restarts_the_frozen_schedule_each_epoch(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    record = SimpleNamespace(task_id="task")
    scheduled_steps = []

    def scheduled(pools, order, step):  # type: ignore[no-untyped-def]
        scheduled_steps.append(step)
        return (record,)

    async def update(*args, **kwargs):  # type: ignore[no-untyped-def]
        return {"local.train_tokens": 1.0}

    monkeypatch.setattr(pilot0_updates, "scheduled_stage_a_records", scheduled)
    monkeypatch.setattr(
        pilot0_updates,
        "sft_datum",
        lambda *args: SimpleNamespace(model_input=SimpleNamespace(length=1)),
    )
    monkeypatch.setattr(pilot0_updates, "apply_update", update)
    asyncio.run(
        pilot0_updates.supervised_update(
            SimpleNamespace(ledger=object()),
            SimpleNamespace(
                seed=11,
                prompt_pools=SimpleNamespace(
                    artifact=SimpleNamespace(bs_slot_order=("boundary",))
                ),
            ),
            SimpleNamespace(),
            step=50,
            learning_rate=1e-4,
            pools={},
            sources={"task": object()},
            output=tmp_path,
            journal=RemoteJournal(tmp_path),
        )
    )
    assert scheduled_steps == [1]
