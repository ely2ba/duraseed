from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from duraseed.pilot0_contract import STAGE_A_GRID, STAGE_B_GRID
from duraseed.runners import pilot0_stage_a, pilot0_stage_b


def test_stage_a_branch_executes_every_adjacent_grid_segment(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    calls = []

    async def segment(*args, start: int, stop: int, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((start, stop))
        return {"sampler_path": f"sampler-{stop}", "state_path": f"state-{stop}"}

    monkeypatch.setattr(pilot0_stage_a, "ordered_stage_a_pools", lambda source: {})
    monkeypatch.setattr(pilot0_stage_a, "_branch_segment", segment)
    result = asyncio.run(
        pilot0_stage_a._branch(
            SimpleNamespace(),
            SimpleNamespace(seed=11),
            {"sampler_path": "origin-sampler", "state_path": "origin-state"},
            method="B-G",
            output=tmp_path,
            preflight_sha256="sha256:" + "1" * 64,
        )
    )

    expected = list(zip(STAGE_A_GRID[:-1], STAGE_A_GRID[1:], strict=True))
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
