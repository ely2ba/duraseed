"""Synthetic new-pair dispatch order; never makes a Tinker call."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from duraseed.replay_matching import STAGE_A_GRID
from duraseed.replay_remote import write_json

SPEC = importlib.util.spec_from_file_location(
    "replication_runner", Path(__file__).parents[1] / "tools/run_replay_replication.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(tmp_path, monkeypatch, *, matched=True, failed=False):
    events = []
    source = NS(seed=47)
    records = {"R-S": object(), "R-P": object()}
    remote = NS(root=tmp_path, snapshot=lambda: {"synthetic": True})

    async def acquire(remote, passed_source, arm, passed_records):
        assert source is passed_source and records[arm] is passed_records
        events.append(("acquire", arm))
        for step in STAGE_A_GRID:
            write_json(
                tmp_path / f"seed-47/{arm}/stage_a/u{step}/checkpoint.json",
                {"sampler_path": f"synthetic-{arm}-{step}"},
            )
        return [
            {"update": step, "successes": 31, "trials": 96} for step in STAGE_A_GRID
        ]

    async def assess(remote, passed_source, checkpoint, manifest, **kw):
        assert events[:2] == [("acquire", "R-S"), ("acquire", "R-P")]
        assert (tmp_path / "nominations.json").exists()
        assert not (tmp_path / "matching.json").exists()
        assert (kw["stage"], kw["draws"], kw["cap"]) == ("stage_a", 16, 4096)
        assert kw["purpose"] == "candidate"
        assert checkpoint["sampler_path"] == f"synthetic-{kw['arm']}-{kw['update']}"
        events.append(("candidate", kw["arm"], kw["update"]))
        if failed:
            raise RuntimeError("synthetic pending candidate")
        return {
            "item_counts": [
                {
                    "panel_role": "targeted",
                    "successes": 1200 if matched or kw["arm"] == "R-S" else 1600,
                    "trials": 4096,
                }
            ]
        }

    async def followup(remote, passed_source, arm, selected):
        selection = runner.read_json(tmp_path / "matching.json")["47"]
        assert selection["stage_b_allowed"]
        assert sum(row[0] == "candidate" for row in events) == 6
        assert selected == selection["selected"][arm]
        events.append(("stage_b", arm, selected))

    monkeypatch.setattr(runner, "stage_a", acquire)
    monkeypatch.setattr(runner, "candidate_manifest", lambda source: object())
    monkeypatch.setattr(runner, "evaluate", assess)
    monkeypatch.setattr(runner, "stage_b", followup)
    return remote, source, records, events


@pytest.mark.parametrize("matched", [True, False])
def test_both_acquisitions_and_all_candidates_precede_stage_b(
    tmp_path, monkeypatch, matched
):
    remote, source, records, events = fixture(tmp_path, monkeypatch, matched=matched)
    result = asyncio.run(runner.run_pair(remote, source, records))
    assert result["status"] == ("COMPLETED" if matched else "NO_MATCH")
    assert len(events) == (10 if matched else 8)
    assert not any(row[0] == "stage_b" for row in events) if not matched else True
    assert result["matching"] == runner.read_json(tmp_path / "matching.json")
    assert result["source_block"] == 11 and result["acquisition_order_seed"] == 47


def test_pending_candidate_is_not_retried_and_blocks_stage_b(tmp_path, monkeypatch):
    remote, source, records, events = fixture(tmp_path, monkeypatch, failed=True)
    with pytest.raises(RuntimeError, match="pending candidate"):
        asyncio.run(runner.run_pair(remote, source, records))
    assert len(events) == 3 and events[-1] == ("candidate", "R-S", 10)
    assert not (tmp_path / "matching.json").exists()


def test_existing_dispatch_is_never_relaunched(tmp_path):
    (tmp_path / "remote").mkdir()
    with pytest.raises(RuntimeError, match="already dispatched"):
        asyncio.run(runner.execute(tmp_path, tmp_path))


def test_conversion_cache_preserves_inputs_and_invalidates_at_boundaries(monkeypatch):
    remote = object.__new__(runner.CachedDatumRemote)
    remote.datums, remote.renderer_signature = {}, None
    tokenizer_text, restores, conversions = ["original"], [], []
    runtime = NS(
        renderer=object(),
        tokenizer=NS(
            backend_tokenizer=NS(to_str=lambda: tokenizer_text[0]),
            special_tokens_map={"eos_token": "synthetic-stop"},
        ),
    )

    async def restore(self, path, **kwargs):
        restores.append((path, kwargs))
        self.runtime = runtime

    def convert(runtime, source, *, max_length):
        value = (source.prompt_text, source.verified_completion_text, max_length)
        conversions.append(value)
        return NS(unchanged=value)

    monkeypatch.setattr(runner.ReplayRemote, "restore", restore)
    monkeypatch.setattr(runner, "sft_datum", convert)
    source = NS(prompt_text="same prompt", verified_completion_text="full trace")
    asyncio.run(remote.restore("origin", full_state=False, coordinate={}))
    first = remote.datum(runtime, source, max_length=4217)
    asyncio.run(remote.restore("next", full_state=True, coordinate={}))
    assert remote.datum(runtime, source, max_length=4217) is first
    assert len(conversions) == 1 and len(restores) == 2
    changed = NS(prompt_text="changed prompt", verified_completion_text="full trace")
    assert remote.datum(runtime, changed, max_length=4217) is not first
    tokenizer_text[0] = "changed tokenizer normalization"
    asyncio.run(remote.restore("third", full_state=True, coordinate={}))
    assert not remote.datums
    second = remote.datum(runtime, source, max_length=4217)
    asyncio.run(remote.restore("new-arm", full_state=False, coordinate={}))
    assert remote.datum(runtime, source, max_length=4217) is not second
    assert len(restores) == 4
