"""Synthetic-only replay execution ordering; no SDK, credentials, or sampling."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from duraseed import replay_evaluation, replay_inputs
from duraseed.replay_matching import ARMS, STAGE_A_GRID, STAGE_B_GRID
from duraseed.replay_remote import write_json
from duraseed.runners import replay


def manifest(name, count):
    return NS(manifest_id=name, record_count=count, records=tuple(range(count)))


def source(seed):
    return NS(
        seed=seed,
        a_cadence=manifest("cadence", 192),
        a_validation=manifest("validation", 512),
        b_validation=manifest("maps", 512),
        prompt_pools=NS(a_monitor_manifest=manifest("monitor", 384)),
    )


class FakeRemote:
    def __init__(self, root):
        self.repo = self.root = root
        self.run_id = "synthetic-replay"
        self.events = []
        self.runtime = NS(renderer=None)
        self.config = {
            "max_length": 5000,
            "stage_a_lr": 1e-4,
            "stage_b_lr": 3e-4,
            "blocks": {
                str(seed): {"m0_state_path": "synthetic-m0", "m0_sampler_path": "m0"}
                for seed in (11, 29)
            },
        }

    async def restore(self, path, *, full_state, coordinate, sources=()):
        self.events.append(
            {
                "kind": "restore",
                "full_state": full_state,
                "sources": len(sources),
                "path": path,
                **coordinate,
            }
        )
        self.runtime.renderer = object()

    async def update(self, datums, lr, path, coordinate):
        self.events.append(
            {"kind": "train", "datums": tuple(datums), "lr": lr, **coordinate}
        )

    async def save(self, name, path, coordinate):
        value = {**coordinate, "state_path": name + ":state", "sampler_path": name}
        write_json(path, value)
        self.events.append({"kind": "save", **coordinate})
        return value

    def snapshot(self):
        return {"synthetic": True, "observed_usd": 0}


def setup_runner(monkeypatch, tmp_path, no_match=()):
    remote = FakeRemote(tmp_path)
    monkeypatch.setattr(
        replay,
        "block_inputs",
        lambda repo, config, seed: (
            source(seed),
            {arm: tuple(range(41)) for arm in ARMS},
        ),
    )
    monkeypatch.setattr(replay, "stage_b_sources", lambda source: tuple(range(43)))
    monkeypatch.setattr(replay, "sft_datum", lambda runtime, row, max_length: row)
    monkeypatch.setattr(
        replay, "candidate_manifest", lambda source: manifest("candidate", 256)
    )
    monkeypatch.setattr(replay, "write_profile", lambda *args: None)

    async def evaluation(remote, source, checkpoint, manifest, **options):
        stage, purpose = options["stage"], options["purpose"]
        if purpose == "candidate":
            assert (tmp_path / "nominations.json").exists()
            assert (
                len(
                    [
                        e
                        for e in remote.events
                        if e["kind"] == "save"
                        and e["stage"] == "stage_a"
                        and e["update"] == 294
                    ]
                )
                == 4
            )
        if stage == "stage_b":
            decisions = replay.read_json(tmp_path / "matching.json")
            assert set(decisions) == {"11", "29"}
            assert decisions[str(source.seed)]["stage_b_allowed"]
            assert (
                len(
                    [
                        e
                        for e in remote.events
                        if e["kind"] == "eval" and e["purpose"] == "candidate"
                    ]
                )
                == 12
            )
        remote.events.append(
            {
                "kind": "eval",
                "seed": source.seed,
                "manifest": manifest.manifest_id,
                **options,
            }
        )
        trials = 96 if purpose == "cadence" else 4096
        successes = (
            (31 if source.seed == 11 else 17)
            if trials == 96
            else (1323 if source.seed == 11 else 725)
        )
        if purpose == "candidate" and source.seed in no_match:
            successes = 0
        return {
            "item_counts": [
                {"panel_role": "targeted", "successes": successes, "trials": trials}
            ]
        }

    monkeypatch.setattr(replay, "evaluate", evaluation)
    return remote


@pytest.mark.parametrize("no_match", [(), (29,), (11, 29)])
def test_four_acquisitions_and_all_nominees_precede_any_stage_b(
    monkeypatch, tmp_path, no_match
):
    remote = setup_runner(monkeypatch, tmp_path, no_match)
    result = asyncio.run(replay.run_package(remote))
    assert result["status"] == ("NO_MATCH" if len(no_match) == 2 else "COMPLETED")
    for seed in (11, 29):
        for arm in ARMS:
            train = [
                e
                for e in remote.events
                if e["kind"] == "train" and e["seed"] == seed and e["replay_arm"] == arm
            ]
            a = [e for e in train if e["stage"] == "stage_a"]
            assert [e["update"] for e in a] == list(range(1, 295))
            assert {e["lr"] for e in a} == {1e-4}
            assert all(
                e["datums"]
                == tuple(((e["update"] - 1) * 32 + j) % 41 for j in range(32))
                for e in a
            )
            b = [e for e in train if e["stage"] == "stage_b"]
            assert [e["update"] for e in b] == (
                [] if seed in no_match else list(range(1, 481))
            )
            assert all(
                e["lr"] == 3e-4
                and e["datums"]
                == tuple(((e["update"] - 1) * 32 + j) % 43 for j in range(32))
                for e in b
            )
            retained = [
                e["update"]
                for e in remote.events
                if e["kind"] == "save"
                and e["seed"] == seed
                and e["replay_arm"] == arm
                and e["stage"] == "stage_a"
            ]
            assert retained == list(STAGE_A_GRID)
    nominees = replay.read_json(tmp_path / "nominations.json")
    assert sum(len(v) for b in nominees.values() for v in b["nominees"].values()) == 12


def test_eval_grid_caps_and_selected_origin_are_not_duplicated(monkeypatch, tmp_path):
    remote = setup_runner(monkeypatch, tmp_path)
    asyncio.run(replay.run_package(remote))
    for seed in (11, 29):
        for arm in ARMS:
            rows = [
                e
                for e in remote.events
                if e["kind"] == "eval" and e["seed"] == seed and e["arm"] == arm
            ]
            cadence = [r for r in rows if r["purpose"] == "cadence"]
            assert [r["update"] for r in cadence] == list(STAGE_A_GRID)
            assert {(r["draws"], r["cap"]) for r in cadence} == {(1, 4096)}
            candidates = [r for r in rows if r["purpose"] == "candidate"]
            assert [r["update"] for r in candidates] == [10, 20, 30]
            assert {(r["draws"], r["cap"]) for r in candidates} == {(16, 4096)}
            for purpose, grid, draws, cap in (
                ("a_monitor", STAGE_B_GRID, 4, 4096),
                ("b_validation", STAGE_B_GRID, 16, 128),
                ("a_validation", (0, 480), 16, 4096),
            ):
                panel = [r for r in rows if r["purpose"] == purpose]
                assert [r["update"] for r in panel] == list(grid)
                assert {(r["draws"], r["cap"]) for r in panel} == {(draws, cap)}
                assert len([r for r in panel if r["update"] == 0]) == 1


def test_optimizer_is_fresh_on_each_entry_then_full_state_per_segment(
    monkeypatch, tmp_path
):
    remote = setup_runner(monkeypatch, tmp_path)
    asyncio.run(replay.run_package(remote))
    for seed in (11, 29):
        for arm in ARMS:
            for stage, grid, population in (
                ("stage_a", STAGE_A_GRID, 41),
                ("stage_b", STAGE_B_GRID[1:], 43),
            ):
                rows = [
                    e
                    for e in remote.events
                    if e["kind"] == "restore"
                    and e["seed"] == seed
                    and e["replay_arm"] == arm
                    and e["stage"] == stage
                ]
                assert [r["update"] for r in rows] == list(grid)
                assert [r["full_state"] for r in rows] == [False] + [True] * (
                    len(grid) - 1
                )
                assert [r["sources"] for r in rows] == [population] + [0] * (
                    len(grid) - 1
                )
                if stage == "stage_a":
                    assert rows[0]["path"] == "synthetic-m0"
                else:
                    assert "stage_a-u10:state" in rows[0]["path"]


def test_clean_segment_resume_does_not_repeat_updates(monkeypatch, tmp_path):
    remote = setup_runner(monkeypatch, tmp_path)
    previous = {"state_path": "synthetic-m0"}
    args = (remote, source(11), "R-S", "stage_a", tuple(range(41)), previous, 0, 10)
    saved = asyncio.run(replay.train_segment(*args))
    before = len(remote.events)
    assert asyncio.run(replay.train_segment(*args)) == saved
    assert len(remote.events) == before
    previous["state_path"] = "different-origin"
    with pytest.raises(ValueError, match="lineage"):
        asyncio.run(replay.train_segment(*args))
    assert len(remote.events) == before


def test_incomplete_segment_is_not_automatically_replayed(monkeypatch, tmp_path):
    remote = setup_runner(monkeypatch, tmp_path)
    write_json(tmp_path / "seed-11/R-S/stage_a/u10/update-1.json", {"synthetic": True})
    with pytest.raises(ValueError, match="incomplete training segment"):
        asyncio.run(
            replay.train_segment(
                remote,
                source(11),
                "R-S",
                "stage_a",
                tuple(range(41)),
                {"state_path": "synthetic-m0"},
                0,
                10,
            )
        )
    assert remote.events == []


def test_candidate_manifest_uses_targeted_family_ids_not_panel_label(monkeypatch):
    records = tuple(NS(intended_family="family-targeted") for _ in range(256)) + tuple(
        NS(intended_family="family-sentinel") for _ in range(256)
    )
    original = NS(
        records=records,
        split="a_validation",
        generator_version="synthetic",
        root_seed=11,
        manifest_id="original",
    )
    sample = NS(
        a_validation=original,
        prompt_pools=NS(
            artifact=NS(targeted_panel="A", boundary_family_ids=("family-targeted",))
        ),
    )
    monkeypatch.setattr(replay_inputs, "build_manifest", lambda **kwargs: kwargs)
    value = replay_inputs.candidate_manifest(sample)
    assert len(value["records"]) == 256
    assert value["parent_manifest_id"] == "original"
    assert all(r.intended_family == "family-targeted" for r in value["records"])


def test_evaluation_namespaces_pair_arms_but_separate_candidate_and_f3(
    monkeypatch, tmp_path
):
    calls = []
    renderer = NS(build_generation_prompt=lambda *a, **kw: NS(length=7))

    async def sampler(*args):
        return object()

    async def collect(*args, **kwargs):
        calls.append(kwargs)
        return {"row_count": 16}

    remote = NS(
        runtime=NS(renderer=renderer),
        sampler=sampler,
        eval_inputs=object(),
        begin=lambda *args: None,
        complete=lambda *args: None,
    )
    monkeypatch.setattr(replay_evaluation, "evaluate_manifest", collect)
    monkeypatch.setattr(replay_evaluation, "read_evaluation", lambda *args: None)
    monkeypatch.setattr(replay_evaluation, "_prompt", lambda record: "synthetic")
    for arm in ARMS:
        for purpose, stage, update in (
            ("candidate", "stage_a", 10),
            ("a_validation", "stage_b", 0),
        ):
            asyncio.run(
                replay_evaluation.evaluate(
                    remote,
                    source(11),
                    {"sampler_path": arm},
                    manifest("synthetic", 1),
                    stage=stage,
                    update=update,
                    arm=arm,
                    purpose=purpose,
                    draws=16,
                    cap=4096,
                    output=tmp_path / arm / purpose,
                )
            )
    assert calls[0]["seed_namespace"] == calls[2]["seed_namespace"]
    assert calls[1]["seed_namespace"] == calls[3]["seed_namespace"]
    assert calls[0]["seed_namespace"] != calls[1]["seed_namespace"]
    assert all(
        c["method"] is None and c.get("sample_index_start", 0) == 0 for c in calls
    )
    assert len({c["label"] for c in calls}) == 4
    # A cache hit still reaches the existing evidence validator, without a sampler.
    monkeypatch.setattr(
        replay_evaluation, "read_evaluation", lambda *args: {"complete": True}
    )
    remote.sampler = lambda *args: pytest.fail("cached evidence must not call remote")
    kwargs = dict(
        stage="stage_a",
        update=10,
        arm="R-S",
        purpose="candidate",
        draws=16,
        cap=4096,
        output=tmp_path / "R-S" / "candidate",
    )
    asyncio.run(
        replay_evaluation.evaluate(
            remote,
            source(11),
            {"sampler_path": "R-S"},
            manifest("synthetic", 1),
            **kwargs,
        )
    )
    assert len(calls) == 5 and calls[-1]["sampler"] is None
    kwargs["cap"] = 128
    with pytest.raises(ValueError, match="different arm/checkpoint"):
        asyncio.run(
            replay_evaluation.evaluate(
                remote,
                source(11),
                {"sampler_path": "R-S"},
                manifest("synthetic", 1),
                **kwargs,
            )
        )
    assert len(calls) == 5
