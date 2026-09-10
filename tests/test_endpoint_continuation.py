"""Synthetic execution checks: optimizer continuity, schedule, independent RNG."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from duraseed import endpoint_continuation as run, replay_evaluation
from duraseed.replay_remote import write_json


class Remote:
    def __init__(self, root):
        self.root, self.run_id = root, "synthetic-endpoint"
        self.config = {"max_length": 5000, "stage_b_lr": 3e-4}
        self.runtime, self.events = NS(renderer=None), []

    async def restore(self, path, **kwargs):
        self.events.append(("restore", path, kwargs))
        self.runtime.renderer = object()

    async def update(self, datums, lr, path, coordinate):
        self.events.append(
            ("update", coordinate["update"], lr, [d.index for d in datums])
        )
        write_json(path, coordinate)

    async def save(self, name, path, coordinate):
        self.events.append(("save", coordinate["update"]))
        value = {**coordinate, "state_path": name + "-state", "sampler_path": name}
        write_json(path, value)
        return value

    def snapshot(self):
        return {"synthetic": True}


def setup(tmp_path, monkeypatch):
    def panel(size):
        return NS(
            record_count=size,
            records=["targeted"] * (size // 2) + ["sentinel"] * (size // 2),
        )

    source = NS(
        seed=11,
        prompt_pools=NS(a_monitor_manifest=panel(384)),
        a_validation=panel(512),
        b_validation=panel(512),
    )
    remote = Remote(tmp_path)
    built = []
    monkeypatch.setattr(run, "stage_b_sources", lambda source: tuple(range(4096)))
    monkeypatch.setattr(run, "_panel_role", lambda source, row: row)

    def datum(runtime, row, max_length):
        built.append(row)
        return NS(index=row, model_input=NS(length=10))

    monkeypatch.setattr(run, "sft_datum", datum)

    async def evaluate(remote, source, checkpoint, manifest, **kwargs):
        remote.events.append(("evaluate", checkpoint.copy(), kwargs))
        return {"row_count": manifest.record_count * kwargs["draws"]}

    monkeypatch.setattr(run, "evaluate", evaluate)
    return remote, source, built


@pytest.mark.parametrize("label,stop", list(run.RUN_STOPS.items()))
def test_resident_optimizer_fixed_data_order_and_sampling(
    tmp_path, monkeypatch, label, stop
):
    remote, source, built = setup(tmp_path, monkeypatch)
    origin = {
        "state_path": "selected-state",
        "sampler_path": "selected-sampler",
        "update": 30,
    }
    result = asyncio.run(run.run_continuation(remote, source, origin, label, stop))
    restores = [e for e in remote.events if e[0] == "restore"]
    assert len(restores) == 1 and restores[0][1] == "selected-state"
    assert restores[0][2]["full_state"] is False
    assert len(restores[0][2]["sources"]) == 4096
    assert built == list(range(4096))  # Tokenization once, not every segment.
    updates = [e for e in remote.events if e[0] == "update"]
    assert [e[1] for e in updates] == list(range(1, stop + 1))
    for _, update, lr, batch in updates:
        assert lr == 3e-4
        assert batch == [((update - 1) * 32 + j) % 4096 for j in range(32)]
    expected = run.grids(stop)
    assert [e[1] for e in remote.events if e[0] == "save"] == list(
        expected["a_monitor"][1:]
    )
    evaluations = [e for e in remote.events if e[0] == "evaluate"]
    for purpose, grid in expected.items():
        actual = [e for e in evaluations if e[2]["purpose"] == purpose]
        assert [e[2]["update"] for e in actual] == list(grid)
        for _, checkpoint, options in actual:
            assert checkpoint["update"] == options["update"]
            assert checkpoint["origin_sampler_path"] == "selected-sampler"
            assert (
                options["seed_namespace"]
                == f"endpoint_clone.{label}.{purpose}.{options['update']}"
            )
            assert (options["draws"], options["cap"]) == run.PANELS[purpose][1:]
    assert sum(e[2]["update"] == 0 for e in evaluations) == 2
    assert result["updates"] == stop and result["training_examples"] == stop * 32
    assert result["training_tokens"] == stop * 32 * 10
    assert origin["update"] == 30  # Never mutate acquisition identity.
    before = len(remote.events)
    assert (
        asyncio.run(run.run_continuation(remote, source, origin, label, stop)) == result
    )
    assert len(remote.events) == before


def test_partial_run_cannot_be_silently_replayed(tmp_path, monkeypatch):
    remote, source, _ = setup(tmp_path, monkeypatch)
    write_json(tmp_path / "stage_b/u1/update-1.json", {"update": 1})
    with pytest.raises(ValueError, match="reconciliation"):
        asyncio.run(
            run.run_continuation(
                remote, source, {"state_path": "s", "sampler_path": "p"}, "T2", 20
            )
        )
    assert not remote.events


def test_label_controls_predetermined_duration(tmp_path, monkeypatch):
    remote, source, _ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="fixed matrix"):
        asyncio.run(run.run_continuation(remote, source, {}, "T2", 480))
    assert not remote.events


def test_namespace_reaches_sampler_and_durable_identity(tmp_path, monkeypatch):
    remote = NS(root=tmp_path, eval_inputs=object(), events=[])
    source, manifest = (
        NS(seed=11),
        NS(manifest_id="synthetic", records=(), record_count=0),
    )
    remote.runtime = NS(renderer=object())

    async def sampler(path, coordinate):
        remote.events.append(coordinate)
        return object()

    remote.sampler, remote.begin, remote.complete = (
        sampler,
        lambda *a: None,
        lambda *a: None,
    )

    async def manifest_eval(*args, **kwargs):
        remote.events.append(kwargs)
        return {"row_count": 0}

    monkeypatch.setattr(replay_evaluation, "evaluate_manifest", manifest_eval)
    for label in ("T1", "T2", "S1", "S2"):
        namespace = f"endpoint_clone.{label}.a_monitor.0"
        asyncio.run(
            replay_evaluation.evaluate(
                remote,
                source,
                {"sampler_path": "frozen"},
                manifest,
                stage="stage_b",
                update=0,
                arm=label,
                purpose="a_monitor",
                draws=4,
                cap=4096,
                output=tmp_path / label,
                seed_namespace=namespace,
            )
        )
    assert {r["seed_namespace"] for r in remote.events} == {
        f"endpoint_clone.{label}.a_monitor.0" for label in ("T1", "T2", "S1", "S2")
    }
