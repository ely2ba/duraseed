"""Offline integration of acquisition, immutable selection, and launch ordering."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from duraseed.replay_inputs import read_json


def module():
    path = Path(__file__).parents[1] / "tools/run_endpoint_clone.py"
    spec = importlib.util.spec_from_file_location("endpoint_runner", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize(
    "selection_status,confirmation_status",
    [("NO_MATCH", None), ("SELECTED", "CONFIRMATION_FAILED"), ("SELECTED", "MATCHED")],
)
def test_selection_confirmation_then_four_fixed_continuations(
    tmp_path, monkeypatch, selection_status, confirmation_status
):
    run, events = module(), []
    monkeypatch.setattr(
        run.pilot0_concurrent_sampling,
        "EVALUATION_CONCURRENCY",
        run.pilot0_concurrent_sampling.EVALUATION_CONCURRENCY,
    )
    monkeypatch.setattr(
        run.pilot0_sampling,
        "sample_manifest_groups",
        run.pilot0_sampling.sample_manifest_groups,
    )
    config = {
        "project_id": "fake",
        "m0_state_path": "m0-state",
        "m0_sampler_path": "m0-sampler",
    }
    source = NS(a_validation=object(), seed=11)
    monkeypatch.setattr(
        run, "load_inputs", lambda *a: (config, source, "clone", "confirmation")
    )
    monkeypatch.setattr(run, "corpus_coordinates", lambda m: (("a", 1), ("a", 0)))

    def remote(repo, root, cfg, worker):
        events.append(("worker", worker))

        async def corpus(*a):
            return tuple(
                {"generation": {"task_id": "a", "sample_index": i}} for i in (0, 1)
            )

        return NS(root=root / worker, sample_corpus=corpus, snapshot=lambda: {})

    monkeypatch.setattr(run, "worker_remote", remote)

    async def teacher(remote, source, m0):
        assert m0 == {"state_path": "m0-state", "sampler_path": "m0-sampler"}
        return {"checkpoint": {"update": 30}, "cadence": [{}]}

    async def student(remote, source, m0, rows, order):
        assert order == [1, 0]
        return {"checkpoints": [{"update": u} for u in (10, 20, 30)], "cadence": []}

    async def profile(remote, source, checkpoint, manifest, name):
        events.append(("profile", name))
        return {}

    selection = {
        "status": selection_status,
        "selected_update": 20 if selection_status == "SELECTED" else None,
    }
    matching = {"status": confirmation_status, "selected_update": 20}
    monkeypatch.setattr(run, "train_teacher", teacher)
    monkeypatch.setattr(run, "train_student", student)
    monkeypatch.setattr(run, "profile", profile)
    monkeypatch.setattr(
        run, "nominate", lambda *a: {"nominees": [{"update": u} for u in (10, 20, 30)]}
    )
    monkeypatch.setattr(run, "select", lambda *a: selection)
    monkeypatch.setattr(run, "confirm", lambda *a: matching)

    async def continuation(remote, source, origin, label, stop):
        assert read_json(tmp_path / "matching.json")["status"] == "MATCHED"
        assert origin["update"] == (30 if label.startswith("T") else 20)
        assert stop == (480 if label.endswith("1") else 20)
        events.append(("continue", label))
        return {"label": label}

    monkeypatch.setattr(run, "run_continuation", continuation)
    asyncio.run(run.execute(tmp_path, tmp_path))
    result = read_json(tmp_path / "result.json")
    assert result["status"] == (
        "COMPLETED" if confirmation_status == "MATCHED" else "CLONE_UNAVAILABLE"
    )
    assert result["stage_b_started"] == (confirmation_status == "MATCHED")
    continuations = [e[1] for e in events if e[0] == "continue"]
    assert continuations == (
        ["T1", "T2", "S1", "S2"] if confirmation_status == "MATCHED" else []
    )
    profiles = [e[1] for e in events if e[0] == "profile"]
    assert len(profiles) == (4 if selection_status == "NO_MATCH" else 7)


def test_never_relaunch_ambiguous_root(tmp_path):
    run = module()
    (tmp_path / "remote-call-state.json").write_text("{}")
    with pytest.raises(RuntimeError, match="already dispatched"):
        asyncio.run(run.execute(tmp_path, tmp_path))
