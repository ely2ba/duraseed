"""Mock-service replay execution tests; no paid service is constructed."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from duraseed import replay_inputs, replay_remote
from duraseed.runners import RunnerGateError
from duraseed.runtime import SDKBundle


class Future:
    def __init__(self, value):
        self.value = value

    async def result_async(self):
        return self.value


class Model:
    def __init__(self, optimizer_step):
        self.optimizer_step = optimizer_step
        self.calls = []

    def get_tokenizer(self):
        return "synthetic-tokenizer"

    async def forward_backward_async(self, datums, loss_fn):
        self.calls.append(("forward", len(datums), loss_fn))
        return Future(SimpleNamespace(metrics={"loss:sum": 1.0}))

    async def optim_step_async(self, params):
        self.optimizer_step += 1
        self.calls.append(("optimizer", params))
        return Future(SimpleNamespace(metrics={"step": self.optimizer_step}))


class Service:
    def __init__(self):
        self.calls = []

    def _get_session_holder(self):
        return SimpleNamespace(get_session_id=lambda: "synthetic-session")

    async def create_training_client_from_state_async(self, path, **kwargs):
        self.calls.append(("weights_only", path))
        return Model(0)

    async def create_training_client_from_state_with_optimizer_async(
        self, path, **kwargs
    ):
        self.calls.append(("full_state", path))
        return Model(10)


def remote_fixture(tmp_path, monkeypatch):
    repo, root = tmp_path, tmp_path / "synthetic-replay-run"
    tokens = {"prefill": 100000, "sample": 100000, "train": 100000}
    (repo / "preflight.json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "token_budget": tokens,
                "main_cost_usd": 10.0,
                "main_usd": 10.0,
            }
        )
    )
    config = {
        "preflight": {"path": "preflight.json"},
        "evaluation": {"temperature": 1.0, "top_p": 0.95},
    }
    service = Service()
    sdk = SDKBundle(
        SimpleNamespace(AdamParams=lambda **values: values),
        lambda *args, **kwargs: "synthetic-renderer",
        None,
        None,
        "synthetic",
        "synthetic",
    )
    monkeypatch.setattr(replay_remote, "load_sdk", lambda: sdk)
    monkeypatch.setattr(
        replay_remote, "create_service", lambda *args, **kwargs: service
    )
    return repo, root, config, service


def test_weights_only_stage_entry_then_preserved_optimizer_resume(
    tmp_path, monkeypatch
):
    repo, root, config, service = remote_fixture(tmp_path, monkeypatch)
    remote = replay_remote.ReplayRemote(repo, config, root, "synthetic-project")
    asyncio.run(
        remote.restore(
            "synthetic-M0", full_state=False, coordinate={"arm": "R-P", "stage": "A"}
        )
    )
    assert remote.runtime.model.optimizer_step == 0
    asyncio.run(
        remote.restore(
            "synthetic-step10-state",
            full_state=True,
            coordinate={"arm": "R-P", "stage": "A"},
        )
    )
    assert remote.runtime.model.optimizer_step == 10
    asyncio.run(
        remote.restore(
            "synthetic-selected-state",
            full_state=False,
            coordinate={"arm": "R-P", "stage": "B"},
        )
    )
    assert remote.runtime.model.optimizer_step == 0
    assert service.calls == [
        ("weights_only", "synthetic-M0"),
        ("full_state", "synthetic-step10-state"),
        ("weights_only", "synthetic-selected-state"),
    ]


def test_update_pins_optimizer_and_does_not_double_execute(tmp_path, monkeypatch):
    repo, root, config, _ = remote_fixture(tmp_path, monkeypatch)
    remote = replay_remote.ReplayRemote(repo, config, root, "synthetic-project")
    asyncio.run(
        remote.restore("synthetic-M0", full_state=False, coordinate={"arm": "R-S"})
    )
    datum = SimpleNamespace(model_input=SimpleNamespace(length=4500))
    path = root / "update-1.json"
    asyncio.run(remote.update([datum], 1e-4, path, {"arm": "R-S", "update": 1}))
    assert remote.runtime.model.optimizer_step == 1
    params = remote.runtime.model.calls[-1][1]
    assert params == {
        "learning_rate": 1e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-12,
        "weight_decay": 0.0,
        "grad_clip_norm": 0.0,
    }
    assert remote.ledger.observed.train == 4500
    with pytest.raises(ValueError, match="never replay"):
        asyncio.run(remote.update([datum], 1e-4, path, {"arm": "R-S", "update": 1}))
    assert remote.runtime.model.optimizer_step == 1


def test_pending_remote_state_is_not_retried_after_exception(tmp_path, monkeypatch):
    repo, root, config, service = remote_fixture(tmp_path, monkeypatch)
    remote = replay_remote.ReplayRemote(repo, config, root, "synthetic-project")
    asyncio.run(
        remote.restore("synthetic-M0", full_state=False, coordinate={"arm": "R-P"})
    )
    calls = []

    async def ambiguous(*args, **kwargs):
        calls.append("submitted")
        raise TimeoutError("synthetic ambiguous in-flight response")

    monkeypatch.setattr(replay_remote, "apply_update", ambiguous)
    datum = SimpleNamespace(model_input=SimpleNamespace(length=5000))
    with pytest.raises(TimeoutError):
        asyncio.run(
            remote.update(
                [datum], 1e-4, root / "update.json", {"arm": "R-P", "update": 1}
            )
        )
    with pytest.raises(RunnerGateError, match="ambiguous pending"):
        replay_remote.ReplayRemote(repo, config, root, "synthetic-project")
    assert calls == ["submitted"] and len(service.calls) == 1
    state = json.loads((root / "remote" / "remote-call-state.json").read_text())
    assert state["pending"]["reservation"]["train_tokens"] == 5000


def test_candidate_population_uses_family_ids_not_panel_label(monkeypatch):
    records = [
        SimpleNamespace(intended_family="target", task_id=f"synthetic-{i}")
        for i in range(256)
    ] + [
        SimpleNamespace(intended_family="sentinel", task_id=f"synthetic-s-{i}")
        for i in range(256)
    ]
    original = SimpleNamespace(
        records=records,
        split="a_validation",
        generator_version="1.0.0",
        root_seed=11,
        manifest_id="original-manifest",
    )
    source = SimpleNamespace(
        a_validation=original,
        prompt_pools=SimpleNamespace(
            artifact=SimpleNamespace(
                targeted_panel="A", boundary_family_ids=("target",)
            )
        ),
    )
    monkeypatch.setattr(replay_inputs, "build_manifest", lambda **values: values)
    result = replay_inputs.candidate_manifest(source)
    assert len(result["records"]) == 256 and result["records"] == records[:256]
    assert result["parent_manifest_id"] == original.manifest_id
