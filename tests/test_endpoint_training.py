"""Token-preserving clone loss and concurrently dispatched frozen RL updates."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from duraseed import endpoint_remote as module
from duraseed import endpoint_training as training
from duraseed.endpoint_training import raw_datums
from duraseed.runners.pilot0_updates import _group_seeds
from duraseed.runners.remote_journal import RemoteJournal
from duraseed.runtime import SampleObservation, TokenBudget, TokenLedger


class Input:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.length = len(tokens)

    def to_ints(self):
        return self.tokens

    def append(self, chunk):
        return Input(self.tokens + chunk.tokens)

    from_ints = classmethod(lambda cls, tokens: cls(tokens))


class Row:
    def __init__(self, **values):
        self.__dict__.update(values)

    def model_dump(self, **kwargs):
        return self.__dict__


def runtime():
    return NS(
        sdk=NS(
            tinker=NS(
                ModelInput=Input,
                Datum=lambda **values: NS(**values),
                EncodedTextChunk=lambda **values: NS(**values),
            )
        ),
        renderer=NS(build_generation_prompt=lambda *a, **kw: Input([11, 12, 13])),
    )


def test_raw_loss_preserves_capped_invalid_empty_and_eos_responses():
    rows = [
        {
            "prompt_token_ids": [10, 11],
            "generation": {
                "completion_token_ids": tokens,
                "stop_reason": reason,
                "reward": reward,
            },
        }
        for tokens, reason, reward in [
            ([20, 21, 22], "length", 0),
            ([99], "stop", 1),
            ([], "stop", 0),
        ]
    ]
    datums = raw_datums(runtime(), rows, max_length=4)
    assert [d.model_input.tokens for d in datums] == [[10, 11, 20, 21], [10, 11], [10]]
    assert [d.loss_fn_inputs["target_tokens"] for d in datums] == [
        [0, 20, 21, 22],
        [0, 99],
        [0],
    ]
    assert [d.loss_fn_inputs["weights"] for d in datums] == [
        [0, 0.75, 0.75, 0.75],
        [0, 0.75],
        [0],
    ]
    assert sum(datums[0].loss_fn_inputs["weights"]) == 3 * sum(
        datums[1].loss_fn_inputs["weights"]
    )
    with pytest.raises(ValueError, match="truncated"):
        raw_datums(runtime(), rows, max_length=3)
    with pytest.raises(ValueError, match="normalization"):
        raw_datums(runtime(), rows, max_length=4, mean_response_length=4)


def remote_fixture(tmp_path, monkeypatch):
    remote = module.EndpointRemote.__new__(module.EndpointRemote)
    remote.root, remote.run_id, remote.session_ids = (
        tmp_path / "worker",
        "test-endpoint",
        ["test-session"],
    )
    remote.ledger = TokenLedger(TokenBudget(1000000, 1000000, 1000000), 100)
    remote.journal = RemoteJournal(remote.root / "remote")
    remote.config = {
        "tces_max_tokens": 4096,
        "evaluation": {"temperature": 1, "top_p": 0.95},
        "rl_group_concurrency": 16,
        "corpus_group_concurrency": 4,
    }
    remote.runtime = runtime()
    source = NS(
        seed=11,
        prompt_pools=NS(
            artifact=NS(bg_group_order=()),
            a_rl_train_manifest=NS(manifest_id="training", split="a_rl_train"),
        ),
    )
    records = [
        NS(
            task_id=f"task-{i}",
            intended_family="family",
            item_index=i,
            to_task=lambda: "task",
        )
        for i in range(16)
    ]
    monkeypatch.setattr(module, "render_prompt", lambda task: "frozen prompt")
    monkeypatch.setattr(module, "scheduled_stage_a_records", lambda *args: records)
    return remote, source, records


def observations(index, rewards):
    return tuple(
        SampleObservation(
            Row(
                sample_id=f"g{index}-s{i}",
                completion_token_ids=[20 + i, 99],
                completion_logprobs=[-1.0, -2.0],
                reward=reward,
            ),
            Row(reward=reward),
            Input([11, 12, 13]),
            (20 + i, 99),
            (-1.0, -2.0),
        )
        for i, reward in enumerate(rewards)
    )


def test_rl_concurrency_preserves_seeds_group_order_and_advantages(
    tmp_path, monkeypatch
):
    remote, source, records = remote_fixture(tmp_path, monkeypatch)
    completed, seeds, concurrency = [], {}, []

    async def sampler():
        return "immutable sampler"

    remote.runtime.model = NS(save_weights_and_get_sampling_client_async=sampler)

    async def sample(
        runtime, sampler, task, coordinate, *, ledger, explicit_seeds, **kwargs
    ):
        index = task.item_index
        seeds[index] = explicit_seeds
        ledger.reserve_call(TokenBudget(24, 8 * 4096, 0))
        concurrency.append(len(seeds) - len(completed))
        await asyncio.sleep((16 - index) * 0.005)
        completed.append(index)
        ledger.settle_call(TokenBudget(24, 16, 0))
        return observations(
            index, ([1] * 8 if index == 0 else [0] * 8 if index == 1 else [0, 1] * 4)
        )

    updated = []

    async def update(runtime, datums, *, loss_fn, learning_rate, ledger):
        assert len(completed) == 16
        updated.append(datums)
        assert loss_fn == "importance_sampling" and learning_rate == 1e-5
        ledger.reserve_call(
            TokenBudget(0, 0, sum(d.model_input.length for d in datums))
        )
        ledger.settle_call(TokenBudget(0, 0, sum(d.model_input.length for d in datums)))
        return {"loss": 1.0}

    monkeypatch.setattr(module, "sample_seeded", sample)
    monkeypatch.setattr(module, "apply_update", update)
    result = asyncio.run(
        remote.rl_update(
            source,
            {},
            step=1,
            learning_rate=1e-5,
            boundary_sampler_path="M0",
            output=remote.root / "u1",
        )
    )
    assert max(concurrency) == 16
    assert completed != list(range(16))
    assert all(
        seeds[i] == _group_seeds(11, 1, i, records[i].task_id) for i in range(16)
    )
    assert len(updated) == 1 and len(updated[0]) == 14 * 8
    assert [d.loss_fn_inputs["advantages"] for d in updated[0]] == [
        [0.0, 0.0, a, a] for a in [-0.5, 0.5] * (14 * 4)
    ]
    assert [d.loss_fn_inputs["logprobs"] for d in updated[0]] == [
        [0.0, 0.0, -1.0, -2.0]
    ] * (14 * 8)
    assert result["metrics"]["mixed_group_count"] == 14
    raw = json.loads((remote.root / "u1/rollouts/group-0000/samples.json").read_text())
    assert raw["records"][0]["prompt_token_ids"] == [11, 12, 13]
    assert raw["records"][0]["generation"]["completion_token_ids"] == [20, 99]
    assert not remote.journal.pending and not remote.ledger.has_pending_call
    with pytest.raises(ValueError, match="never replay"):
        asyncio.run(
            remote.rl_update(
                source,
                {},
                step=1,
                learning_rate=1e-5,
                boundary_sampler_path="M0",
                output=remote.root / "u1",
            )
        )


def test_group_failure_drains_submitted_groups_and_does_not_dispatch_more(
    tmp_path, monkeypatch
):
    remote, source, records = remote_fixture(tmp_path, monkeypatch)
    started, finished = [], []

    async def sample(runtime, sampler, task, coordinate, *, ledger, **kwargs):
        started.append(task.item_index)
        ledger.reserve_call(TokenBudget(24, 8 * 4096, 0))
        await asyncio.sleep(0.001 if task.item_index == 0 else 0.005)
        if task.item_index == 0:
            ledger.abort_call()
            raise TimeoutError("ambiguous request")
        finished.append(task.item_index)
        ledger.settle_call(TokenBudget(24, 16, 0))
        return observations(task.item_index, [0, 1] * 4)

    monkeypatch.setattr(module, "sample_seeded", sample)
    with pytest.raises(TimeoutError, match="ambiguous"):
        asyncio.run(
            remote._groups(
                source,
                source.prompt_pools.a_rl_train_manifest,
                records,
                "sampler",
                "checkpoint",
                "M0",
                remote.root / "samples",
                step=30,
                purpose="corpus",
                concurrency=4,
            )
        )

    assert started == [0, 1, 2, 3] and sorted(finished) == [1, 2, 3]
    assert remote.journal.pending and not remote.ledger.has_pending_call
    assert (remote.root / "samples/group-0003/samples.json").exists()
    with pytest.raises(ValueError, match="never resubmit"):
        asyncio.run(
            remote._groups(
                source,
                source.prompt_pools.a_rl_train_manifest,
                records,
                "sampler",
                "checkpoint",
                "M0",
                remote.root / "samples",
                step=30,
                purpose="corpus",
                concurrency=4,
            )
        )


def test_student_caches_all_datums_once_and_keeps_live_optimizer(tmp_path, monkeypatch):
    rows = [
        {"prompt_token_ids": [10], "generation": {"completion_token_ids": [20]}}
    ] * 6400
    cached, restores, steps, saves = [], [], [], []

    def convert(runtime, rows, **kwargs):
        cached.append(len(rows))
        return list(range(len(rows)))

    async def restore(path, **kwargs):
        restores.append((path, kwargs["full_state"]))

    async def update(datums, lr, path, coordinate):
        assert lr == 1e-4
        steps.append((coordinate["update"], datums))

    async def save(name, path, coordinate):
        saves.append(coordinate["update"])
        return {**coordinate, "state_path": name, "sampler_path": name + "-sampler"}

    async def evaluate(*args, **kwargs):
        return {"item_counts": [{"task_id": "test", "successes": 1, "trials": 1}]}

    remote = NS(
        root=tmp_path,
        run_id="test",
        runtime=runtime(),
        config={"max_length": 4096, "student_updates": 294},
        restore=restore,
        update=update,
        save=save,
    )
    monkeypatch.setattr(training, "raw_datums", convert)
    monkeypatch.setattr(training, "evaluate", evaluate)
    result = asyncio.run(
        training.train_student(
            remote,
            NS(seed=11, a_cadence="test"),
            {"state_path": "M0"},
            rows,
            reversed(range(6400)),
        )
    )
    assert restores == [("M0", False)] and cached == [6400]
    assert len(steps) == 294 and steps[0][1] == list(range(6399, 6367, -1))
    assert steps[200][1] == steps[0][1]
    assert saves == list(range(10, 291, 10)) + [294]
    assert result["checkpoint"]["update"] == 294
    assert len(result["cadence"]) == 30 and "item_counts" in result["cadence"][0]


def test_group_error_drains_other_seven_submitted_requests(tmp_path, monkeypatch):
    remote, source, records = remote_fixture(tmp_path, monkeypatch)
    finished = []

    async def request(index):
        await asyncio.sleep(0.001 if index == 0 else 0.02)
        if index == 0:
            raise TimeoutError("single sampled request failed")
        finished.append(index)

    async def sample(runtime, sampler, task, coordinate, *, ledger, **kwargs):
        ledger.reserve_call(TokenBudget(24, 8 * 4096, 0))
        try:
            await asyncio.gather(*(sampler.sample_async(index=i) for i in range(8)))
        except Exception:
            ledger.abort_call()
            raise

    monkeypatch.setattr(module, "sample_seeded", sample)
    with pytest.raises(TimeoutError, match="single sampled"):
        asyncio.run(
            remote._groups(
                source,
                source.prompt_pools.a_rl_train_manifest,
                records[:1],
                NS(sample_async=request),
                "checkpoint",
                "M0",
                remote.root / "samples",
                step=30,
                purpose="corpus",
                concurrency=1,
            )
        )
    assert sorted(finished) == list(range(1, 8)) and remote.journal.pending
