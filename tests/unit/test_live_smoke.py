from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.config import load_pilot_config
from duraseed.live_smoke_gate import SmokeRun, SmokeSettings, authorize
from duraseed.runners import live_smoke as smoke_runner
from duraseed.runners.live_smoke import execute_smoke
from duraseed.runners.live_smoke_data import build_inputs
from duraseed.runtime import RuntimeBundle, SDKBundle, TokenBudget, TokenLedger
from duraseed.tasks.maps import canonical_program, shortest_program
from duraseed.tasks.tces import enumerate_task


class Input:
    def __init__(self, tokens: list[int], text: str = "") -> None:
        self.tokens, self.text, self.length = tokens, text, len(tokens)

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    def append(self, chunk) -> "Input":  # type: ignore[no-untyped-def]
        return Input(self.tokens + list(chunk.tokens), self.text)


class Datum:
    def __init__(self, model_input, loss_fn_inputs) -> None:  # type: ignore[no-untyped-def]
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


class Tinker:
    class SamplingParams:
        def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
            self.values = values

    class EncodedTextChunk:
        def __init__(self, *, tokens) -> None:  # type: ignore[no-untyped-def]
            self.tokens = tokens

    Datum = Datum

    class AdamParams:
        def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
            self.values = values


class Tokenizer:
    eos_token_id = None

    def decode(self, tokens) -> str:  # type: ignore[no-untyped-def]
        return "".join(chr(value) for value in tokens)


class Renderer:
    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        assert role == "assistant"
        return Input([10, 11], messages[0]["content"])

    def get_stop_sequences(self) -> list[str]:
        return ["\n\nUser:"]

    def parse_response(self, tokens):  # type: ignore[no-untyped-def]
        text = "".join(chr(value) for value in tokens)
        if text.startswith("EOS"):
            return None, "eos"
        return None, "malformed" if text.startswith("MALFORMED") else "stop_sequence"

    def build_supervised_example(self, messages, *, train_on_what):  # type: ignore[no-untyped-def]
        return Input([1, 2, 3, 4, 5, 6]), [0, 0, 0, 1, 1, 1]


class Future:
    def __init__(self, value) -> None:  # type: ignore[no-untyped-def]
        self.value = value

    async def result_async(self):  # type: ignore[no-untyped-def]
        return self.value


class Model:
    def __init__(self, service: "Service", model_id: str) -> None:
        self.service, self.model_id = service, model_id

    def get_tokenizer(self) -> Tokenizer:
        return Tokenizer()

    async def get_info_async(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(model_id=self.model_id)

    async def forward_backward_async(self, datums, loss_fn):  # type: ignore[no-untyped-def]
        self.service.calls.append(("forward", loss_fn, len(datums)))
        return Future(SimpleNamespace(metrics={"loss": 0.5}))

    async def optim_step_async(self, params):  # type: ignore[no-untyped-def]
        self.service.calls.append(("optimizer", self.model_id))
        return Future(SimpleNamespace(metrics={"lr": params.values["learning_rate"]}))

    async def save_weights_for_sampler_async(self, name, **kwargs):  # type: ignore[no-untyped-def]
        path = f"tinker://fake/{self.model_id}/sampler/{name}"
        self.service.calls.append(("save_sampler", path))
        return Future(SimpleNamespace(path=path))

    async def save_state_async(self, name, **kwargs):  # type: ignore[no-untyped-def]
        path = f"tinker://fake/{self.model_id}/state/{name}"
        self.service.calls.append(("save_state", path))
        return Future(SimpleNamespace(path=path))


class Sampler:
    def __init__(self, service: "Service", source: str) -> None:
        self.service, self.source = service, source

    async def sample_async(self, *, prompt, num_samples, sampling_params):  # type: ignore[no-untyped-def]
        assert num_samples == 1 and sampling_params.values["max_tokens"] == 4096
        is_maps = "Allowed instructions:" in prompt.text
        if is_maps:
            answer = self.service.maps_answers[prompt.text]
        else:
            mixed = "after-sft" in self.source and sampling_params.values["seed"] % 2
            answer = (
                "<answer>bad</answer>"
                if mixed
                else self.service.tces_answers[prompt.text]
            )
        self.service.sample_calls += 1
        stop_reason = self.service.stop_reason
        if stop_reason is None and self.source == "Qwen/Qwen3.5-9B-Base":
            remainder = self.service.sample_calls % 4
            if remainder == 1:
                stop_reason = "length"
            elif remainder == 2:
                answer, stop_reason = "EOS without a wrapper", "stop"
            elif remainder == 3:
                answer, stop_reason = "MALFORMED without a wrapper", "stop"
            else:
                stop_reason = "stop"
        tokens = [ord(value) for value in answer]
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=tokens,
                    logprobs=[-0.1] * len(tokens),
                    stop_reason=stop_reason or "stop",
                )
            ]
        )


class Service:
    def __init__(
        self, tces_answers: dict[str, str], maps_answers: dict[str, str]
    ) -> None:
        self.tces_answers, self.maps_answers = tces_answers, maps_answers
        self.calls: list[tuple[object, ...]] = []
        self.next_model = 0
        self.stop_reason: str | None = None
        self.sample_calls = 0

    async def create_sampling_client_async(self, **kwargs):  # type: ignore[no-untyped-def]
        source = kwargs.get("model_path", kwargs.get("base_model"))
        self.calls.append(("sampler", source))
        return Sampler(self, source)

    async def create_training_client_from_state_with_optimizer_async(
        self, path, **kwargs
    ):  # type: ignore[no-untyped-def]
        self.next_model += 1
        return Model(self, f"full-{self.next_model}")

    async def create_training_client_from_state_async(self, path, **kwargs):  # type: ignore[no-untyped-def]
        self.next_model += 1
        return Model(self, f"branch-{self.next_model}")


def _runtime(inputs, ledger: TokenLedger) -> tuple[RuntimeBundle, Service]:  # type: ignore[no-untyped-def]
    service = Service(
        {
            task.prompt_text: (
                "<answer>"
                + enumerate_task(task.exact_task).expressions[0].canonical_expression
                + "</answer>"
            )
            for task in inputs.rl_tasks
        },
        {
            inputs.maps_task.prompt_text: (
                "<answer>"
                + canonical_program(shortest_program(inputs.maps_task.exact_task) or ())
                + "</answer>"
            )
        },
    )

    def datum(messages, renderer, **kwargs):  # type: ignore[no-untyped-def]
        return Datum(Input([1, 2, 3, 4, 5]), {"weights": [0, 0, 1, 1, 1]})

    sdk = SDKBundle(
        Tinker,
        lambda *args, **kwargs: Renderer(),
        SimpleNamespace(LAST_ASSISTANT_MESSAGE="last"),
        datum,
        "0.25.0",
        "0.5.3",
    )
    return RuntimeBundle(
        sdk, service, Model(service, "initial"), Renderer(), Tokenizer()
    ), service


def test_behavioral_fake_uses_live_orchestration_without_real_evidence(
    tmp_path: Path,
) -> None:
    inputs = build_inputs(5)
    ledger = TokenLedger(TokenBudget(200_000, 200_000, 100_000), 25.0)
    runtime, service = _runtime(inputs, ledger)
    settings = SmokeSettings("mock-smoke", output_root=tmp_path / "mock-output")
    result = asyncio.run(
        execute_smoke(
            runtime,
            ledger,
            settings,
            load_pilot_config("duraseed_pilot_config.yaml"),
            inputs,
            evidence_origin="mock",
            project_id="mock-project",
        )
    )
    acceptance = json.loads((result / "acceptance.json").read_text())
    assert acceptance["status"] == "mock_passed"
    assert acceptance["real_data"] is False
    assert acceptance["online_offline_reward_parity"] is True
    assert acceptance["stop_contract_verified"] is True
    assert acceptance["max_tokens"]["protocol_value"] == 4096
    assert acceptance["max_tokens"]["selection_performed"] is False
    assert acceptance["max_tokens"]["truncated_count"] == 4
    assert acceptance["stop_contract"]["classification_counts"] == {
        "configured_stop": 4,
        "eos": 4,
        "length": 4,
        "renderer_malformed_stop": 4,
    }
    assert acceptance["stop_contract"]["wrapper_closed_count"] == 8
    assert acceptance["full_state_resume"] is True
    assert acceptance["weights_only_branch"] is True
    assert acceptance["checkpoint_lineage"]["stage_b_state_path"] is None
    operations = [
        json.loads(line)
        for line in (result / "operations.jsonl").read_text().splitlines()
    ]
    assert "restore:resumed-full-state-roundtrip" in {
        row["operation"] for row in operations
    }
    run_record = json.loads((result / "run.json").read_text())
    assert run_record["final_state_checkpoint_path"] is None
    assert any(
        call[0] == "forward" and call[1] == "importance_sampling"
        for call in service.calls
    )


def test_unclassified_transport_stop_fails_acceptance(tmp_path: Path) -> None:
    inputs = build_inputs(5)
    ledger = TokenLedger(TokenBudget(200_000, 200_000, 100_000), 25.0)
    runtime, service = _runtime(inputs, ledger)
    service.stop_reason = "unknown"
    with pytest.raises(RuntimeError, match="acceptance criteria"):
        asyncio.run(
            execute_smoke(
                runtime,
                ledger,
                SmokeSettings("bad-stop", output_root=tmp_path / "mock-output"),
                load_pilot_config("duraseed_pilot_config.yaml"),
                inputs,
                evidence_origin="mock",
                project_id="mock-project",
            )
        )


def test_reward_parity_checks_generation_reward_join(tmp_path: Path) -> None:
    from duraseed.live_smoke_sampling import SmokeSampler

    inputs = build_inputs(5)
    ledger = TokenLedger(TokenBudget(20_000, 20_000, 100), 25.0)
    runtime, service = _runtime(inputs, ledger)
    settings = SmokeSettings("tamper", output_root=tmp_path / "mock")
    run = SmokeRun.start(
        settings,
        load_pilot_config("duraseed_pilot_config.yaml"),
        inputs.manifests,
        evidence_origin="mock",
        project_id="mock-project",
    )
    samples = SmokeSampler(settings.run_id, settings.seed, run.directory, ledger, run)
    sampler = asyncio.run(service.create_sampling_client_async(base_model="base"))
    rows = asyncio.run(
        samples.collect(
            runtime,
            sampler,
            inputs.tces_task,
            label="probe-tamper",
            stage="base",
            path="base",
            count=1,
            max_tokens=4096,
        )
    )
    row = rows[0]
    samples.observations[0] = row.__class__(
        row.generation.model_copy(update={"reward": 0.5}),
        row.reward,
        row.prompt,
        row.tokens,
        row.logprobs,
    )
    acceptance = samples.acceptance(
        runtime, (*inputs.rl_tasks, inputs.maps_task), probe_rows=rows
    )
    assert acceptance["online_offline_reward_parity"] is False


def test_run_remote_dispatches_same_behavioral_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    inputs = build_inputs(5)
    fake_ledger = TokenLedger(TokenBudget(200_000, 200_000, 100_000), 25.0)
    runtime, service = _runtime(inputs, fake_ledger)

    async def resolve(*args, **kwargs):  # type: ignore[no-untyped-def]
        runtime.model.service = service
        return runtime

    monkeypatch.setattr(smoke_runner, "load_sdk", lambda: runtime.sdk)
    monkeypatch.setattr(smoke_runner, "create_service", lambda *a, **k: service)
    monkeypatch.setattr(smoke_runner, "resolve_model", resolve)
    result = asyncio.run(
        smoke_runner.run_remote(
            SmokeSettings("remote-dispatch", output_root=tmp_path / "remote-test"),
            authorize(
                execute=True,
                authorized_cost_usd="25",
                billing_reconciled=True,
                human_approval=True,
            ),
            project_id="project-1",
        )
    )
    acceptance = json.loads((result / "acceptance.json").read_text())
    assert acceptance["status"] == "passed"
    assert acceptance["real_data"] is True
    assert not (result / "pending_operation.json").exists()


def test_failed_paid_operation_leaves_unresolved_marker(tmp_path: Path) -> None:
    inputs = build_inputs(5)
    settings = SmokeSettings("interrupted", output_root=tmp_path / "mock")
    run = SmokeRun.start(
        settings,
        load_pilot_config("duraseed_pilot_config.yaml"),
        inputs.manifests,
        evidence_origin="mock",
        project_id="mock-project",
    )
    ledger = TokenLedger(TokenBudget(0, 0, 0), 25.0)

    async def fail() -> object:
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(
            run.paid(
                "remote:test",
                ledger,
                fail,
                reservation={"tokens": 0},
                persist=lambda _: {},
            )
        )
    assert (run.directory / "pending_operation.json").exists()
