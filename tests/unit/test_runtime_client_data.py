from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from duraseed.runtime import (
    MODEL_ID,
    LORA_RANK,
    RuntimeBundle,
    SDKBundle,
    TokenBudget,
    TokenLedger,
    apply_update,
    create_sampler,
    create_service,
    resolve_model,
    rl_datums,
    sft_datum,
    teacher_token_measurer,
)
from duraseed.training.sft import SourceKind, VerifiedSourceRecord


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def test_runtime_import_does_not_import_optional_tinker_packages() -> None:
    assert "tinker" not in sys.modules
    assert "tinker_cookbook" not in sys.modules


def test_service_and_sampling_client_construction_use_exact_pinned_kwargs() -> None:
    class ServiceConstructor:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

    sdk = _sdk()
    sdk = SDKBundle(
        SimpleNamespace(ServiceClient=ServiceConstructor),
        sdk.get_renderer,
        sdk.train_on_what,
        sdk.conversation_to_datum,
        sdk.sdk_version,
        sdk.cookbook_version,
    )
    service = create_service(sdk, project_id="project-1", user_metadata={"run": "test"})
    assert service.kwargs == {
        "project_id": "project-1",
        "user_metadata": {"run": "test"},
    }

    class SamplingService:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        async def create_sampling_client_async(self, **kwargs):  # type: ignore[no-untyped-def]
            assert ledger.has_pending_call
            self.calls.append(kwargs)
            return f"sampler-{len(self.calls)}"

    sampling_service = SamplingService()
    runtime = RuntimeBundle(sdk, sampling_service, object(), Renderer(), object())
    ledger = TokenLedger(TokenBudget(0, 0, 0), 0.0)
    assert asyncio.run(create_sampler(runtime, ledger=ledger)) == "sampler-1"
    assert (
        asyncio.run(
            create_sampler(
                runtime,
                ledger=ledger,
                checkpoint_path="tinker://run/weights/1",
            )
        )
        == "sampler-2"
    )
    assert sampling_service.calls == [
        {"base_model": MODEL_ID},
        {"model_path": "tinker://run/weights/1"},
    ]


class Input:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.length = len(tokens)

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    def append(self, chunk) -> "Input":  # type: ignore[no-untyped-def]
        return Input(self.tokens + list(chunk.tokens))


class Datum:
    def __init__(self, model_input, loss_fn_inputs) -> None:  # type: ignore[no-untyped-def]
        self.model_input = model_input
        self.loss_fn_inputs = {
            key: SimpleNamespace(data=value) for key, value in loss_fn_inputs.items()
        }


class Renderer:
    def build_supervised_example(self, messages, *, train_on_what):  # type: ignore[no-untyped-def]
        assert messages[0]["content"] == "prompt"
        assert train_on_what == "last"
        return Input([10, 11, 12, 20, 21, 99]), [0, 0, 0, 1, 1, 1]

    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": "prompt"}]
        assert role == "assistant"
        return Input([10, 11, 12])


class Tinker:
    class EncodedTextChunk:
        def __init__(self, *, tokens) -> None:  # type: ignore[no-untyped-def]
            self.tokens = tokens

    class Datum:
        def __init__(self, *, model_input, loss_fn_inputs) -> None:  # type: ignore[no-untyped-def]
            self.model_input = model_input
            self.loss_fn_inputs = loss_fn_inputs

    class AdamParams:
        def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
            self.values = values


def _source() -> VerifiedSourceRecord:
    return VerifiedSourceRecord(
        prompt_text="prompt",
        verified_completion_text="answer",
        task_id=SHA_A,
        task_family="format",
        source_split="format_train",
        source_kind=SourceKind.TASK_AGNOSTIC_FORMAT,
        strategy_family_id=None,
        exact_verification=None,
        source_manifest_id=SHA_B,
    )


def _sdk() -> SDKBundle:
    def conversion(messages, renderer, **kwargs):  # type: ignore[no-untyped-def]
        assert messages[1]["content"] == "answer"
        assert isinstance(renderer, Renderer)
        assert kwargs == {
            "max_length": 1024,
            "train_on_what": "last",
            "reduction": "mean",
        }
        return Datum(Input([10, 11, 12, 20, 21]), {"weights": [0, 0, 1, 1, 1]})

    return SDKBundle(
        tinker=Tinker,
        get_renderer=None,
        train_on_what=SimpleNamespace(LAST_ASSISTANT_MESSAGE="last"),
        conversation_to_datum=conversion,
        sdk_version="0.25.0",
        cookbook_version="0.5.3",
    )


def test_sft_and_teacher_measurement_preserve_role_colon_contract() -> None:
    runtime = RuntimeBundle(_sdk(), object(), object(), Renderer(), object())
    datum = sft_datum(runtime, _source())
    counts = teacher_token_measurer(runtime)(_source())

    assert datum.model_input.to_ints() == [10, 11, 12, 20, 21]
    assert (counts.prompt, counts.target) == (3, 3)


def test_rl_datums_preserve_prompt_offset_tokens_logprobs_and_advantage() -> None:
    runtime = RuntimeBundle(_sdk(), object(), object(), Renderer(), object())
    row = SimpleNamespace(
        prompt=Input([1, 2, 3]),
        tokens=(5, 6, 7),
        logprobs=(-0.5, -0.4, -0.3),
    )
    datum = rl_datums(runtime, [row], [0.75])[0]

    assert datum.model_input.to_ints() == [1, 2, 3, 5, 6]
    assert datum.loss_fn_inputs == {
        "target_tokens": [0, 0, 5, 6, 7],
        "logprobs": [0.0, 0.0, -0.5, -0.4, -0.3],
        "advantages": [0.0, 0.0, 0.75, 0.75, 0.75],
    }


def test_rl_datums_reject_empty_or_misaligned_sample_data() -> None:
    runtime = RuntimeBundle(_sdk(), object(), object(), Renderer(), object())
    empty = SimpleNamespace(prompt=Input([1]), tokens=(), logprobs=())
    mismatch = SimpleNamespace(prompt=Input([1]), tokens=(2, 3), logprobs=(-0.2,))
    with pytest.raises(ValueError, match="tokens and finite"):
        rl_datums(runtime, [empty], [1.0])
    with pytest.raises(ValueError, match="tokens and finite"):
        rl_datums(runtime, [mismatch], [1.0])


class Future:
    def __init__(self, value) -> None:  # type: ignore[no-untyped-def]
        self.value = value

    async def result_async(self):  # type: ignore[no-untyped-def]
        return self.value


class Model:
    def __init__(self, ledger: TokenLedger | None = None) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, object]] = []

    async def get_info_async(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            is_lora=True,
            lora_rank=LORA_RANK,
            model_data=SimpleNamespace(model_name=MODEL_ID),
        )

    def get_tokenizer(self):  # type: ignore[no-untyped-def]
        return "tokenizer"

    async def forward_backward_async(self, datums, loss_fn):  # type: ignore[no-untyped-def]
        assert self.ledger is not None and self.ledger.has_pending_call
        self.calls.append(("forward", loss_fn))
        return Future(SimpleNamespace(metrics={"loss": 1.25}))

    async def optim_step_async(self, params):  # type: ignore[no-untyped-def]
        self.calls.append(("optim", params))
        return Future(SimpleNamespace(metrics={"lr": 0.01}))


def test_model_creation_and_sft_or_rl_update_use_pinned_sdk_calls() -> None:
    model = Model()

    creation_ledger = TokenLedger(TokenBudget(0, 0, 0), 0.0)

    class Service:
        async def create_lora_training_client_async(self, **kwargs):  # type: ignore[no-untyped-def]
            assert creation_ledger.has_pending_call
            self.kwargs = kwargs
            return model

    service = Service()
    sdk = _sdk()
    sdk = SDKBundle(
        sdk.tinker,
        lambda name, tokenizer, **kwargs: (name, tokenizer, kwargs),
        sdk.train_on_what,
        sdk.conversation_to_datum,
        sdk.sdk_version,
        sdk.cookbook_version,
    )
    runtime = asyncio.run(resolve_model(sdk, service, seed=11, ledger=creation_ledger))
    assert creation_ledger.committed == creation_ledger.observed == TokenBudget(0, 0, 0)
    assert service.kwargs == {
        "base_model": MODEL_ID,
        "rank": LORA_RANK,
        "seed": 11,
        "train_mlp": True,
        "train_attn": True,
        "train_unembed": True,
        "user_metadata": None,
    }

    ledger = TokenLedger(TokenBudget(0, 0, 20), 1.0)
    model.ledger = ledger
    metrics = asyncio.run(
        apply_update(
            runtime,
            [SimpleNamespace(model_input=Input([1, 2, 3]))],
            loss_fn="cross_entropy",
            learning_rate=1e-4,
            ledger=ledger,
        )
    )
    assert model.calls[0] == ("forward", "cross_entropy")
    assert metrics == {
        "forward_backward.loss": 1.25,
        "optimizer.lr": 0.01,
        "local.train_tokens": 3.0,
    }
    assert ledger.committed == ledger.observed == TokenBudget(0, 0, 3)


def test_measurement_rejects_non_suffix_weights() -> None:
    sdk = _sdk()
    sdk = SDKBundle(
        sdk.tinker,
        sdk.get_renderer,
        sdk.train_on_what,
        lambda *args, **kwargs: Datum(
            Input([10, 11, 12, 20, 21]), {"weights": [0, 1, 0, 1, 1]}
        ),
        sdk.sdk_version,
        sdk.cookbook_version,
    )
    runtime = RuntimeBundle(sdk, object(), object(), Renderer(), object())
    with pytest.raises(ValueError, match="one nonempty suffix"):
        teacher_token_measurer(runtime)(_source())


@pytest.mark.parametrize("learning_rate", [0.0, -1e-4, float("nan"), float("inf")])
def test_update_rejects_invalid_learning_rate_before_remote_call(
    learning_rate: float,
) -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 20), 1.0)
    model = Model(ledger)
    runtime = RuntimeBundle(_sdk(), object(), model, Renderer(), object())
    with pytest.raises(ValueError, match="positive finite"):
        asyncio.run(
            apply_update(
                runtime,
                [SimpleNamespace(model_input=Input([1, 2, 3]))],
                loss_fn="cross_entropy",
                learning_rate=learning_rate,
                ledger=ledger,
            )
        )
    assert not model.calls and not ledger.has_pending_call
