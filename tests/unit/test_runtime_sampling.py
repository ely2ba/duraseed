from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from duraseed.provenance import derive_namespaced_seed
from duraseed.runtime import (
    RuntimeBundle,
    SDKBundle,
    SamplingCoordinates,
    SamplingTask,
    TokenBudget,
    TokenLedger,
    sample_seeded,
)
from duraseed.schemas import ExactRational, TCESTask


MANIFEST_ID = "sha256:" + "c" * 64


class Input:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.length = len(tokens)

    def to_ints(self) -> list[int]:
        return list(self.tokens)


class SamplingParams:
    def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
        self.values = values


class Renderer:
    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        assert messages[0]["role"] == "user" and role == "assistant"
        return Input([10, 11])

    def get_stop_sequences(self) -> list[str]:
        return ["<eos>"]

    def parse_response(self, tokens):  # type: ignore[no-untyped-def]
        assert tokens == [7, 8]
        return None, "stop_sequence"


class Tokenizer:
    eos_token_id = None

    def decode(self, tokens) -> str:  # type: ignore[no-untyped-def]
        return "<answer>2+3</answer>"


class Sampler:
    def __init__(self, ledger: TokenLedger, *, missing_stop: bool = False) -> None:
        self.ledger = ledger
        self.missing_stop = missing_stop
        self.seeds: list[int] = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        assert prompt.length == 2 and num_samples == 1
        self.seeds.append(sampling_params.values["seed"])
        return SimpleNamespace(
            sequences=[
                SimpleNamespace(
                    tokens=[7, 8],
                    logprobs=[-0.2, -0.3],
                    stop_reason=None if self.missing_stop else "stop",
                )
            ]
        )


def _runtime() -> RuntimeBundle:
    sdk = SDKBundle(
        tinker=SimpleNamespace(SamplingParams=SamplingParams),
        get_renderer=None,
        train_on_what=None,
        conversation_to_datum=None,
        sdk_version="0.25.0",
        cookbook_version="0.5.3",
    )
    return RuntimeBundle(sdk, object(), object(), Renderer(), Tokenizer())


def _task(task_id: str, item_index: int) -> SamplingTask:
    return SamplingTask(
        manifest_id=MANIFEST_ID,
        task_id=task_id,
        task_family="tces",
        source_split="a_validation",
        prompt_text="Use 2 and 3 to make 5.",
        exact_task=TCESTask(
            operands=(2, 3),
            target=ExactRational(numerator=5),
        ),
        item_index=item_index,
        assigned_family_id="n2__d1__ops_ADD1__ord_N__frac_II",
        panel_role="boundary",
    )


def _coordinates() -> SamplingCoordinates:
    return SamplingCoordinates(
        run_id="run-1",
        label="refinement",
        purpose="evaluation",
        checkpoint_stage="m0",
        training_step=0,
        sampler_checkpoint_path="tinker://m0/sampler",
        origin_sampler_checkpoint_path="tinker://m0/sampler",
        experiment_seed=5,
        seed_namespace="tinker.boundary.refinement",
    )


def test_seeded_refinement_preserves_indices_records_and_unique_task_ids() -> None:
    runtime = _runtime()
    ledger = TokenLedger(TokenBudget(100, 500, 0), 5.0)
    first_sampler = Sampler(ledger)
    first = asyncio.run(
        sample_seeded(
            runtime,
            first_sampler,
            _task("task-a", 7),
            _coordinates(),
            group_size=12,
            max_tokens=16,
            temperature=1.0,
            top_p=0.95,
            ledger=ledger,
            sample_index_start=4,
        )
    )
    second_sampler = Sampler(ledger)
    second = asyncio.run(
        sample_seeded(
            runtime,
            second_sampler,
            _task("task-b", 8),
            _coordinates(),
            group_size=1,
            max_tokens=16,
            temperature=1.0,
            top_p=0.95,
            ledger=ledger,
        )
    )

    assert tuple(row.generation.sample_index for row in first) == tuple(range(4, 16))
    expected_seeds = tuple(
        derive_namespaced_seed(
            5, "tinker.boundary.refinement", "tces", "task-a", 7, index
        )
        for index in range(4, 16)
    )
    assert tuple(row.generation.sampling_seed for row in first) == expected_seeds
    assert tuple(first_sampler.seeds) == expected_seeds
    all_rows = (*first, *second)
    assert len({row.generation.sample_id for row in all_rows}) == 13
    for row in all_rows:
        assert row.reward.sample_id == row.generation.sample_id
        assert row.reward.task_id == row.generation.task_id
        assert row.generation.completion_token_ids == (7, 8)
        assert row.generation.completion_logprobs == (-0.2, -0.3)
        assert row.generation.stop_reason == "stop"
        assert row.generation.renderer_termination == "stop_sequence"
        assert row.generation.reward == row.reward.reward == 1.0
    assert ledger.committed == TokenBudget(26, 208, 0)
    assert ledger.observed == TokenBudget(26, 26, 0)


def test_sampling_failure_conservatively_settles_reserved_usage() -> None:
    runtime = _runtime()
    ledger = TokenLedger(TokenBudget(10, 20, 0), 1.0)
    sampler = Sampler(ledger, missing_stop=True)
    with pytest.raises(RuntimeError, match="stop reason"):
        asyncio.run(
            sample_seeded(
                runtime,
                sampler,
                _task("task-a", 7),
                _coordinates(),
                group_size=1,
                max_tokens=8,
                temperature=1.0,
                top_p=0.9,
                ledger=ledger,
            )
        )
    assert not ledger.has_pending_call
    assert ledger.observed == TokenBudget(2, 8, 0)
