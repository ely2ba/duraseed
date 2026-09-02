from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.provenance import derive_namespaced_seed
from duraseed.runners.pilot0_sampling import evaluate_manifest
from duraseed.runtime import RuntimeBundle, SDKBundle, TokenBudget, TokenLedger
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


class _Input:
    def __init__(self, values: list[int]) -> None:
        self.values, self.length = values, len(values)


class _SamplingParams:
    def __init__(self, **values) -> None:  # type: ignore[no-untyped-def]
        self.values = values


class _Renderer:
    def build_generation_prompt(self, messages, *, role):  # type: ignore[no-untyped-def]
        return _Input([1, 2])

    def get_stop_sequences(self) -> list[str]:
        return ["</answer>"]

    def parse_response(self, tokens):  # type: ignore[no-untyped-def]
        return None, "stop_sequence"


class _Tokenizer:
    eos_token_id = None

    def decode(self, tokens) -> str:  # type: ignore[no-untyped-def]
        return "".join(chr(value) for value in tokens)


class _ConcurrentSampler:
    def __init__(self) -> None:
        self.active = 0
        self.maximum_active = 0

    async def sample_async(self, **kwargs):  # type: ignore[no-untyped-def]
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        try:
            text = "<answer>0</answer>"
            return SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        tokens=[ord(value) for value in text],
                        logprobs=[-0.1] * len(text),
                        stop_reason="stop",
                    )
                ]
            )
        finally:
            self.active -= 1


def _manifest():
    split = "a_validation"
    config = TCESGeneratorConfig(
        n_operands=3,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_valid_expressions=1_000,
        split=split,
    )
    seed = 11
    generator = TCESGenerator(
        derive_namespaced_seed(seed, "dataset.tces.split", split), config
    )
    records = [build_tces_record(generator.generate(index)) for index in range(2)]
    return build_manifest(
        name="pilot-concurrent-test",
        split=split,
        generator_version=records[0].generator_version,
        root_seed=seed,
        records=records,
    )


def test_pilot_evaluation_samples_multiple_items_concurrently(tmp_path: Path) -> None:
    manifest = _manifest()
    sampler = _ConcurrentSampler()
    ledger = TokenLedger(TokenBudget(100, 10_000, 0), 25.0)
    runtime = RuntimeBundle(
        SDKBundle(
            SimpleNamespace(SamplingParams=_SamplingParams),
            None,
            None,
            None,
            "0.25.0",
            "0.5.3",
        ),
        object(),
        object(),
        _Renderer(),
        _Tokenizer(),
    )
    inputs = SimpleNamespace(
        runtime=runtime,
        ledger=ledger,
        run_id="pilot-concurrent",
        config=SimpleNamespace(evaluation={"temperature": 1.0, "top_p": 0.95}),
    )
    families = tuple(record.intended_family for record in manifest.records)
    source = SimpleNamespace(
        seed=11,
        prompt_pools=SimpleNamespace(
            artifact=SimpleNamespace(
                boundary_family_ids=families, sentinel_family_ids=("unused",)
            )
        ),
    )
    result = asyncio.run(
        evaluate_manifest(
            inputs,
            source,
            manifest=manifest,
            sampler=sampler,
            sampler_path="tinker://fake/stage-a",
            origin_sampler_path="tinker://fake/origin",
            method="B-S",
            checkpoint_stage="stage_a",
            training_step=50,
            label="pilot-concurrent-stage-a",
            samples_per_item=1,
            max_tokens=128,
            seed_namespace="pilot0.test.concurrent",
            output=tmp_path / "evaluation",
        )
    )
    assert result["row_count"] == 2
    assert sampler.maximum_active == 2
    assert not ledger.has_pending_call
    assert not (tmp_path / "evaluation/.concurrent-groups").exists()
