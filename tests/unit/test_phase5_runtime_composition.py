from __future__ import annotations

import asyncio

from duraseed.runners.boundary_extension import sample_and_summarize
from duraseed.runners.calibration import apply_group_batch, apply_supervised_batch
from duraseed.runtime import RuntimeBundle, TokenBudget, TokenLedger
from tests.unit.test_boundary_scan import _shared_family_manifest
from tests.unit.test_runtime_client_data import Input, Model, Renderer, _sdk, _source
from tests.unit.test_runtime_sampling import Sampler, _coordinates, _runtime


def test_boundary_runtime_rows_feed_the_carried_summary_reducer() -> None:
    manifest = _shared_family_manifest(2)
    ledger = TokenLedger(TokenBudget(100, 500, 0), 5.0)
    summaries = asyncio.run(
        sample_and_summarize(
            _runtime(),
            Sampler(ledger),
            manifest,
            _coordinates(),
            samples_per_item=4,
            informative_group_size=8,
            sample_index_start=0,
            max_tokens=16,
            temperature=1.0,
            top_p=0.95,
            ledger=ledger,
        )
    )
    assert len(summaries) == 1
    assert summaries[0].group_size == 8
    assert all(item.trials == 4 for item in summaries[0].items)


def test_calibration_supervised_batch_uses_shared_runtime() -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 20), 1.0)
    model = Model(ledger)
    runtime = RuntimeBundle(_sdk(), object(), model, Renderer(), object())
    metrics = asyncio.run(
        apply_supervised_batch(
            runtime,
            (_source(),),
            learning_rate=1e-4,
            ledger=ledger,
        )
    )
    assert model.calls[0] == ("forward", "cross_entropy")
    assert metrics["local.train_tokens"] == 5.0


def test_calibration_group_batch_uses_shared_runtime() -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 20), 1.0)
    model = Model(ledger)
    runtime = RuntimeBundle(_sdk(), object(), model, Renderer(), object())
    row = type(
        "Observation",
        (),
        {"prompt": Input([1, 2]), "tokens": (3, 4), "logprobs": (-0.2, -0.3)},
    )()
    metrics = asyncio.run(
        apply_group_batch(
            runtime,
            (row,),
            (1.0,),
            learning_rate=1e-5,
            ledger=ledger,
        )
    )
    assert model.calls[0] == ("forward", "importance_sampling")
    assert metrics["local.train_tokens"] == 3.0
