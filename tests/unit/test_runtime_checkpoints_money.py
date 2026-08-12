from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import math
from types import SimpleNamespace

import pytest

from duraseed.runtime import (
    PRICE_SNAPSHOT,
    RuntimeBundle,
    SDKBundle,
    TokenBudget,
    TokenLedger,
    UsageQuantities,
    parse_billing_usage,
    restore_checkpoint,
    save_checkpoint,
    save_sampler_checkpoint,
    set_and_verify_ttl,
)


class Future:
    def __init__(self, value) -> None:  # type: ignore[no-untyped-def]
        self.value = value

    async def result_async(self):  # type: ignore[no-untyped-def]
        return self.value


class Model:
    def __init__(self, ledger: TokenLedger) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, str, object]] = []

    async def save_weights_for_sampler_async(self, name, *, ttl_seconds):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        self.calls.append(("sampler", name, ttl_seconds))
        return Future(SimpleNamespace(path=f"tinker://train/weights/{name}"))

    async def save_state_async(self, name, *, ttl_seconds, overwrite):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call and overwrite is False
        self.calls.append(("state", name, ttl_seconds))
        return Future(SimpleNamespace(path=f"tinker://train/state/{name}"))


class Service:
    def __init__(self, ledger: TokenLedger) -> None:
        self.ledger = ledger
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def create_training_client_from_state_with_optimizer_async(
        self,
        path,
        **kwargs,  # type: ignore[no-untyped-def]
    ):
        assert self.ledger.has_pending_call
        self.calls.append(("full", path, kwargs))
        return "full-client"

    async def create_training_client_from_state_async(self, path, **kwargs):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        self.calls.append(("weights", path, kwargs))
        return "weights-client"


class ParsedPath:
    @classmethod
    def from_tinker_path(cls, path: str):  # type: ignore[no-untyped-def]
        assert path.startswith("tinker://train/")
        return SimpleNamespace(training_run_id="train")


def _runtime(ledger: TokenLedger) -> RuntimeBundle:
    sdk = SDKBundle(
        tinker=SimpleNamespace(ParsedCheckpointTinkerPath=ParsedPath),
        get_renderer=None,
        train_on_what=None,
        conversation_to_datum=None,
        sdk_version="0.25.0",
        cookbook_version="0.5.3",
    )
    return RuntimeBundle(sdk, Service(ledger), Model(ledger), object(), object())


def test_sampler_pair_save_and_exact_restore_methods() -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 0), 2.0)
    runtime = _runtime(ledger)
    pair = asyncio.run(
        save_checkpoint(
            runtime,
            name="step-10",
            ttl_seconds=3600,
            ledger=ledger,
            reserved_storage_usd=0.2,
        )
    )
    sampler_path = asyncio.run(
        save_sampler_checkpoint(
            runtime,
            name="step-20",
            ttl_seconds=None,
            ledger=ledger,
            reserved_storage_usd=0.1,
        )
    )
    full = asyncio.run(
        restore_checkpoint(
            runtime,
            pair.state_path,
            full_state=True,
            ledger=ledger,
            user_metadata={"phase": "stage_a"},
        )
    )
    weights = asyncio.run(
        restore_checkpoint(
            runtime,
            pair.state_path,
            full_state=False,
            ledger=ledger,
        )
    )

    assert pair.sampler_path.endswith("step-10-sampler")
    assert pair.state_path.endswith("step-10-state")
    assert sampler_path.endswith("step-20")
    assert runtime.model.calls == [
        ("sampler", "step-10-sampler", 3600),
        ("state", "step-10-state", 3600),
        ("sampler", "step-20", None),
    ]
    assert full == "full-client" and weights == "weights-client"
    assert runtime.service.calls == [
        ("full", pair.state_path, {"user_metadata": {"phase": "stage_a"}}),
        ("weights", pair.state_path, {"user_metadata": None}),
    ]
    assert all("base_model" not in call[2] for call in runtime.service.calls)
    assert ledger.committed_fixed_usd == ledger.observed_fixed_usd == pytest.approx(0.3)


class RestClient:
    def __init__(self, ledger: TokenLedger) -> None:
        self.ledger = ledger
        self.expiry = object()
        self.expiry_offset = 0
        self.set_calls: list[tuple[str, int | None]] = []

    async def set_checkpoint_ttl_from_tinker_path_async(self, path, ttl):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        self.set_calls.append((path, ttl))
        self.expiry = (
            None
            if ttl is None
            else datetime.now(UTC) + timedelta(seconds=ttl + self.expiry_offset)
        )

    async def list_checkpoints_async(self, run_id):  # type: ignore[no-untyped-def]
        assert self.ledger.has_pending_call
        assert run_id == "train"
        return SimpleNamespace(
            checkpoints=[
                SimpleNamespace(
                    tinker_path="tinker://train/state/checkpoint",
                    expires_at=self.expiry,
                    checkpoint_type="state",
                )
            ]
        )


def test_ttl_verifies_expiring_and_nonexpiring_semantics() -> None:
    ledger = TokenLedger(TokenBudget(0, 0, 0), 1.0)
    runtime = _runtime(ledger)
    rest = RestClient(ledger)
    path = "tinker://train/state/checkpoint"
    finite = asyncio.run(
        set_and_verify_ttl(runtime, rest, [path], ttl_seconds=7200, ledger=ledger)
    )
    forever = asyncio.run(
        set_and_verify_ttl(runtime, rest, [path], ttl_seconds=None, ledger=ledger)
    )
    assert finite[0].expires_at is not None and finite[0].ttl_seconds == 7200
    assert forever[0].expires_at is None and forever[0].ttl_seconds is None
    assert rest.set_calls == [(path, 7200), (path, None)]

    rest.expiry_offset = -600
    with pytest.raises(RuntimeError, match="TTL could not be verified"):
        asyncio.run(
            set_and_verify_ttl(runtime, rest, [path], ttl_seconds=7200, ledger=ledger)
        )


def _event(kind: str, session: str | None = "session", **values):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        session_id=session,
        project_id="project",
        event_info=SimpleNamespace(type=kind, **values),
    )


def test_billing_parser_filters_session_and_prices_all_usage_categories() -> None:
    export = SimpleNamespace(
        data=[
            _event("sampling_prefill", token_count=100, cached=False),
            _event("sampling_prefill", token_count=50, cached=True),
            _event("sampling_sample", token_count=200),
            _event("training", token_count=300),
            _event("checkpoint", count=2),
            _event("storage", session=None, gigabyte_hours=72.0),
            _event("training", session="other", token_count=999),
        ]
    )
    usage = parse_billing_usage(export, session_id="session", project_id="project")
    assert usage == UsageQuantities(100, 50, 200, 300, 2, 72.0)
    expected = (100 * 0.66 + 50 * 0.132 + 200 * 1.995 + 300 * 1.463) / 1e6
    expected += 72 * 0.10 / (30 * 24)
    assert PRICE_SNAPSHOT.cost(replace(usage, checkpoint_count=0)) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="checkpoint event pricing"):
        PRICE_SNAPSHOT.cost(usage)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1.0])
def test_ledger_rejects_nonfinite_or_negative_money(value: float) -> None:
    with pytest.raises(ValueError):
        TokenLedger(TokenBudget(1, 1, 1), value)
    ledger = TokenLedger(TokenBudget(1, 1, 1), 1.0)
    with pytest.raises(ValueError):
        ledger.reserve_call(TokenBudget(0, 0, 0), fixed_usd=value)


def test_fixed_reservations_are_observed_on_success_and_failure() -> None:
    ledger = TokenLedger(TokenBudget(10, 10, 10), 2.0)
    ledger.reserve_call(TokenBudget(1, 2, 3), fixed_usd=0.25)
    ledger.settle_call(TokenBudget(1, 1, 2))
    ledger.reserve_call(TokenBudget(2, 3, 4), fixed_usd=0.5)
    ledger.abort_call()
    assert ledger.observed == TokenBudget(3, 4, 6)
    assert ledger.observed_fixed_usd == pytest.approx(0.75)
    assert ledger.observed_cost_usd > 0.75


def test_ledger_fails_closed_on_nested_or_over_budget_reservations() -> None:
    ledger = TokenLedger(TokenBudget(2, 2, 2), 0.01)
    ledger.reserve_call(TokenBudget(1, 1, 1))
    with pytest.raises(RuntimeError, match="previous paid call"):
        ledger.reserve_call(TokenBudget(0, 0, 0))
    ledger.settle_call(TokenBudget(1, 1, 1))
    with pytest.raises(RuntimeError, match="token reservation"):
        ledger.reserve_call(TokenBudget(2, 0, 0))


def test_storage_usage_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        UsageQuantities(storage_gb_hours=math.nan)
    with pytest.raises(ValueError, match="price rates"):
        replace(PRICE_SNAPSHOT, sample_per_million_usd=math.nan)
