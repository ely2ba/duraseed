"""Lazy, pinned construction of the concrete Tinker runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import importlib.metadata
import math
from typing import Any, Literal

from duraseed.runtime.ledger import TokenBudget, TokenLedger
from duraseed.training_metric_errors import NonFiniteTrainingMetricError


TINKER_VERSION = "0.25.0"
COOKBOOK_VERSION = "0.5.3"
MODEL_ID = "Qwen/Qwen3.5-9B-Base"
RENDERER_NAME = "role_colon"
LORA_RANK = 32


class RuntimeUnavailableError(RuntimeError):
    """The optional paid runtime is absent or differs from the frozen contract."""


@dataclass(frozen=True, slots=True)
class SDKBundle:
    tinker: Any
    get_renderer: Any
    train_on_what: Any
    conversation_to_datum: Any
    sdk_version: str
    cookbook_version: str


@dataclass(frozen=True, slots=True)
class RuntimeBundle:
    sdk: SDKBundle
    service: Any
    model: Any
    renderer: Any
    tokenizer: Any


def bind_model(sdk: SDKBundle, service: Any, model: Any) -> RuntimeBundle:
    tokenizer = model.get_tokenizer()
    renderer = sdk.get_renderer(RENDERER_NAME, tokenizer, model_name=MODEL_ID)
    return RuntimeBundle(sdk, service, model, renderer, tokenizer)


def load_sdk() -> SDKBundle:
    """Load the optional SDK only at the authorized execution boundary."""

    try:
        sdk_version = importlib.metadata.version("tinker")
        cookbook_version = importlib.metadata.version("tinker-cookbook")
        if sdk_version != TINKER_VERSION or cookbook_version != COOKBOOK_VERSION:
            raise RuntimeUnavailableError(
                "Tinker runtime version mismatch: "
                f"tinker={sdk_version}, cookbook={cookbook_version}"
            )
        tinker = importlib.import_module("tinker")
        renderers = importlib.import_module("tinker_cookbook.renderers")
        supervised = importlib.import_module("tinker_cookbook.supervised.data")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise RuntimeUnavailableError(
            "install the pinned runtime with `uv sync --extra tinker`"
        ) from error
    return SDKBundle(
        tinker=tinker,
        get_renderer=renderers.get_renderer,
        train_on_what=renderers.TrainOnWhat,
        conversation_to_datum=supervised.conversation_to_datum,
        sdk_version=sdk_version,
        cookbook_version=cookbook_version,
    )


def create_service(
    sdk: SDKBundle,
    *,
    project_id: str,
    user_metadata: dict[str, str] | None = None,
) -> Any:
    if not project_id.strip():
        raise ValueError("an explicit Tinker project_id is required")
    return sdk.tinker.ServiceClient(
        project_id=project_id,
        user_metadata=user_metadata,
    )


async def resolve_model(
    sdk: SDKBundle,
    service: Any,
    *,
    seed: int,
    ledger: TokenLedger,
    user_metadata: dict[str, str] | None = None,
) -> RuntimeBundle:
    """Create the pinned rank-32 LoRA model and its role-colon renderer."""

    ledger.reserve_call(TokenBudget(0, 0, 0))
    try:
        model = await service.create_lora_training_client_async(
            base_model=MODEL_ID,
            rank=LORA_RANK,
            seed=seed,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
            user_metadata=user_metadata,
        )
        if hasattr(model, "get_info_async"):
            info = await model.get_info_async()
            model_name = getattr(getattr(info, "model_data", None), "model_name", None)
            if (
                getattr(info, "is_lora", None) is not True
                or getattr(info, "lora_rank", None) != LORA_RANK
                or model_name not in (None, MODEL_ID)
            ):
                raise RuntimeError("created model violates the pinned LoRA contract")
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(TokenBudget(0, 0, 0))
    return bind_model(sdk, service, model)


async def create_sampler(
    runtime: RuntimeBundle,
    *,
    ledger: TokenLedger,
    checkpoint_path: str | None = None,
) -> Any:
    if checkpoint_path is not None and not checkpoint_path.strip():
        raise ValueError("checkpoint_path must be nonempty")
    ledger.reserve_call(TokenBudget(0, 0, 0))
    try:
        if checkpoint_path is None:
            result = await runtime.service.create_sampling_client_async(
                base_model=MODEL_ID
            )
        else:
            result = await runtime.service.create_sampling_client_async(
                model_path=checkpoint_path
            )
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(TokenBudget(0, 0, 0))
    return result


def _metrics(prefix: str, response: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in (getattr(response, "metrics", None) or {}).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"non-numeric Tinker metric {prefix}.{name}") from error
        if not math.isfinite(numeric):
            raise NonFiniteTrainingMetricError(f"{prefix}.{name}")
        result[f"{prefix}.{name}"] = numeric
    return result


async def apply_update(
    runtime: RuntimeBundle,
    datums: list[Any],
    *,
    loss_fn: Literal["cross_entropy", "importance_sampling"],
    learning_rate: float,
    ledger: TokenLedger,
) -> dict[str, float]:
    if not datums:
        raise ValueError("an update requires at least one datum")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number")
    train_tokens = sum(int(datum.model_input.length) for datum in datums)
    usage = TokenBudget(0, 0, train_tokens)
    ledger.reserve_call(usage)
    try:
        forward_future = await runtime.model.forward_backward_async(datums, loss_fn)
        optimizer_future = await runtime.model.optim_step_async(
            runtime.sdk.tinker.AdamParams(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1.0e-12,
                weight_decay=0.0,
                grad_clip_norm=0.0,
            )
        )
        forward, optimizer = await asyncio.gather(
            forward_future.result_async(), optimizer_future.result_async()
        )
    except Exception:
        ledger.abort_call()
        raise
    ledger.settle_call(usage)
    return {
        **_metrics("forward_backward", forward),
        **_metrics("optimizer", optimizer),
        "local.train_tokens": float(train_tokens),
    }


__all__ = [
    "COOKBOOK_VERSION",
    "LORA_RANK",
    "MODEL_ID",
    "RENDERER_NAME",
    "RuntimeBundle",
    "RuntimeUnavailableError",
    "SDKBundle",
    "TINKER_VERSION",
    "apply_update",
    "bind_model",
    "create_sampler",
    "create_service",
    "load_sdk",
    "resolve_model",
]
