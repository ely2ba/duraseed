"""Exact worst-case budget and launch plan for the frozen B-S dose run."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from math import isclose
from typing import Any

from duraseed.runners import Action, RunPlan, RunnerGateError
from duraseed.runners.calibration import validate_frozen_maps
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities
from duraseed.tasks.tces import render_prompt
from duraseed.training.capability_dose import CADENCE_UPDATES


DOSE_TRAIN_TOKEN_CEILING = 1_898_772
DOSE_FIXED_STORAGE_USD = 1.60
DOSE_TOKEN_CEILING = TokenBudget(875_493, 27_131_904, DOSE_TRAIN_TOKEN_CEILING)
DOSE_PINNED_UPPER_USD = 59.083877296000004
DOSE_CAP_USD = Decimal("59.09")


@dataclass(frozen=True, slots=True)
class CapabilityDoseBudget:
    tokens: TokenBudget
    fixed_storage_usd: float
    upper_bound_usd: float
    cent_ceiling_usd: Decimal


def _records(inputs: Any, role: str) -> tuple[Any, ...]:
    families = (
        inputs.prompt_pools.artifact.boundary_family_ids
        if role == "targeted"
        else inputs.prompt_pools.artifact.sentinel_family_ids
    )
    rows = []
    for family in families:
        values = sorted(
            (
                row
                for row in inputs.prompt_pools.a_monitor_manifest.records
                if row.intended_family == family
            ),
            key=lambda row: (row.item_index, row.task_id),
        )[:8]
        if len(values) != 8:
            raise RunnerGateError("capability-dose monitor family is incomplete")
        rows.extend(values)
    if len(rows) != 96:
        raise RunnerGateError("capability-dose monitor must contain 96 items")
    return tuple(rows)


def _prompt_tokens(inputs: Any, rows: tuple[Any, ...]) -> int:
    total = 0
    for row in rows:
        text = render_prompt(row.to_task())
        prompt = inputs.runtime.renderer.build_generation_prompt(
            [{"role": "user", "content": text}], role="assistant"
        )
        total += int(prompt.length)
    return total


def capability_dose_budget(inputs: Any) -> CapabilityDoseBudget:
    """Compute and verify the exact frozen full-path ceiling."""

    validate_frozen_maps(inputs.config)
    targeted = _records(inputs, "targeted")
    sentinel = _records(inputs, "sentinel")
    target_prompt = _prompt_tokens(inputs, targeted)
    sentinel_prompt = _prompt_tokens(inputs, sentinel)
    target_multiplicity = 2 + len(CADENCE_UPDATES) + 6 + 2
    sentinel_multiplicity = 1 + len(CADENCE_UPDATES)
    completions = 96 * (target_multiplicity + sentinel_multiplicity)
    tokens = TokenBudget(
        target_prompt * target_multiplicity + sentinel_prompt * sentinel_multiplicity,
        completions * inputs.max_tokens.selected_max_tokens,
        DOSE_TRAIN_TOKEN_CEILING,
    )
    upper = (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        + DOSE_FIXED_STORAGE_USD
    )
    cap = Decimal(str(upper)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if (
        tokens != DOSE_TOKEN_CEILING
        or not isclose(upper, DOSE_PINNED_UPPER_USD, rel_tol=0, abs_tol=1e-12)
        or cap != DOSE_CAP_USD
    ):
        raise RunnerGateError("capability-dose workload differs from its frozen cap")
    return CapabilityDoseBudget(tokens, DOSE_FIXED_STORAGE_USD, upper, cap)


def build_capability_dose_plan(config: Any) -> RunPlan:
    validate_frozen_maps(config)
    command = (
        "uv run --isolated --with-editable '.[tinker]' duraseed capability-dose-live"
    )
    return RunPlan(
        name="capability-dose",
        actions=(Action("b-s-capability-dose", DOSE_CAP_USD),),
        launch_preconditions=("actual_lifetime_reconciled", "human_approval"),
        dry_run_command=f"{command} --help",
        mock_command="uv run pytest tests/unit/test_capability_dose.py",
        authorization_command=(
            f"{command} --authorized-cost-usd {DOSE_CAP_USD} --confirm-human-launch"
        ),
    )


__all__ = [
    "CapabilityDoseBudget",
    "DOSE_CAP_USD",
    "DOSE_FIXED_STORAGE_USD",
    "DOSE_PINNED_UPPER_USD",
    "DOSE_TOKEN_CEILING",
    "build_capability_dose_plan",
    "capability_dose_budget",
]
