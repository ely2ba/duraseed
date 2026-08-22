"""Manifest-derived pessimistic preflight for one paired-seed Pilot launch."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

from duraseed.pilot0_contract import (
    BG_STAGE_A_GRID,
    BS_STAGE_A_GRID,
    EPHEMERAL_SAMPLER_FIXED_USD,
    PILOT_PAIR_PLANNING_CAP_USD,
    STAGE_B_GRID,
    STAGE_B_MAX_TOKENS,
    Pilot0Inputs,
)
from duraseed.pilot0_data import (
    ordered_stage_a_pools,
    scheduled_stage_a_records,
    stage_a_solver_sources,
    stage_b_sources,
)
from duraseed.runners import Action, RunPlan, RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities, sft_datum
from duraseed.tasks.maps import render_prompt as render_maps_prompt
from duraseed.tasks.tces import render_prompt as render_tces_prompt
from duraseed.training.capability_dose_evidence import EPOCH_UPDATES


@dataclass(frozen=True, slots=True)
class Pilot0Budget:
    tokens: TokenBudget
    fixed_storage_usd: float
    upper_bound_usd: float
    cent_ceiling_usd: Decimal
    passed: bool
    rerun_policy: str = "no_rerun_without_new_authorization"


def _prompt_length(inputs: Pilot0Inputs, record: object) -> int:
    prompt_text = (
        render_tces_prompt(record.to_task())  # type: ignore[attr-defined]
        if getattr(record, "task_family", None) == "tces"
        else render_maps_prompt(record.to_task())  # type: ignore[attr-defined]
    )
    prompt = inputs.runtime.renderer.build_generation_prompt(
        [{"role": "user", "content": prompt_text}], role="assistant"
    )
    return int(prompt.length)


def _evaluation(
    inputs: Pilot0Inputs,
    manifest: object,
    *,
    repetitions: int,
    samples: int,
    max_tokens: int,
) -> TokenBudget:
    records = manifest.records  # type: ignore[attr-defined]
    prompt = sum(_prompt_length(inputs, row) for row in records)
    count = manifest.record_count  # type: ignore[attr-defined]
    return TokenBudget(
        prompt * samples * repetitions,
        count * samples * max_tokens * repetitions,
        0,
    )


def _sum(values: list[TokenBudget]) -> TokenBudget:
    total = TokenBudget(0, 0, 0)
    for value in values:
        total = total.plus(value)
    return total


def calculate_pilot0_budget(inputs: Pilot0Inputs) -> Pilot0Budget:
    """Bound the exact frozen pair path before a service is created."""

    source = inputs.source
    budgets: list[TokenBudget] = []
    pools = ordered_stage_a_pools(source)
    solvers = stage_a_solver_sources(source)
    bs_train = 0
    for step in range(1, BS_STAGE_A_GRID[-1] + 1):
        records = scheduled_stage_a_records(
            pools,
            source.prompt_pools.artifact.bs_slot_order,
            ((step - 1) % EPOCH_UPDATES) + 1,
        )
        bs_train += sum(
            int(sft_datum(inputs.runtime, solvers[row.task_id]).model_input.length)
            for row in records
        )
    budgets.append(TokenBudget(0, 0, bs_train))

    bg_prefill = bg_sample = bg_train = 0
    for step in range(1, BG_STAGE_A_GRID[-1] + 1):
        records = scheduled_stage_a_records(
            pools, source.prompt_pools.artifact.bg_group_order, step
        )
        for record in records:
            prompt = _prompt_length(inputs, record)
            bg_prefill += prompt * 8
            bg_sample += inputs.acquisition.selected_max_tokens * 8
            bg_train += (prompt + inputs.acquisition.selected_max_tokens - 1) * 8
    budgets.append(TokenBudget(bg_prefill, bg_sample, bg_train))

    cadence_count = (len(BS_STAGE_A_GRID) - 2) + (len(BG_STAGE_A_GRID) - 1)
    budgets.append(
        _evaluation(
            inputs,
            source.a_cadence,
            repetitions=cadence_count,
            samples=1,
            max_tokens=inputs.acquisition.selected_max_tokens,
        )
    )
    budgets.append(
        _evaluation(
            inputs,
            source.prompt_pools.a_monitor_manifest,
            repetitions=2,
            samples=4,
            max_tokens=inputs.acquisition.selected_max_tokens,
        )
    )
    budgets.append(
        _evaluation(
            inputs,
            source.a_validation,
            repetitions=4,
            samples=16,
            max_tokens=inputs.acquisition.selected_max_tokens,
        )
    )

    maps = [sft_datum(inputs.runtime, row) for row in stage_b_sources(source)]
    stage_b_train = 0
    for step in range(1, STAGE_B_GRID[-1] + 1):
        stage_b_train += sum(
            int(maps[((step - 1) * 32 + offset) % len(maps)].model_input.length)
            for offset in range(32)
        )
    budgets.append(TokenBudget(0, 0, stage_b_train * 2))
    budgets.append(
        _evaluation(
            inputs,
            source.b_validation,
            repetitions=2 * len(STAGE_B_GRID),
            samples=16,
            max_tokens=STAGE_B_MAX_TOKENS,
        )
    )
    budgets.append(
        _evaluation(
            inputs,
            source.prompt_pools.a_monitor_manifest,
            repetitions=2 * (len(STAGE_B_GRID) - 1),
            samples=4,
            max_tokens=inputs.acquisition.selected_max_tokens,
        )
    )
    tokens = _sum(budgets)
    stage_a_pairs = (len(BS_STAGE_A_GRID) - 2) + (len(BG_STAGE_A_GRID) - 1)
    stage_b_pairs = 2 * (0.1 * (len(STAGE_B_GRID) - 2) + 1.0)
    fixed = (
        0.1 * stage_a_pairs
        + BG_STAGE_A_GRID[-1] * EPHEMERAL_SAMPLER_FIXED_USD
        + stage_b_pairs
    )
    upper = (
        PRICE_SNAPSHOT.cost(
            UsageQuantities(
                prefill_tokens=tokens.prefill,
                sample_tokens=tokens.sample,
                train_tokens=tokens.train,
            )
        )
        + fixed
    )
    cap = Decimal(str(upper)).quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if float(cap) > PILOT_PAIR_PLANNING_CAP_USD:
        raise RunnerGateError(
            f"manifest-derived Pilot pair ceiling ${cap} exceeds the frozen "
            "$774.04 maximum"
        )
    authorized = Decimal(str(inputs.ledger.authorized_usd))
    within_tokens = tokens == inputs.ledger.limits
    return Pilot0Budget(
        tokens,
        fixed,
        upper,
        cap,
        authorized == cap and within_tokens,
    )


def build_pilot0_pair_plan(cap: Decimal) -> RunPlan:
    command = "uv run --isolated --with-editable '.[tinker]' duraseed pilot0-pair-live"
    return RunPlan(
        name="pilot0-pair",
        actions=(Action("pilot0-paired-seed", cap),),
        launch_preconditions=(
            "actual_spend_reconciled",
            "pair_order_valid",
            "human_approval",
        ),
        dry_run_command=f"{command} --help",
        mock_command="uv run pytest tests/unit/test_pilot0_live.py",
        authorization_command=(
            f"{command} --authorized-cost-usd {cap} --confirm-human-launch"
        ),
    )


__all__ = [
    "Pilot0Budget",
    "build_pilot0_pair_plan",
    "calculate_pilot0_budget",
]
