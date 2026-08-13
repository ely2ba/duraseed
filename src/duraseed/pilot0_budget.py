"""Pessimistic token/storage preflight for the frozen Pilot-0 path."""

from __future__ import annotations

from dataclasses import dataclass
from duraseed.data.manifests import DatasetManifest
from duraseed.pilot0_contract import (
    EPHEMERAL_SAMPLER_FIXED_USD,
    Pilot0Inputs,
    STAGE_A_GRID,
    STAGE_B_GRID,
    STAGE_B_MAX_TOKENS,
)
from duraseed.pilot0_data import (
    boundary_teacher_sources,
    ordered_stage_a_pools,
    scheduled_stage_a_records,
    stage_a_solver_sources,
    stage_b_sources,
)
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities, sft_datum
from duraseed.tasks.maps import render_prompt as render_maps_prompt
from duraseed.tasks.tces import render_prompt as render_tces_prompt


@dataclass(frozen=True, slots=True)
class Pilot0Budget:
    tokens: TokenBudget
    fixed_storage_usd: float
    authorized_usd: float
    upper_bound_usd: float
    passed: bool
    rerun_reservation_usd: float = 0.0
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
    manifest: DatasetManifest,
    *,
    repetitions: int,
    samples: int,
    max_tokens: int,
) -> TokenBudget:
    prompt = sum(_prompt_length(inputs, row) for row in manifest.records)
    return TokenBudget(
        prompt * samples * repetitions,
        manifest.record_count * samples * max_tokens * repetitions,
        0,
    )


def _plus(values: list[TokenBudget]) -> TokenBudget:
    total = TokenBudget(0, 0, 0)
    for value in values:
        total = total.plus(value)
    return total


def calculate_pilot0_budget(inputs: Pilot0Inputs) -> Pilot0Budget:
    """Compute every frozen call reservation before permitting a paid call."""

    budgets: list[TokenBudget] = []
    for source in inputs.seed_sources:
        # M0 and the post-seed shared origin are both observed before branching.
        budgets.extend(
            (
                _evaluation(
                    inputs,
                    source.prompt_pools.a_monitor_manifest,
                    repetitions=1,
                    samples=int(inputs.config.stage_a.monitor_samples_per_item),
                    max_tokens=inputs.acquisition.selected_max_tokens,
                ),
                _evaluation(
                    inputs,
                    source.a_validation,
                    repetitions=1,
                    samples=int(inputs.config.evaluation["pilot_samples_per_item"]),
                    max_tokens=inputs.acquisition.selected_max_tokens,
                ),
            )
        )
        boundary = [
            sft_datum(inputs.runtime, row)
            for row in boundary_teacher_sources(inputs, source)
        ]
        boundary_train = 0
        for step in range(1, inputs.config.teacher_dose.calibration_updates + 1):
            boundary_train += sum(
                int(
                    boundary[
                        ((step - 1) * 32 + offset) % len(boundary)
                    ].model_input.length
                )
                for offset in range(32)
            )
        budgets.append(TokenBudget(0, 0, boundary_train))
        budgets.extend(
            (
                _evaluation(
                    inputs,
                    source.prompt_pools.a_monitor_manifest,
                    repetitions=1,
                    samples=int(inputs.config.stage_a.monitor_samples_per_item),
                    max_tokens=inputs.acquisition.selected_max_tokens,
                ),
                _evaluation(
                    inputs,
                    source.a_validation,
                    repetitions=1,
                    samples=int(inputs.config.evaluation["pilot_samples_per_item"]),
                    max_tokens=inputs.acquisition.selected_max_tokens,
                ),
            )
        )
        pools = ordered_stage_a_pools(source)
        solvers = stage_a_solver_sources(source)
        bs_train = 0
        for step in range(1, STAGE_A_GRID[-1] + 1):
            records = scheduled_stage_a_records(
                pools, source.prompt_pools.artifact.bs_slot_order, step
            )
            bs_train += sum(
                int(sft_datum(inputs.runtime, solvers[row.task_id]).model_input.length)
                for row in records
            )
        budgets.append(TokenBudget(0, 0, bs_train))
        rl_prefill = rl_sample = rl_train = 0
        for step in range(1, STAGE_A_GRID[-1] + 1):
            records = scheduled_stage_a_records(
                pools, source.prompt_pools.artifact.bg_group_order, step
            )
            for record in records:
                prompt = _prompt_length(inputs, record)
                rl_prefill += prompt * 8
                rl_sample += inputs.acquisition.selected_max_tokens * 8
                rl_train += (prompt + inputs.acquisition.selected_max_tokens - 1) * 8
        budgets.append(TokenBudget(rl_prefill, rl_sample, rl_train))
        budgets.append(
            _evaluation(
                inputs,
                source.prompt_pools.a_monitor_manifest,
                repetitions=6,
                samples=int(inputs.config.stage_a.monitor_samples_per_item),
                max_tokens=inputs.acquisition.selected_max_tokens,
            )
        )
        budgets.append(
            _evaluation(
                inputs,
                source.a_validation,
                repetitions=2,
                samples=int(inputs.config.evaluation["pilot_samples_per_item"]),
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
                samples=int(inputs.config.evaluation["pilot_samples_per_item"]),
                max_tokens=STAGE_B_MAX_TOKENS,
            )
        )
        budgets.append(
            _evaluation(
                inputs,
                source.prompt_pools.a_monitor_manifest,
                repetitions=2 * (len(STAGE_B_GRID) - 1),
                samples=int(inputs.config.stage_a.monitor_samples_per_item),
                max_tokens=inputs.acquisition.selected_max_tokens,
            )
        )
        budgets.append(
            _evaluation(
                inputs,
                source.a_validation,
                repetitions=2,
                samples=int(inputs.config.evaluation["pilot_samples_per_item"]),
                max_tokens=inputs.acquisition.selected_max_tokens,
            )
        )
    tokens = _plus(budgets)
    ephemeral_samplers = len(inputs.seed_sources) * STAGE_A_GRID[-1]
    fixed = (
        2 * 1.0
        + 4 * (2 * 0.1 + 1.0)
        + 4 * (9 * 0.1 + 1.0)
        + ephemeral_samplers * EPHEMERAL_SAMPLER_FIXED_USD
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
    authorized = inputs.ledger.authorized_usd
    limits = inputs.ledger.limits
    within_tokens = (
        tokens.prefill <= limits.prefill
        and tokens.sample <= limits.sample
        and tokens.train <= limits.train
    )
    return Pilot0Budget(
        tokens, fixed, authorized, upper, upper <= authorized and within_tokens
    )


__all__ = ["Pilot0Budget", "calculate_pilot0_budget"]
