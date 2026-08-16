"""Pessimistic token, storage, and action-cap bounds for calibration."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
from typing import Any

from duraseed.data.io import atomic_write_bytes
from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runners.stage_a_evidence import (
    ordered_pools,
    scheduled_records,
)
from duraseed.runners.teacher_dose_evidence import (
    cyclic_batch,
    gate_records,
    teacher_families,
)
from duraseed.runtime import PRICE_SNAPSHOT, TokenBudget, UsageQuantities
from duraseed.runtime.data import SUPERVISED_MAX_LENGTH
from duraseed.tasks.tces import render_prompt
from duraseed.training.stage_a_calibration import STAGE_A_LEARNING_RATE_GRIDS


@dataclass(frozen=True, slots=True)
class CalibrationBudget:
    tokens: TokenBudget
    fixed_storage_usd: float
    upper_bound_usd: float


@dataclass(frozen=True, slots=True)
class CalibrationAllocation:
    teacher_tokens: TokenBudget
    stage_a_tokens: TokenBudget
    teacher_cap_usd: float
    stage_a_cap_usd: float
    aggregate_cap_usd: float = 300.0


def _safe_arm(value: str) -> str:
    return value.replace("+", "plus").replace(".", "p")


def _plus(*values: TokenBudget) -> TokenBudget:
    result = TokenBudget(0, 0, 0)
    for value in values:
        result = result.plus(value)
    return result


def _cost(tokens: TokenBudget, fixed: float) -> CalibrationBudget:
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
    return CalibrationBudget(tokens, fixed, upper)


def _cent_ceiling(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def _prompt_length(inputs: Any, record: Any) -> int:
    text = render_prompt(record.to_task())
    prompt = inputs.runtime.renderer.build_generation_prompt(
        [{"role": "user", "content": text}], role="assistant"
    )
    return int(prompt.length)


def _sample(inputs: Any, records: tuple, samples: int) -> TokenBudget:
    return TokenBudget(
        sum(_prompt_length(inputs, row) for row in records) * samples,
        len(records) * samples * inputs.max_tokens.selected_max_tokens,
        0,
    )


_SFT_TOKENS_PER_DATUM = SUPERVISED_MAX_LENGTH - 1


def _teacher_source_count(inputs: Any, families: tuple[str, ...], dose: int) -> int:
    records = inputs.teacher_sources.target_train_manifest.records
    if any(not isinstance(row, TCESTaskManifestRecord) for row in records):
        raise RunnerGateError("teacher training manifest is not TCES")
    by_family = {
        family: sum(row.intended_family == family for row in records)
        for family in families
    }
    if any(count < dose for count in by_family.values()):
        raise RunnerGateError("teacher training manifest cannot supply configured dose")
    return len(families) * dose


def _cyclic_train(lengths: list[int], updates: int) -> int:
    return sum(
        row for step in range(1, updates + 1) for row in cyclic_batch(lengths, step)
    )


def teacher_dose_budget(
    inputs: Any, completed_arm_ids: frozenset[str] = frozenset()
) -> CalibrationBudget:
    """Bound every configured arm plus the possible seed-37 verification arm."""

    updates = inputs.config.teacher_dose.calibration_updates
    rates = inputs.config.tinker.learning_rates.teacher_seed_sft.grid
    calibration_target, calibration_sentinel = teacher_families(inputs, 17)
    verification_target, verification_sentinel = teacher_families(inputs, 37)
    budgets = []
    if "baseline-seed-17" not in completed_arm_ids:
        budgets.append(_sample(inputs, gate_records(inputs, calibration_sentinel), 1))
    if "baseline-seed-37" not in completed_arm_ids:
        budgets.append(_sample(inputs, gate_records(inputs, verification_sentinel), 1))
    remaining_training_arms = 0
    for dose in inputs.config.teacher_dose.demonstrations_per_family:
        source_count = _teacher_source_count(inputs, calibration_target, dose)
        lengths = [_SFT_TOKENS_PER_DATUM] * source_count
        arm = _plus(
            TokenBudget(0, 0, _cyclic_train(lengths, updates)),
            _sample(
                inputs,
                gate_records(inputs, calibration_target),
                inputs.config.teacher_dose.gate_samples_per_item,
            ),
            _sample(inputs, gate_records(inputs, calibration_sentinel), 1),
        )
        for rate in rates:
            arm_id = _safe_arm(f"seed-17-dose-{dose}-lr-{rate:.0e}")
            if arm_id not in completed_arm_ids:
                budgets.append(arm)
                remaining_training_arms += 1
    verification_trains = []
    for dose in inputs.config.teacher_dose.demonstrations_per_family:
        source_count = _teacher_source_count(inputs, verification_target, dose)
        lengths = [_SFT_TOKENS_PER_DATUM] * source_count
        verification_trains.append(_cyclic_train(lengths, updates))
    verification = _plus(
        TokenBudget(0, 0, max(verification_trains)),
        _sample(
            inputs,
            gate_records(inputs, verification_target),
            inputs.config.teacher_dose.gate_samples_per_item,
        ),
        _sample(inputs, gate_records(inputs, verification_sentinel), 1),
    )
    verification_prefix = "seed-37-dose-"
    if not any(arm.startswith(verification_prefix) for arm in completed_arm_ids):
        budgets.append(verification)
        remaining_training_arms += 1
    return _cost(_plus(*budgets), remaining_training_arms * 0.05)


def _monitor(inputs: Any, role: str) -> tuple:
    artifact = inputs.prompt_pools.artifact
    families = (
        artifact.boundary_family_ids
        if role == "targeted"
        else artifact.sentinel_family_ids
    )
    return tuple(
        row
        for family in families
        for row in sorted(
            (
                value
                for value in inputs.prompt_pools.a_monitor_manifest.records
                if value.intended_family == family
            ),
            key=lambda value: (value.item_index, value.task_id),
        )[:8]
    )


def stage_a_budget(
    inputs: Any, selected_dose: int, *, completed: bool = False
) -> CalibrationBudget:
    """Bound the complete indivisible six-screen/two-continuation Stage-A run."""

    if completed:
        return _cost(TokenBudget(0, 0, 0), 0.0)
    pools = ordered_pools(inputs.prompt_pools)
    targeted_families, _ = teacher_families(inputs, 17)
    boundary_count = _teacher_source_count(inputs, targeted_families, selected_dose)
    boundary = [_SFT_TOKENS_PER_DATUM] * boundary_count
    budgets = [
        TokenBudget(
            0,
            0,
            _cyclic_train(boundary, inputs.config.teacher_dose.calibration_updates),
        )
    ]
    targeted, sentinel = _monitor(inputs, "targeted"), _monitor(inputs, "sentinel")
    budgets.extend((_sample(inputs, targeted, 2), _sample(inputs, sentinel, 1)))
    bs_train = 0
    bs_schedule = (*range(1, 11),) * len(STAGE_A_LEARNING_RATE_GRIDS["B-S"])
    bs_schedule += tuple(range(11, 51))
    for schedule_step in bs_schedule:
        records = scheduled_records(
            pools, inputs.prompt_pools.artifact.bs_slot_order, schedule_step
        )
        bs_train += len(records) * _SFT_TOKENS_PER_DATUM
    budgets.append(TokenBudget(0, 0, bs_train))
    bg_prefill = bg_sample = bg_train = 0
    bg_schedule = (*range(1, 11),) * len(STAGE_A_LEARNING_RATE_GRIDS["B-G"])
    bg_schedule += tuple(range(11, 51))
    for schedule_step in bg_schedule:
        records = scheduled_records(
            pools, inputs.prompt_pools.artifact.bg_group_order, schedule_step
        )
        for row in records:
            prompt = _prompt_length(inputs, row)
            bg_prefill += prompt * 8
            bg_sample += inputs.max_tokens.selected_max_tokens * 8
            bg_train += (prompt + inputs.max_tokens.selected_max_tokens - 1) * 8
    budgets.append(TokenBudget(bg_prefill, bg_sample, bg_train))
    budgets.extend(
        (
            _sample(inputs, targeted, 6),
            _sample(inputs, sentinel, 6),
            _sample(inputs, targeted, 4),
            _sample(inputs, sentinel, 2),
        )
    )
    fixed = 1.0 + 8 * 0.05 + len(bg_schedule) * 0.05
    return _cost(_plus(*budgets), fixed)


def calibration_allocation(inputs: Any) -> CalibrationAllocation:
    """Allocate the one `$300` launch from complete local workload bounds."""

    teacher = teacher_dose_budget(inputs)
    stage = tuple(
        stage_a_budget(inputs, dose)
        for dose in inputs.config.teacher_dose.demonstrations_per_family
    )
    stage_tokens = TokenBudget(
        max(row.tokens.prefill for row in stage),
        max(row.tokens.sample for row in stage),
        max(row.tokens.train for row in stage),
    )
    stage_envelope = _cost(stage_tokens, max(row.fixed_storage_usd for row in stage))
    teacher_cap = _cent_ceiling(teacher.upper_bound_usd)
    stage_cap = _cent_ceiling(stage_envelope.upper_bound_usd)
    if (
        not all(
            math.isfinite(value) and value >= 0 for value in (teacher_cap, stage_cap)
        )
        or teacher_cap + stage_cap > 300
    ):
        raise RunnerGateError(
            "complete teacher-dose plus worst-case Stage-A workload exceeds "
            "the aggregate $300 calibration cap"
        )
    return CalibrationAllocation(
        teacher.tokens,
        stage_tokens,
        teacher_cap,
        stage_cap,
    )


def require_remaining_budget(
    budget: CalibrationBudget,
    ledger: Any,
    *,
    prior_billed_usd: float,
) -> dict[str, Any]:
    """Reject before a call unless prior floors plus the whole workload fit."""

    candidate = ledger.committed.plus(budget.tokens)
    token_fit = all(
        getattr(candidate, name) <= getattr(ledger.limits, name)
        for name in ("prefill", "sample", "train")
    )
    spend_floor = max(ledger.committed_cost_usd, prior_billed_usd)
    dollar_fit = spend_floor + budget.upper_bound_usd <= ledger.authorized_usd
    if not token_fit or not dollar_fit:
        raise RunnerGateError(
            "conservative prior spend plus the complete remaining calibration "
            "workload exceeds its preflight-allocated action cap"
        )
    return {
        "schema_version": "duraseed-calibration-budget-preflight-v1",
        "prior_reservation_tokens": ledger.committed,
        "prior_reservation_cost_usd": ledger.committed_cost_usd,
        "prior_reconciled_billed_usd": prior_billed_usd,
        "remaining_token_upper_bound": budget.tokens,
        "remaining_fixed_storage_upper_bound_usd": budget.fixed_storage_usd,
        "remaining_cost_upper_bound_usd": budget.upper_bound_usd,
        "total_cost_upper_bound_usd": spend_floor + budget.upper_bound_usd,
        "action_cap_usd": ledger.authorized_usd,
    }


def persist_budget_preflight(root: Any, value: dict[str, Any]) -> str:
    """Persist an idempotent bound keyed by its canonical payload."""

    payload = canonical_json_bytes(value)
    digest = sha256_bytes(payload)
    path = root / f"budget-preflight-{digest.removeprefix('sha256:')}.json"
    if path.exists() and path.read_bytes() != payload:
        raise RunnerGateError("calibration budget preflight hash collision")
    atomic_write_bytes(path, payload)
    return digest


__all__ = [
    "CalibrationAllocation",
    "CalibrationBudget",
    "calibration_allocation",
    "persist_budget_preflight",
    "require_remaining_budget",
    "stage_a_budget",
    "teacher_dose_budget",
]
