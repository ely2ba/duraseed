"""Offline REPLAY-V1 full-cap token, storage, and protected-balance preflight."""

from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
from typing import Any

from duraseed.runtime.ledger import TokenBudget

RATES = {
    "prefill": Decimal("0.66"),
    "sample": Decimal("1.995"),
    "train": Decimal("1.463"),
    "storage_gb_month": Decimal("0.10"),
}
PROTECTED_USD = Decimal("1343.74")
PACKAGE_MAX_USD = Decimal("2400")
STAGE_A_GRID = (*range(10, 291, 10), 294)
STAGE_B_GRID = (0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480)


def money(value: Any) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("USD amounts must be finite and nonnegative")
    return result


def token_cost(tokens: TokenBudget) -> Decimal:
    return (
        sum(
            (
                getattr(tokens, key) * RATES[key]
                for key in ("prefill", "sample", "train")
            ),
            Decimal(0),
        )
        / 1_000_000
    )


def storage_cost(
    *,
    pairs: int,
    state_bytes: int,
    sampler_bytes: int,
    ttl_seconds: int,
    backup_seconds: int = 0,
) -> Decimal:
    """Charge every retained byte for its entire TTL, including backup liability."""
    values = (pairs, state_bytes, sampler_bytes, ttl_seconds, backup_seconds)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(
            "storage counts, sizes and durations must be nonnegative integers"
        )
    if not state_bytes or not sampler_bytes or not ttl_seconds:
        raise ValueError("finite nonzero checkpoint sizes and TTL are required")
    return (
        Decimal(pairs * (state_bytes + sampler_bytes))
        / 1_000_000_000
        * Decimal(ttl_seconds + backup_seconds)
        / (30 * 86400)
        * RATES["storage_gb_month"]
    )


def _lengths(values: list[int], count: int, label: str) -> list[int]:
    if len(values) != count or any(type(x) is not int or x <= 0 for x in values):
        raise ValueError(f"{label} requires exactly {count} positive token lengths")
    return values


def build_preflight(
    blocks: dict[int, dict[str, Any]],
    *,
    storage: dict[str, Any],
    current_balance_usd: str | None = None,
    other_committed_usd: str | None = None,
) -> dict[str, Any]:
    """Actual rendered lengths; four eligible Stage-B arms; no origin duplication."""
    if set(blocks) != {11, 29}:
        raise ValueError("REPLAY-V1 requires blocks 11 and 29")
    components = []
    total = TokenBudget(0, 0, 0)

    def append(name: str, seed: int, tokens: TokenBudget, **counts: int) -> None:
        nonlocal total
        total = total.plus(tokens)
        components.append(
            {
                "component": name,
                "seed": seed,
                **counts,
                "prefill": tokens.prefill,
                "sample": tokens.sample,
                "train": tokens.train,
                "cost_usd": str(token_cost(tokens)),
            }
        )

    for seed, block in sorted(blocks.items()):
        for arm in ("R-S", "R-P"):
            train = _lengths(block["per_arm_per_update_train_tokens"][arm], 294, arm)
            append(
                f"stage_a_{arm}_train",
                seed,
                TokenBudget(0, 0, sum(train)),
                arms=1,
                updates=294,
                batch=32,
                presentations=9408,
            )
        train_b = _lengths(block["stage_b_per_update_train_tokens"], 480, "Stage B")
        append(
            "stage_b_train",
            seed,
            TokenBudget(0, 0, 2 * sum(train_b)),
            arms=2,
            updates=480,
            batch=32,
            presentations=30720,
        )
        # Pre-B F3 = Stage-B origin, counted once; all 12 nominees are additional.
        evaluations = (
            ("stage_a_cadence", "a_cadence", 192, 1, 30, 4096),
            ("candidate_assessments", "targeted_a_validation", 256, 16, 3, 4096),
            ("pre_b_and_final_validation", "a_validation", 512, 16, 2, 4096),
            ("stage_b_monitor_including_pre_b", "a_monitor", 384, 4, 11, 4096),
            ("stage_b_maps_including_pre_b", "b_validation", 512, 16, 11, 128),
        )
        for name, panel, items, draws, points, cap in evaluations:
            lengths = _lengths(block["prompt_lengths"][panel], items, panel)
            tokens = TokenBudget(
                sum(lengths) * draws * 2 * points, items * draws * 2 * points * cap, 0
            )
            append(
                name,
                seed,
                tokens,
                items=items,
                draws=draws,
                arms=2,
                points=points,
                cap=cap,
            )
    required_pairs = 4 * len(STAGE_A_GRID) + 4 * (len(STAGE_B_GRID) - 1)
    if storage["pairs"] != required_pairs:
        raise ValueError(
            f"all {required_pairs} Stage-A/Stage-B checkpoint pairs must be reserved"
        )
    storage_usd = storage_cost(
        **{
            key: storage[key]
            for key in (
                "pairs",
                "state_bytes",
                "sampler_bytes",
                "ttl_seconds",
                "backup_seconds",
            )
        }
    )
    main = token_cost(total) + storage_usd
    recovery = main * Decimal("0.10")
    package = Decimal("20") + main + recovery
    ceiling = package.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    blockers = []
    if ceiling > PACKAGE_MAX_USD:
        blockers.append("BUDGET_BLOCKED")
    if storage.get("size_bound_verified") is not True:
        blockers.append("CHECKPOINT_SIZE_BOUND_UNVERIFIED")
    if storage.get("backup_policy_verified") is not True:
        blockers.append("BACKUP_LIABILITY_UNVERIFIED")
    if current_balance_usd is None or other_committed_usd is None:
        blockers.append("LIVE_BALANCE_AND_OTHER_COMMITMENTS_UNVERIFIED")
    elif (
        money(current_balance_usd) - money(other_committed_usd) - ceiling
        < PROTECTED_USD
    ):
        blockers.append("PROTECTED_BALANCE_BLOCKED")
    return {
        "schema_version": "duraseed-replay-v1-preflight-v1",
        "status": "BUDGET_BLOCKED"
        if "BUDGET_BLOCKED" in blockers
        else "preparation_blocked"
        if blockers
        else "budget_fits_not_authorized",
        "components": components,
        "token_budget": {
            key: getattr(total, key) for key in ("prefill", "sample", "train")
        },
        "prices_per_million_usd": {
            key: str(RATES[key]) for key in ("prefill", "sample", "train")
        },
        "cache_discount_assumed": False,
        "storage": {**storage, "cost_usd": str(storage_usd)},
        "main_usd": str(main),
        "smoke_reservation_usd": "20",
        "recovery_reservation_usd": str(recovery),
        "package_usd": str(package),
        "approval_ceiling_usd": str(ceiling),
        "package_max_usd": "2400",
        "protected_reserve_usd": str(PROTECTED_USD),
        "current_balance_usd": current_balance_usd,
        "other_committed_usd": other_committed_usd,
        "blockers": blockers,
    }
