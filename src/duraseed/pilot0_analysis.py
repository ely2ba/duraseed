"""Primary validation summaries for the fixed-budget Pilot-0 paths."""

from __future__ import annotations

from dataclasses import dataclass

from duraseed.evaluation.analysis import (
    BinomialObservation,
    NormalizedAUC,
    equal_item_posterior_mean,
    normalized_stage_b_auc,
    posterior_soft_cover,
    posterior_mean_gain,
    stage_a_to_b_cover_change,
    targeted_minus_sentinel_specificity,
)
from duraseed.pilot0_contract import STAGE_B_GRID
from duraseed.runners import RunnerGateError


@dataclass(frozen=True, slots=True)
class Pilot0MethodSummary:
    seed: int
    method: str
    targeted_acquisition_gain: float
    sentinel_acquisition_gain: float
    acquisition_specificity: float
    pre_b_targeted_score: float
    fixed_budget_targeted_retention: float
    fixed_budget_targeted_change: float
    pre_b_sentinel_score: float
    fixed_budget_sentinel_retention: float
    fixed_budget_sentinel_change: float
    maps_auc: NormalizedAUC
    targeted_monitor_retention_absolute_auc: float
    sentinel_monitor_retention_absolute_auc: float
    maps_scores: tuple[float, ...]
    targeted_retention_curve: tuple[float, ...]
    sentinel_retention_curve: tuple[float, ...]
    specificity_curve: tuple[float, ...]
    targeted_cover_curve: tuple[dict[str, float], ...]
    sentinel_cover_curve: tuple[dict[str, float], ...]
    matched_a_selection: str


def _items(result: dict, role: str | None = None) -> dict[str, BinomialObservation]:
    rows = result.get("item_counts")
    if not isinstance(rows, list) or not rows:
        raise RunnerGateError("Pilot-0 analysis requires item-level validation counts")
    values = {}
    for row in rows:
        if role is not None and row.get("panel_role") != role:
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or task_id in values:
            raise RunnerGateError("Pilot-0 analysis has duplicate/malformed task IDs")
        values[task_id] = BinomialObservation(
            int(row.get("successes", -1)), int(row.get("trials", -1))
        )
    if not values:
        raise RunnerGateError(f"Pilot-0 analysis has no {role or 'requested'} items")
    return values


def _paired(
    origin: dict, current: dict, role: str
) -> tuple[tuple[BinomialObservation, ...], tuple[BinomialObservation, ...]]:
    before = _items(origin, role)
    after = _items(current, role)
    if set(before) != set(after):
        raise RunnerGateError(
            "Pilot-0 retention changed the validation item population"
        )
    task_ids = tuple(sorted(before))
    paired_before = tuple(before[task_id] for task_id in task_ids)
    paired_after = tuple(after[task_id] for task_id in task_ids)
    if any(
        origin.trials != current.trials
        for origin, current in zip(paired_before, paired_after, strict=True)
    ):
        raise RunnerGateError("Pilot-0 paired evidence changed its sample budget")
    return paired_before, paired_after


def monitor_retention_summary(stage_b_retention: tuple[dict, ...]) -> dict:
    """Absolute AUC on the common low-cost a_monitor population."""

    if len(stage_b_retention) != len(STAGE_B_GRID):
        raise RunnerGateError("monitor retention AUC is missing a Stage-B point")
    reference = stage_b_retention[0]
    targeted = tuple(
        equal_item_posterior_mean(_paired(reference, result, "targeted")[1])
        for result in stage_b_retention
    )
    sentinel = tuple(
        equal_item_posterior_mean(_paired(reference, result, "sentinel")[1])
        for result in stage_b_retention
    )
    return {
        "targeted_monitor_retention_curve": targeted,
        "sentinel_monitor_retention_curve": sentinel,
        "targeted_monitor_retention_absolute_auc": normalized_stage_b_auc(
            STAGE_B_GRID, targeted
        ).absolute_auc,
        "sentinel_monitor_retention_absolute_auc": normalized_stage_b_auc(
            STAGE_B_GRID, sentinel
        ).absolute_auc,
    }


def summarize_method(
    *,
    seed: int,
    method: str,
    m0_validation: dict,
    m0_monitor: dict,
    stage_a_final: dict,
    stage_b_maps: tuple[dict, ...],
    stage_b_retention: tuple[dict, ...],
    stage_b_final_retention: dict,
    cover_thresholds: tuple[float, ...] = (
        0.01,
        0.02,
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
    ),
) -> Pilot0MethodSummary:
    """Compute endpoints only from complete validation evidence."""

    if len(stage_b_maps) != len(STAGE_B_GRID) or len(stage_b_retention) != len(
        STAGE_B_GRID
    ):
        raise RunnerGateError("Pilot-0 frontier is missing an evaluation point")
    target_origin, target_a = _paired(m0_validation, stage_a_final, "targeted")
    sentinel_origin, sentinel_a = _paired(m0_validation, stage_a_final, "sentinel")
    target_pre_b, target_post_b = _paired(
        stage_a_final, stage_b_final_retention, "targeted"
    )
    sentinel_pre_b, sentinel_post_b = _paired(
        stage_a_final, stage_b_final_retention, "sentinel"
    )
    maps_reference = stage_b_maps[0]
    maps_scores = tuple(
        equal_item_posterior_mean(_paired(maps_reference, result, "stage-b")[1])
        for result in stage_b_maps
    )
    target_retention = tuple(
        equal_item_posterior_mean(_paired(m0_monitor, result, "targeted")[1])
        for result in stage_b_retention
    )
    sentinel_retention = tuple(
        equal_item_posterior_mean(_paired(m0_monitor, result, "sentinel")[1])
        for result in stage_b_retention
    )
    m0_target = equal_item_posterior_mean(_items(m0_monitor, "targeted").values())
    m0_sentinel = equal_item_posterior_mean(_items(m0_monitor, "sentinel").values())
    specificity_curve = tuple(
        (target - m0_target) - (sentinel - m0_sentinel)
        for target, sentinel in zip(target_retention, sentinel_retention, strict=True)
    )
    target_gain = posterior_mean_gain(target_origin, target_a)
    sentinel_gain = posterior_mean_gain(sentinel_origin, sentinel_a)
    pre_target = equal_item_posterior_mean(target_pre_b)
    post_target = equal_item_posterior_mean(target_post_b)
    pre_sentinel = equal_item_posterior_mean(sentinel_pre_b)
    post_sentinel = equal_item_posterior_mean(sentinel_post_b)
    if (
        cover_thresholds != tuple(sorted(set(cover_thresholds)))
        or 0.25 not in cover_thresholds
    ):
        raise RunnerGateError("Pilot-0 Cover curve omits its frozen thresholds")

    def cover(before, after):  # type: ignore[no-untyped-def]
        return tuple(
            {
                "tau": tau,
                "stage_a": posterior_soft_cover(before, tau),
                "stage_b": posterior_soft_cover(after, tau),
                "change": stage_a_to_b_cover_change(before, after, tau),
            }
            for tau in cover_thresholds
        )

    return Pilot0MethodSummary(
        seed=seed,
        method=method,
        targeted_acquisition_gain=target_gain,
        sentinel_acquisition_gain=sentinel_gain,
        acquisition_specificity=targeted_minus_sentinel_specificity(
            target_origin, target_a, sentinel_origin, sentinel_a
        ),
        pre_b_targeted_score=pre_target,
        fixed_budget_targeted_retention=post_target,
        fixed_budget_targeted_change=post_target - pre_target,
        pre_b_sentinel_score=pre_sentinel,
        fixed_budget_sentinel_retention=post_sentinel,
        fixed_budget_sentinel_change=post_sentinel - pre_sentinel,
        maps_auc=normalized_stage_b_auc(STAGE_B_GRID, maps_scores),
        targeted_monitor_retention_absolute_auc=normalized_stage_b_auc(
            STAGE_B_GRID, target_retention
        ).absolute_auc,
        sentinel_monitor_retention_absolute_auc=normalized_stage_b_auc(
            STAGE_B_GRID, sentinel_retention
        ).absolute_auc,
        maps_scores=maps_scores,
        targeted_retention_curve=target_retention,
        sentinel_retention_curve=sentinel_retention,
        specificity_curve=specificity_curve,
        targeted_cover_curve=cover(target_pre_b, target_post_b),
        sentinel_cover_curve=cover(sentinel_pre_b, sentinel_post_b),
        matched_a_selection="pending_post_pilot_target_freeze",
    )


def paired_primary_aggregate(
    summaries: tuple[Pilot0MethodSummary, ...],
) -> dict:
    """Return the declared paired B-S minus B-G Pilot-0 contrasts."""

    by_seed = {(row.seed, row.method): row for row in summaries}
    seeds = tuple(sorted({row.seed for row in summaries}))
    if len(summaries) != 2 * len(seeds) or any(
        (seed, method) not in by_seed for seed in seeds for method in ("B-S", "B-G")
    ):
        raise RunnerGateError("Pilot-0 paired aggregate requires one B-S/B-G pair")
    differences = tuple(
        {
            "seed": seed,
            "raw_gain_stage_b_auc": (
                by_seed[seed, "B-S"].maps_auc.raw_gain_auc
                - by_seed[seed, "B-G"].maps_auc.raw_gain_auc
            ),
            "fixed_budget_targeted_retention": (
                by_seed[seed, "B-S"].fixed_budget_targeted_retention
                - by_seed[seed, "B-G"].fixed_budget_targeted_retention
            ),
            "targeted_monitor_retention_absolute_auc": (
                by_seed[seed, "B-S"].targeted_monitor_retention_absolute_auc
                - by_seed[seed, "B-G"].targeted_monitor_retention_absolute_auc
            ),
        }
        for seed in seeds
    )
    if len(differences) != 2:
        raise RunnerGateError("Pilot-0 primary aggregate requires exactly two seeds")
    return {
        "contrast": "B-S_minus_B-G",
        "paired_seed_differences": differences,
        "mean_raw_gain_stage_b_auc_difference": sum(
            row["raw_gain_stage_b_auc"] for row in differences
        )
        / 2,
        "mean_fixed_budget_targeted_retention_difference": sum(
            row["fixed_budget_targeted_retention"] for row in differences
        )
        / 2,
        "mean_targeted_monitor_retention_absolute_auc_difference": sum(
            row["targeted_monitor_retention_absolute_auc"] for row in differences
        )
        / 2,
        "paired_seed_count": 2,
        "degrees_of_freedom": 1,
        "limitation": "variance estimate has one degree of freedom",
    }


__all__ = [
    "Pilot0MethodSummary",
    "monitor_retention_summary",
    "paired_primary_aggregate",
    "summarize_method",
]
