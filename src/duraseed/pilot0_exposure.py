"""Stage-A acquisition exposure reconstructed from completed Pilot-0 segments."""

from __future__ import annotations

import json
from pathlib import Path

from duraseed.evaluation.analysis import BinomialObservation, equal_item_posterior_mean
from duraseed.pilot0_evidence import read_evaluation
from duraseed.run_records import GenerationRecord
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_remote import read_segment
from duraseed.runtime.billing import PriceSnapshot, UsageQuantities


def _monitor_scores(directory: Path) -> dict[str, float]:
    monitor = read_evaluation(directory)
    if monitor is None:
        raise RunnerGateError("Stage-A exposure requires its monitor evidence")
    scores = {}
    for role in ("targeted", "sentinel"):
        try:
            observations = tuple(
                BinomialObservation(int(row["successes"]), int(row["trials"]))
                for row in monitor["item_counts"]
                if row["panel_role"] == role
            )
            scores[f"{role}_monitor_posterior_mean"] = equal_item_posterior_mean(
                observations
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RunnerGateError("Stage-A monitor evidence is malformed") from error
    return scores


def _segment(
    directory: Path, *, seed: int, method: str, start: int, stop: int
) -> tuple[dict[str, int], int, dict[str, float]]:
    completed = read_segment(
        directory,
        {
            "kind": "stage-a",
            "seed": seed,
            "method": method,
            "start": start,
            "stop": stop,
        },
    )
    if completed is None:
        raise RunnerGateError("Stage-A exposure requires a completed segment")
    try:
        metrics = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        generation_path = directory / "generations.jsonl"
        generations = (
            [
                GenerationRecord.model_validate_json(line)
                for line in generation_path.read_text().splitlines()
            ]
            if generation_path.exists()
            else []
        )
        train_values = [float(row["metrics"]["local.train_tokens"]) for row in metrics]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerGateError("Stage-A exposure evidence is unreadable") from error
    if (
        tuple(row.get("training_step") for row in metrics)
        != tuple(range(start + 1, stop + 1))
        or any(row.get("method") != method for row in metrics)
        or any(row.get("phase") != "stage_a" for row in metrics)
        or any(not value.is_integer() or value < 0 for value in train_values)
        or (method == "B-S" and generations)
        or any(
            row.method != method
            or row.seed != seed
            or not start < row.training_step <= stop
            for row in generations
        )
    ):
        raise RunnerGateError("Stage-A exposure rows changed coordinates")
    return (
        {
            "prefill": sum(row.prompt_tokens for row in generations),
            "sample": sum(row.sampled_tokens for row in generations),
            "train": sum(int(value) for value in train_values),
        },
        len(generations),
        _monitor_scores(directory / "a-monitor"),
    )


def stage_a_exposure_report(
    root: Path, seeds: tuple[int, ...], prices: PriceSnapshot
) -> dict:
    """Report actual update/rollout tokens, separate from evaluation and storage."""

    cells = []
    for seed in seeds:
        origin = root / f"seed-{seed}" / "boundary-origin"
        if read_segment(origin, {"kind": "boundary-origin", "seed": seed}) is None:
            raise RunnerGateError("Stage-A exposure requires its shared origin")
        baseline_scores = _monitor_scores(origin / "a-monitor")
        for method in ("B-S", "B-G"):
            cumulative = {"prefill": 0, "sample": 0, "train": 0}
            cumulative_rollouts = 0
            points = [
                {
                    "stage_a_step": 0,
                    "incremental_optimizer_updates": 0,
                    "cumulative_optimizer_updates": 0,
                    "incremental_tokens": dict(cumulative),
                    "cumulative_tokens": dict(cumulative),
                    "cumulative_rollout_completions": 0,
                    "cumulative_token_cost_usd": 0.0,
                    **baseline_scores,
                }
            ]
            for start, stop in zip((0, 10, 25), (10, 25, 50), strict=True):
                directory = root / f"seed-{seed}" / method / f"steps-{start}-{stop}"
                incremental, rollouts, monitor_scores = _segment(
                    directory,
                    seed=seed,
                    method=method,
                    start=start,
                    stop=stop,
                )
                cumulative = {
                    name: cumulative[name] + incremental[name] for name in cumulative
                }
                cumulative_rollouts += rollouts
                cost = prices.cost(
                    UsageQuantities(
                        prefill_tokens=cumulative["prefill"],
                        sample_tokens=cumulative["sample"],
                        train_tokens=cumulative["train"],
                    )
                )
                points.append(
                    {
                        "stage_a_step": stop,
                        "incremental_optimizer_updates": stop - start,
                        "cumulative_optimizer_updates": stop,
                        "incremental_tokens": incremental,
                        "cumulative_tokens": dict(cumulative),
                        "cumulative_rollout_completions": cumulative_rollouts,
                        "cumulative_token_cost_usd": cost,
                        **monitor_scores,
                    }
                )
            cells.append({"seed": seed, "method": method, "points": points})
    return {
        "schema_version": "duraseed-stage-a-exposure-v1",
        "scope": (
            "training updates and B-G acquisition rollouts; excludes evaluation "
            "and checkpoint storage"
        ),
        "cost_basis": (
            "exact local token counts priced at the pinned snapshot; final billing "
            "reconciliation may adjust cached prefill"
        ),
        "price_snapshot_id": prices.snapshot_id,
        "cells": cells,
    }


__all__ = ["stage_a_exposure_report"]
