"""Terminal B-G update-health outcomes, separate from remote interruptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from duraseed.runners.remote_journal import RemoteJournal
from duraseed.training_metric_errors import NonFiniteTrainingMetricError
from duraseed.training.stage_a_update_health import (
    StageAUpdateHealthFailure,
    StageAUpdateHealthFailureEvidence,
    write_stage_a_update_health_failure,
)


def update_health_failure(
    output: Path,
    branch: Any,
    step: int,
    *,
    reason: Literal["zero_mixed_group", "nonfinite_training_metric"],
    mixed: int,
    all_zero: int,
    all_one: int,
    optimizer_update_completed: bool,
    metric_name: str | None = None,
) -> StageAUpdateHealthFailure:
    evidence = StageAUpdateHealthFailureEvidence(
        "B-G",
        branch.learning_rate,
        step,
        "screen" if step <= 10 else "continuation",
        reason,
        step - 1,
        optimizer_update_completed,
        mixed + all_zero + all_one,
        8 * (mixed + all_zero + all_one),
        mixed,
        all_zero,
        all_one,
        metric_name,
    )
    write_stage_a_update_health_failure(output, evidence)
    return StageAUpdateHealthFailure(evidence)


async def apply_grouped_update_or_fail(
    inputs: Any,
    branch: Any,
    step: int,
    output: Path,
    journal: RemoteJournal,
    datums: list[Any],
    update: Any,
    *,
    mixed: int,
    all_zero: int,
    all_one: int,
) -> dict[str, float]:
    journal.begin(
        "stage-a-rl-update",
        {"learning_rate": branch.learning_rate, "step": step},
        {
            "prefill_tokens": 0,
            "sample_tokens": 0,
            "train_tokens": sum(int(row.model_input.length) for row in datums),
        },
    )
    try:
        return await update(
            branch.runtime,
            datums,
            loss_fn="importance_sampling",
            learning_rate=branch.learning_rate,
            ledger=inputs.stage_a_ledger,
        )
    except NonFiniteTrainingMetricError as error:
        failure = update_health_failure(
            output,
            branch,
            step,
            reason="nonfinite_training_metric",
            mixed=mixed,
            all_zero=all_zero,
            all_one=all_one,
            optimizer_update_completed=True,
            metric_name=error.metric_name,
        )
        journal.complete(
            {
                "operation": "stage-a-rl-update",
                "step": step,
                "health_failure": failure.evidence.reason,
                "metric_name": error.metric_name,
            }
        )
        raise failure from error


__all__ = ["apply_grouped_update_or_fail", "update_health_failure"]
