"""Exact archived supervised-token measurement for boundary panel matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from duraseed.data.boundary import BoundaryFamilySummary
from duraseed.data.manifests import TCESTaskManifestRecord
from duraseed.runtime import MODEL_ID, RENDERER_NAME
from duraseed.tasks.tces import (
    enumerate_task,
    generate_teacher_trace,
    render_prompt,
    verify_completion,
)


class BoundaryTeacherTokenError(ValueError):
    """An authenticated family cannot be measured with the archived renderer."""


def load_archived_renderer() -> tuple[Any, Any]:
    """Load the pinned local renderer with the v0 call signature."""

    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    tokenizer = get_tokenizer(MODEL_ID)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    return renderer, TrainOnWhat.LAST_ASSISTANT_MESSAGE


def teacher_trace_token_counts(
    summary: BoundaryFamilySummary,
    records: Sequence[TCESTaskManifestRecord],
    renderer: Any,
    train_on_what: Any,
) -> dict[str, int]:
    """Count positive-loss tokens using the archived v0 rendering operation."""

    counts: dict[str, int] = {}
    for record in records:
        task = record.to_task()
        enumeration = enumerate_task(task)
        if (
            not enumeration.complete
            or enumeration.family_ids != record.valid_family_ids
            or len(enumeration.expressions) != record.valid_expression_count
        ):
            raise BoundaryTeacherTokenError(
                "panel candidate exact enumeration differs from its manifest"
            )
        representative = enumeration.family_representatives.get(
            summary.intended_family_id
        )
        if representative is None:
            raise BoundaryTeacherTokenError(
                "panel family is absent from an authenticated task"
            )
        trace = generate_teacher_trace(representative)
        verified = verify_completion(trace, task)
        if (
            verified.reward != 1.0
            or verified.strategy_family_id != summary.intended_family_id
        ):
            raise BoundaryTeacherTokenError(
                "panel teacher trace failed exact verification"
            )
        model_input, weights = renderer.build_supervised_example(
            [
                {"role": "user", "content": render_prompt(task)},
                {"role": "assistant", "content": trace},
            ],
            train_on_what=train_on_what,
        )
        weight_values = tuple(float(value) for value in weights.tolist())
        if len(weight_values) != int(model_input.length):
            raise BoundaryTeacherTokenError(
                "panel teacher trace mask differs from its rendered input"
            )
        target_token_count = sum(value > 0 for value in weight_values)
        if target_token_count < 1:
            raise BoundaryTeacherTokenError(
                "panel teacher trace has no supervised target tokens"
            )
        counts[record.task_id] = target_token_count
    return counts


def archived_token_counter():
    """Return one counter sharing the exact pinned renderer instance."""

    renderer, train_on_what = load_archived_renderer()

    def count(
        summary: BoundaryFamilySummary,
        records: tuple[TCESTaskManifestRecord, ...],
    ) -> Mapping[str, int]:
        return teacher_trace_token_counts(summary, records, renderer, train_on_what)

    return count


__all__ = [
    "BoundaryTeacherTokenError",
    "archived_token_counter",
    "load_archived_renderer",
    "teacher_trace_token_counts",
]
