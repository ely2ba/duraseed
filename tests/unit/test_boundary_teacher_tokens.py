from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from duraseed.boundary_teacher_tokens import (
    BoundaryTeacherTokenError,
    teacher_trace_token_counts,
)
from duraseed.data.boundary import BoundaryFamilySummary, BoundaryItemSummary
from duraseed.data.manifests import build_manifest, build_tces_record
from duraseed.data.splits import TCESSplitBuilder, derive_tces_split_seed
from duraseed.tasks.tces import TCESFamilyGenerator, TCESGeneratorConfig


class _Weights:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class _Renderer:
    def __init__(self, weights=(0.0, 1.0, 1.0)):
        self.weights = weights
        self.calls = []

    def build_supervised_example(self, messages, *, train_on_what):
        self.calls.append((messages, train_on_what))
        return SimpleNamespace(length=len(self.weights)), _Weights(self.weights)


def _source():
    config = TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=256,
    )
    template = TCESSplitBuilder(5, config).lazy_split("a_candidate", size=1)[0]
    variants = TCESFamilyGenerator(
        derive_tces_split_seed(5, "a_candidate"),
        template,
        replace(config, split="a_candidate"),
    ).generate_many(2, start_index=16)
    records = tuple(build_tces_record(value) for value in variants)
    build_manifest(
        name="unused",
        split="a_candidate",
        generator_version="1.0.0",
        root_seed=5,
        records=records,
        task_family="tces",
    )
    family_id = template.intended_family
    items = tuple(
        BoundaryItemSummary(
            record.task_id,
            record.item_index,
            family_id,
            16,
            2,
            0.1,
            0.5,
            1.0,
            1.0,
            0.0,
        )
        for record in records
    )
    summary = BoundaryFamilySummary(
        family_id,
        "tinker://m0/sampler",
        8,
        items,
        0.1,
        0.1,
        0.1,
        0.0,
        0.5,
        0.5,
        2,
        4,
        0.5,
        1.0,
        1.0,
        0.0,
        (family_id,),
    )
    return summary, records


def test_archived_teacher_mask_counts_only_positive_loss_tokens() -> None:
    summary, records = _source()
    renderer = _Renderer()

    counts = teacher_trace_token_counts(
        summary, records, renderer, "last_assistant_message"
    )

    assert counts == {record.task_id: 2 for record in records}
    assert len(renderer.calls) == 2
    assert all(call[1] == "last_assistant_message" for call in renderer.calls)
    assert all(call[0][0]["role"] == "user" for call in renderer.calls)
    assert all(call[0][1]["role"] == "assistant" for call in renderer.calls)


def test_archived_teacher_mask_rejects_renderer_length_mismatch() -> None:
    summary, records = _source()
    renderer = _Renderer()
    renderer.build_supervised_example = lambda *_args, **_kwargs: (
        SimpleNamespace(length=2),
        _Weights([1.0]),
    )

    with pytest.raises(BoundaryTeacherTokenError, match="mask differs"):
        teacher_trace_token_counts(summary, records, renderer, "last_assistant_message")
