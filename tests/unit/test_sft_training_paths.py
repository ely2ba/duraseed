from __future__ import annotations

from functools import lru_cache

import pytest
from pydantic import ValidationError

from duraseed.data.format import build_format_manifest
from duraseed.data.manifests import (
    DatasetManifest,
    MAPSTaskManifestRecord,
    TCESTaskManifestRecord,
    build_manifest,
    build_maps_record,
    task_semantic_hash,
)
from duraseed.provenance import derive_namespaced_seed
from duraseed.schemas import ExactRational, TCESTask
from duraseed.tasks.maps import MAPSGenerator, render_teacher_answer
from duraseed.tasks.tces import enumerate_task, render_prompt as render_tces_prompt
from duraseed.training.sft import (
    SourceKind,
    SourceRecordError,
    VerifiedSourceRecord,
    build_current_policy_verified_record,
    build_format_warmup_record,
    build_solver_teacher_record,
    build_stage_b_maps_record,
    build_teacher_dose_records,
)
from duraseed.training.reward import verify_task_completion


@lru_cache(maxsize=None)
def _tces_manifest(
    split: str,
    specifications: tuple[tuple[tuple[int, ...], int], ...],
) -> DatasetManifest:
    root_seed = 29
    generator_seed = derive_namespaced_seed(
        root_seed,
        "dataset.tces.split",
        split,
    )
    records: list[TCESTaskManifestRecord] = []
    for index, (operands, target) in enumerate(specifications):
        provisional = TCESTask(
            operands=operands,
            target=ExactRational(numerator=target),
            split=split,
        )
        task_id = task_semantic_hash(provisional)
        task = provisional.model_copy(update={"task_id": task_id})
        enumeration = enumerate_task(task)
        assert enumeration.shortest_depth is not None
        family_ids = tuple(sorted(enumeration.family_ids))
        records.append(
            TCESTaskManifestRecord(
                task_id=task_id,
                split=split,
                generator_version="1.0.0",
                generator_seed=generator_seed,
                item_index=index,
                accepted_attempt=0,
                prompt_template_id="tces_v1",
                content_hash=task_id,
                operands=tuple(sorted(task.operands)),
                target=task.target,
                allowed_ops=task.allowed_ops,
                constraints=task.constraints,
                intended_family=family_ids[0],
                valid_family_ids=family_ids,
                valid_family_count=len(family_ids),
                valid_expression_count=len(enumeration.expressions),
                minimum_depth=enumeration.shortest_depth,
            )
        )
    return build_manifest(
        name=f"tces-{split}-source-test",
        split=split,
        generator_version="1.0.0",
        root_seed=root_seed,
        records=records,
    )


def _record(manifest: DatasetManifest, index: int = 0) -> TCESTaskManifestRecord:
    record = manifest.records[index]
    assert isinstance(record, TCESTaskManifestRecord)
    return record


@lru_cache(maxsize=None)
def _maps_manifest() -> tuple[DatasetManifest, MAPSTaskManifestRecord, str]:
    root_seed = 101
    generator_seed = derive_namespaced_seed(
        root_seed,
        "dataset.maps.split",
        "b_train",
    )
    instance = MAPSGenerator(generator_seed).generate(0)
    record = build_maps_record(instance, split="b_train")
    manifest = build_manifest(
        name="maps-b-train-source-test",
        split="b_train",
        generator_version="1.0.0",
        root_seed=root_seed,
        records=[record],
    )
    return manifest, record, render_teacher_answer(instance)


def test_format_warmup_builder_keeps_only_task_agnostic_source_fields() -> None:
    source_manifest = build_format_manifest("format_train", 1)
    source_record = source_manifest.records[0]
    record = build_format_warmup_record(
        source_manifest=source_manifest,
        source_record=source_record,
    )

    assert record.source_kind is SourceKind.TASK_AGNOSTIC_FORMAT
    assert record.task_family == "format"
    assert record.source_split == "format_train"
    assert record.task_id == source_record.record_id
    assert record.source_manifest_id == source_manifest.manifest_id
    assert record.strategy_family_id is None
    assert record.exact_verification is None


def test_format_warmup_builder_rejects_reasoning_manifests_and_foreign_records() -> (
    None
):
    format_manifest = build_format_manifest("format_train", 1)
    tces_manifest = _tces_manifest("a_seed_train", (((2, 3), 5),))
    maps_manifest, maps_record, _ = _maps_manifest()

    with pytest.raises(TypeError, match="FormatManifest"):
        build_format_warmup_record(
            source_manifest=tces_manifest,
            source_record=_record(tces_manifest),
        )
    with pytest.raises(TypeError, match="FormatManifest"):
        build_format_warmup_record(
            source_manifest=maps_manifest,
            source_record=maps_record,
        )
    with pytest.raises(TypeError, match="FormatRecord"):
        build_format_warmup_record(
            source_manifest=format_manifest,
            source_record=_record(tces_manifest),
        )
    with pytest.raises(TypeError, match="FormatRecord"):
        build_format_warmup_record(
            source_manifest=format_manifest,
            source_record=maps_record,
        )

    foreign = build_format_manifest("format_eval", 1).records[0]
    with pytest.raises(SourceRecordError, match="not present"):
        build_format_warmup_record(
            source_manifest=format_manifest,
            source_record=foreign,
        )

    format_eval = build_format_manifest("format_eval", 1)
    with pytest.raises(SourceRecordError, match="require format_train"):
        build_format_warmup_record(
            source_manifest=format_eval,
            source_record=format_eval.records[0],
        )


def test_solver_teacher_builder_authenticates_manifest_and_exact_success() -> None:
    manifest = _tces_manifest("a_seed_train", (((2, 3), 5),))
    source = _record(manifest)
    built = build_solver_teacher_record(
        source_manifest=manifest,
        source_record=source,
        completion="<answer>2+3</answer>",
    )

    assert built.prompt_text == render_tces_prompt(source.to_task())
    assert built.verified_completion_text == "<answer>2+3</answer>"
    assert built.task_id == source.task_id
    assert built.task_family == "tces"
    assert built.source_split == "a_seed_train"
    assert built.source_kind is SourceKind.SOLVER_TEACHER
    assert built.source_manifest_id == manifest.manifest_id
    assert built.exact_verification is not None
    assert built.exact_verification.reward == 1.0
    assert built.strategy_family_id == built.exact_verification.strategy_family_id

    with pytest.raises(SourceRecordError, match="pass the authoritative"):
        build_solver_teacher_record(
            source_manifest=manifest,
            source_record=source,
            completion="<answer>3-2</answer>",
        )


def test_solver_teacher_builder_rejects_record_not_in_manifest() -> None:
    manifest = _tces_manifest("a_seed_train", (((2, 3), 5),))
    other = _tces_manifest("a_seed_train", (((4, 5), 9),))
    with pytest.raises(SourceRecordError, match="not present"):
        build_solver_teacher_record(
            source_manifest=manifest,
            source_record=_record(other),
            completion="<answer>4+5</answer>",
        )


def test_teacher_dose_is_exact_and_deterministic_per_selected_family() -> None:
    manifest = _tces_manifest(
        "a_seed_train",
        (
            ((2, 3), 5),
            ((4, 5), 9),
            ((6, 7), 13),
            ((2, 4), 8),
            ((2, 3), 6),
            ((3, 4), 12),
        ),
    )
    records = tuple(
        record
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
    )
    add_family = next(
        record.intended_family
        for record in records
        if record.target.as_fraction() == sum(record.operands)
    )
    multiply_family = next(
        record.intended_family
        for record in records
        if record.target.as_fraction() == record.operands[0] * record.operands[1]
    )
    completions = tuple(
        (
            record,
            (
                f"<answer>{record.operands[0]}+{record.operands[1]}</answer>"
                if record.intended_family == add_family
                else f"<answer>{record.operands[0]}*{record.operands[1]}</answer>"
            ),
        )
        for record in reversed(records)
    )

    dose = build_teacher_dose_records(
        source_manifest=manifest,
        solver_completions=completions,
        selected_families=(multiply_family, add_family),
        demonstrations_per_family=2,
    )
    repeated = build_teacher_dose_records(
        source_manifest=manifest,
        solver_completions=reversed(completions),
        selected_families=(add_family, multiply_family),
        demonstrations_per_family=2,
    )

    assert dose == repeated
    assert [record.strategy_family_id for record in dose] == [
        add_family,
        add_family,
        multiply_family,
        multiply_family,
    ]
    assert [record.task_id for record in dose] == [
        record.task_id
        for family in sorted((add_family, multiply_family))
        for record in sorted(
            (record for record in records if record.intended_family == family),
            key=lambda record: record.item_index,
        )[:2]
    ]


def test_teacher_dose_rejects_insufficient_verified_family_supply() -> None:
    manifest = _tces_manifest(
        "a_seed_train",
        (((2, 3), 5), ((2, 4), 8)),
    )
    records = tuple(
        record
        for record in manifest.records
        if isinstance(record, TCESTaskManifestRecord)
    )
    completions = tuple(
        (
            record,
            (
                f"<answer>{record.operands[0]}+{record.operands[1]}</answer>"
                if record.target.as_fraction() == sum(record.operands)
                else f"<answer>{record.operands[0]}*{record.operands[1]}</answer>"
            ),
        )
        for record in records
    )

    with pytest.raises(SourceRecordError, match="insufficient verified solver"):
        build_teacher_dose_records(
            source_manifest=manifest,
            solver_completions=completions,
            selected_families=tuple(
                sorted({record.intended_family for record in records})
            ),
            demonstrations_per_family=2,
        )


def test_teacher_dose_requires_unique_tasks_in_their_manifested_family() -> None:
    manifest = _tces_manifest("a_seed_train", (((2, 3, 4), 9),))
    record = _record(manifest)
    enumeration = enumerate_task(record.to_task())
    alternate = next(
        expression
        for expression in enumeration.expressions
        if expression.family_id != record.intended_family
    )

    with pytest.raises(SourceRecordError, match="intended_family"):
        build_teacher_dose_records(
            source_manifest=manifest,
            solver_completions=(
                (record, f"<answer>{alternate.canonical_expression}</answer>"),
            ),
            selected_families=(alternate.family_id,),
            demonstrations_per_family=1,
        )

    intended = next(
        expression
        for expression in enumeration.expressions
        if expression.family_id == record.intended_family
    )
    completion = f"<answer>{intended.canonical_expression}</answer>"
    with pytest.raises(SourceRecordError, match="insufficient verified solver"):
        build_teacher_dose_records(
            source_manifest=manifest,
            solver_completions=((record, completion), (record, completion)),
            selected_families=(record.intended_family,),
            demonstrations_per_family=2,
        )


def test_current_policy_builder_requires_verified_a_rl_train_output() -> None:
    manifest = _tces_manifest("a_rl_train", (((2, 3), 5),))
    built = build_current_policy_verified_record(
        source_manifest=manifest,
        source_record=_record(manifest),
        completion="<answer>3+2</answer>",
    )
    assert built.source_kind is SourceKind.CURRENT_POLICY_VERIFIED
    assert built.source_split == "a_rl_train"
    assert built.exact_verification is not None
    assert built.exact_verification.canonical_expression == "(2+3)"

    seed_manifest = _tces_manifest("a_seed_train", (((2, 3), 5),))
    with pytest.raises(SourceRecordError, match="a_rl_train"):
        build_current_policy_verified_record(
            source_manifest=seed_manifest,
            source_record=_record(seed_manifest),
            completion="<answer>2+3</answer>",
        )


def test_stage_b_builder_requires_manifested_shortest_maps_completion() -> None:
    manifest, source, completion = _maps_manifest()
    built = build_stage_b_maps_record(
        source_manifest=manifest,
        source_record=source,
        completion=completion,
    )
    assert built.task_family == "maps"
    assert built.source_split == "b_train"
    assert built.source_kind is SourceKind.SOLVER_TEACHER
    assert built.exact_verification is not None
    assert built.exact_verification.canonical_expression in source.shortest_programs

    with pytest.raises(SourceRecordError):
        build_stage_b_maps_record(
            source_manifest=manifest,
            source_record=source,
            completion="<answer>NEG</answer>",
        )


def test_source_record_cannot_claim_failed_or_mismatched_verification() -> None:
    manifest = _tces_manifest("a_rl_train", (((2, 3), 5),))
    source = _record(manifest)
    task = source.to_task()
    failed = verify_task_completion("<answer>3-2</answer>", task)
    payload = build_current_policy_verified_record(
        source_manifest=manifest,
        source_record=source,
        completion="<answer>2+3</answer>",
    ).model_dump(mode="python")

    payload["exact_verification"] = failed
    with pytest.raises(ValidationError, match="pass the exact verifier"):
        VerifiedSourceRecord.model_validate(payload, strict=True)

    payload["exact_verification"] = build_current_policy_verified_record(
        source_manifest=manifest,
        source_record=source,
        completion="<answer>2+3</answer>",
    ).exact_verification
    payload["strategy_family_id"] = "wrong-family"
    with pytest.raises(ValidationError, match="strategy family"):
        VerifiedSourceRecord.model_validate(payload, strict=True)
