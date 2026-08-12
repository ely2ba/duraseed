from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from duraseed.data.leakage import (
    FamilyDisjointnessRule,
    LeakageAuditError,
    LeakageCode,
    audit_leakage,
    collect_family_ids,
)
from duraseed.data.splits import (
    EVALUATION_SPLITS,
    FINAL_TEST_SPLITS,
    TEACHER_SPLITS,
    TCESSplitBuilder,
    TCESOODProfile,
    TCES_SPLIT_NAMES,
    TCES_SPLIT_SIZES,
    derive_tces_split_seed,
    exact_valid_family_count,
    is_multi_family,
    is_single_family,
    tces_numeric_key,
)
from duraseed.provenance import MAX_ROOT_SEED, derive_namespaced_seed
from duraseed.tasks.tces import TCESGeneratorConfig


EXPECTED_SPLIT_SIZES = {
    "a_candidate": 8_192,
    "a_seed_train": 4_096,
    "a_seed_gate": 768,
    "a_rl_train": 16_384,
    "a_monitor": 384,
    "a_validation": 512,
    "a_test_single": 512,
    "a_test_multi": 512,
    "a_ood": 512,
}


def _fast_config() -> TCESGeneratorConfig:
    return TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=12,
        target_min=-100,
        target_max=100,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_answer_length=64,
        max_attempts=128,
        exclude_target_in_operands=False,
    )


def _record(
    *,
    split: str,
    operands: tuple[int, ...],
    target_num: int,
    target_den: int = 1,
    suffix: str,
    families: tuple[str, ...] = ("family-a",),
) -> dict[str, object]:
    digest = suffix.rjust(64, "0")
    record: dict[str, object] = {
        "task_id": f"tces-{digest}",
        "task_family": "tces",
        "split": split,
        "operands": operands,
        "target": {"numerator": target_num, "denominator": target_den},
        "content_hash": f"sha256:{digest}",
        "valid_family_ids": families,
        "valid_family_count": len(families),
    }
    return record


def test_declared_tces_split_names_and_size_ceilings_match_the_spec() -> None:
    assert tuple(EXPECTED_SPLIT_SIZES) == TCES_SPLIT_NAMES
    assert dict(TCES_SPLIT_SIZES) == EXPECTED_SPLIT_SIZES
    assert FINAL_TEST_SPLITS == frozenset({"a_test_single", "a_test_multi", "b_test"})
    assert "b_train" in TEACHER_SPLITS
    assert {"b_monitor", "b_validation", "b_test"}.issubset(EVALUATION_SPLITS)
    assert "format_train" not in TEACHER_SPLITS
    assert "format_eval" not in EVALUATION_SPLITS


@pytest.mark.parametrize("split", ("format_train", "format_eval"))
def test_tces_split_builder_rejects_format_only_splits(split: str) -> None:
    builder = TCESSplitBuilder(0, _fast_config())

    with pytest.raises(ValueError, match="unknown TCES split"):
        builder.lazy_split(split)
    with pytest.raises(ValueError, match="unknown TCES splits"):
        builder.build_splits({split: 1})


def test_exact_family_filters_reject_incomplete_or_out_of_band_counts() -> None:
    assert is_single_family(SimpleNamespace(valid_family_count=1))
    assert not is_single_family(SimpleNamespace(valid_family_count=2))
    assert is_multi_family(SimpleNamespace(valid_family_count=3))
    assert is_multi_family(SimpleNamespace(valid_family_count=8))
    assert not is_multi_family(SimpleNamespace(valid_family_count=2))
    assert not is_multi_family(SimpleNamespace(valid_family_count=9))

    incomplete = SimpleNamespace(
        valid_family_count=1,
        enumeration=SimpleNamespace(complete=False),
    )
    assert exact_valid_family_count(incomplete) is None
    assert not is_single_family(incomplete)


def test_namespaced_split_seeds_are_stable_and_split_specific() -> None:
    first = derive_tces_split_seed(17291, "a_seed_train")
    assert first == derive_tces_split_seed(17291, "a_seed_train")
    assert first == derive_namespaced_seed(17291, "dataset.tces.split", "a_seed_train")
    assert 0 <= first <= MAX_ROOT_SEED
    assert first != derive_tces_split_seed(17291, "a_validation")
    assert first != derive_tces_split_seed(17292, "a_seed_train")

    with pytest.raises(ValueError):
        derive_tces_split_seed(17291, "not_a_split")
    with pytest.raises(ValueError):
        derive_tces_split_seed(True, "a_seed_train")
    with pytest.raises(ValueError):
        derive_tces_split_seed(-1, "a_seed_train")
    with pytest.raises(ValueError):
        derive_tces_split_seed(MAX_ROOT_SEED + 1, "a_seed_train")
    with pytest.raises(ValueError):
        TCESSplitBuilder(-1, _fast_config())


def test_lazy_real_generation_enforces_single_and_multi_family_filters() -> None:
    builder = TCESSplitBuilder(0, _fast_config())
    single = builder.lazy_split("a_test_single", size=2)
    multi = builder.lazy_split("a_test_multi", size=2)

    assert single.generated_count == multi.generated_count == 0
    single_items = single.take(2)
    multi_items = multi.take(2)

    assert all(item.valid_family_count == 1 for item in single_items)
    assert all(3 <= item.valid_family_count <= 8 for item in multi_items)
    assert all(item.enumeration.complete for item in (*single_items, *multi_items))
    assert all(item.task.split == "a_test_single" for item in single_items)
    assert all(item.task.split == "a_test_multi" for item in multi_items)


def test_bundle_construction_is_order_independent_and_numerically_disjoint() -> None:
    requested_forward = {"a_seed_train": 2, "a_validation": 2}
    requested_reverse = {"a_validation": 2, "a_seed_train": 2}

    forward = TCESSplitBuilder(41, _fast_config()).build_splits(requested_forward)
    reverse = TCESSplitBuilder(41, _fast_config()).build_splits(requested_reverse)

    assert {
        split: tuple(item.content_hash for item in items)
        for split, items in forward.items()
    } == {
        split: tuple(item.content_hash for item in items)
        for split, items in reverse.items()
    }
    teacher_keys = {tces_numeric_key(item) for item in forward["a_seed_train"]}
    evaluation_keys = {tces_numeric_key(item) for item in forward["a_validation"]}
    assert teacher_keys.isdisjoint(evaluation_keys)
    assert audit_leakage(forward).clean


def test_split_size_ceilings_and_unknown_names_fail_closed() -> None:
    builder = TCESSplitBuilder(0, _fast_config())
    with pytest.raises(ValueError):
        builder.lazy_split("a_monitor", size=385)
    with pytest.raises(ValueError):
        builder.build_splits({"mystery": 1})
    with pytest.raises(ValueError):
        builder.build_splits({"a_monitor": True})


def test_ood_split_requires_and_records_an_explicit_held_out_operand_profile() -> None:
    base = _fast_config()
    with pytest.raises(ValueError, match="explicit TCESOODProfile"):
        TCESSplitBuilder(0, base).lazy_split("a_ood", size=1)

    with pytest.raises(ValueError, match="must differ"):
        TCESSplitBuilder(
            0,
            base,
            ood_profile=TCESOODProfile(name="not-actually-ood", config=base),
        )

    profile = TCESOODProfile(
        name="held-out-four-operands-v1",
        config=replace(
            base,
            n_operands=4,
            max_tree_depth=4,
            max_ast_nodes=7,
        ),
    )
    builder = TCESSplitBuilder(0, base, ood_profile=profile)
    expected_metadata = {
        "ood_profile": "held-out-four-operands-v1",
        "held_out_dimension": "operand_count",
        "in_distribution_operand_count": 3,
        "ood_operand_count": 4,
    }
    assert builder.split_generation_metadata("a_ood") == expected_metadata

    stream = builder.lazy_split("a_ood", size=1)
    assert stream.generation_metadata == expected_metadata
    instance = stream[0]
    assert len(instance.task.operands) == 4
    assert instance.task.split == "a_ood"

    built = builder.build_splits({"a_ood": 1})
    assert len(built["a_ood"][0].task.operands) == 4
    # The same accessor remains available to populate the built manifest.
    assert builder.split_generation_metadata("a_ood") == expected_metadata


def test_clean_duck_typed_manifest_records_pass_the_audit() -> None:
    records = {
        "a_seed_train": [
            _record(
                split="a_seed_train",
                operands=(2, 3, 7),
                target_num=12,
                suffix="1",
            )
        ],
        "a_validation": [
            _record(
                split="a_validation",
                operands=(4, 6, 9),
                target_num=18,
                suffix="2",
            )
        ],
    }

    report = audit_leakage(records)
    assert report.clean
    assert report.record_count == 2
    assert report.audited_splits == ("a_seed_train", "a_validation")
    assert report.assert_clean() is report
    assert report.to_dict()["clean"] is True


def test_audit_detects_numeric_hash_and_cross_split_leakage() -> None:
    left = _record(
        split="a_seed_train",
        operands=(2, 3, 7),
        target_num=12,
        suffix="a",
    )
    right = _record(
        split="a_validation",
        operands=(7, 2, 3),
        target_num=24,
        target_den=2,
        suffix="a",
    )

    report = audit_leakage([left, right])
    codes = {finding.code for finding in report.findings}
    assert codes.issuperset(
        {
            LeakageCode.DUPLICATE_OPERANDS_TARGET,
            LeakageCode.DUPLICATE_CONTENT_HASH,
            LeakageCode.DUPLICATE_TASK_ID,
            LeakageCode.TASK_ACROSS_SPLITS,
            LeakageCode.TEACHER_EVALUATION_NUMERIC_OVERLAP,
        }
    )
    assert not report.clean
    with pytest.raises(LeakageAuditError) as caught:
        report.assert_clean()
    assert caught.value.report is report


def test_declared_family_disjointness_rules_report_every_overlap() -> None:
    left = _record(
        split="a_seed_train",
        operands=(2, 3, 4),
        target_num=9,
        suffix="1",
        families=("f1", "f2"),
    )
    right = _record(
        split="b_test",
        operands=(5, 6, 7),
        target_num=18,
        suffix="2",
        families=("f2", "f3"),
    )
    rule = FamilyDisjointnessRule.from_records(
        name="stage-a-vs-stage-b",
        left_label="selected A",
        left_records=(left,),
        right_label="held-out B",
        right_records=(right,),
    )

    report = audit_leakage(
        [left, right],
        forbidden_family_rules=(rule,),
    )
    overlaps = report.findings_for(LeakageCode.FORBIDDEN_FAMILY_OVERLAP)
    assert len(overlaps) == 1
    assert overlaps[0].key == "f2"
    assert collect_family_ids((left, right)) == frozenset({"f1", "f2", "f3"})


def test_invalid_records_fail_closed() -> None:
    malformed = {"split": "a_monitor", "task_id": "broken"}
    report = audit_leakage([malformed])
    assert report.findings_for(LeakageCode.INVALID_RECORD)
    with pytest.raises(LeakageAuditError):
        report.assert_clean()


def test_mapping_split_disagreement_is_an_invalid_record() -> None:
    record = _record(
        split="a_seed_train",
        operands=(2, 3, 4),
        target_num=9,
        suffix="1",
    )
    report = audit_leakage({"a_validation": [record]})
    assert report.findings_for(LeakageCode.INVALID_RECORD)
