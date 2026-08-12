from __future__ import annotations

from dataclasses import replace

import pytest

from duraseed.data.panel_capacity import (
    PanelCapacityError,
    audit_family_split_capacity,
)
from duraseed.data.splits import derive_tces_split_seed
from duraseed.tasks.tces import (
    TCESFamilyGenerator,
    TCESGenerator,
    TCESGeneratorConfig,
)


def _config(*, split: str | None = None) -> TCESGeneratorConfig:
    return TCESGeneratorConfig(
        n_operands=3,
        operand_min=2,
        operand_max=20,
        max_tree_depth=3,
        max_ast_nodes=5,
        max_attempts=256,
        split=split,
    )


def test_capacity_audit_is_deterministic_and_cross_split_disjoint() -> None:
    template = TCESGenerator(5, _config(split="a_candidate")).generate(0)
    requirements = {"a_seed_train": 2, "a_validation": 2}

    first = audit_family_split_capacity(
        template,
        _config(),
        root_seed=11,
        requirements=requirements,
        probe_multiplier=2,
    )
    second = audit_family_split_capacity(
        template,
        _config(),
        root_seed=11,
        requirements=requirements,
        probe_multiplier=2,
    )

    assert first == second
    assert first.passed is True
    assert first.family_id == template.intended_family
    assert first.available_by_split == {"a_seed_train": 4, "a_validation": 4}
    assert first.available_disjoint_instances == 8
    assert all(
        row.first_generation_failure_index is None for row in first.split_capacities
    )


@pytest.mark.parametrize("split", ["a_candidate", "a_ood", "not-a-split"])
def test_capacity_audit_rejects_non_panel_split_requirements(split: str) -> None:
    template = TCESGenerator(5, _config(split="a_candidate")).generate(0)

    with pytest.raises(PanelCapacityError, match="protected in-distribution"):
        audit_family_split_capacity(
            template,
            _config(),
            root_seed=11,
            requirements={split: 1},
        )


def test_capacity_audit_rejects_invalid_counts() -> None:
    template = TCESGenerator(5, _config(split="a_candidate")).generate(0)

    with pytest.raises(PanelCapacityError, match="positive integers"):
        audit_family_split_capacity(
            template,
            _config(),
            root_seed=11,
            requirements={"a_validation": 0},
        )


def test_capacity_audit_excludes_candidate_numeric_tasks() -> None:
    template = TCESGenerator(5, _config(split="a_candidate")).generate(0)
    validation_config = replace(_config(), split="a_validation")
    forbidden = TCESFamilyGenerator(
        derive_tces_split_seed(11, "a_validation"),
        template,
        validation_config,
    ).generate(0)

    audit = audit_family_split_capacity(
        template,
        _config(),
        root_seed=11,
        requirements={"a_validation": 1},
        probe_multiplier=1,
        forbidden_records=(forbidden,),
    )

    assert audit.passed is False
    assert audit.available_by_split == {"a_validation": 0}


def test_single_test_capacity_is_relative_to_intervention_families() -> None:
    template = TCESGenerator(5, _config(split="a_candidate")).generate(0)
    test_config = replace(_config(), split="a_test_single")
    first_test_item = TCESFamilyGenerator(
        derive_tces_split_seed(11, "a_test_single"),
        template,
        test_config,
    ).generate(0)

    audit = audit_family_split_capacity(
        template,
        _config(),
        root_seed=11,
        requirements={"a_test_single": 1},
        protected_family_ids=(template.intended_family,),
    )

    assert audit.passed is True
    assert audit.available_by_split == {"a_test_single": 1}

    alternative = next(
        family_id
        for family_id in first_test_item.valid_family_ids
        if family_id != template.intended_family
    )
    blocked = audit_family_split_capacity(
        template,
        _config(),
        root_seed=11,
        requirements={"a_test_single": 1},
        probe_multiplier=1,
        protected_family_ids=(template.intended_family, alternative),
    )
    assert blocked.passed is False

    with pytest.raises(PanelCapacityError, match="protected family set"):
        audit_family_split_capacity(
            template,
            _config(),
            root_seed=11,
            requirements={"a_test_single": 1},
        )
