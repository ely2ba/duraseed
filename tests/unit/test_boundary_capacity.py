from __future__ import annotations

from duraseed.boundary_capacity import audit_family_split_capacities
from duraseed.data.panel_capacity import audit_family_split_capacity
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


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


def test_parallel_capacity_audits_equal_serial_and_preserve_input_order() -> None:
    config = _config()
    generator = TCESGenerator(5, _config(split="a_candidate"))
    instances = tuple(generator.generate(index) for index in range(4))
    family_ids = tuple(row.intended_family for row in reversed(instances))
    templates = {row.intended_family: row for row in instances}
    requirements = {"a_validation": 1}
    expected = tuple(
        audit_family_split_capacity(
            templates[family_id], config, root_seed=11, requirements=requirements
        )
        for family_id in family_ids
    )

    observed = audit_family_split_capacities(
        templates,
        family_ids,
        config,
        root_seed=11,
        requirements=requirements,
    )

    assert observed == expected
    assert tuple(row.family_id for row in observed) == family_ids
