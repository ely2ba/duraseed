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


def test_three_and_six_worker_capacity_audits_are_identical_and_ordered() -> None:
    config = _config()
    generator = TCESGenerator(5, _config(split="a_candidate"))
    instances = tuple(generator.generate(index) for index in range(6))
    family_ids = tuple(row.intended_family for row in reversed(instances))
    templates = {row.intended_family: row for row in instances}
    requirements = {"a_validation": 1}
    serial = tuple(
        audit_family_split_capacity(
            templates[family_id], config, root_seed=11, requirements=requirements
        )
        for family_id in family_ids
    )
    three_workers = audit_family_split_capacities(
        templates,
        family_ids,
        config,
        root_seed=11,
        requirements=requirements,
        max_workers=3,
    )
    six_workers = audit_family_split_capacities(
        templates,
        family_ids,
        config,
        root_seed=11,
        requirements=requirements,
        max_workers=6,
    )

    assert serial == three_workers == six_workers
    assert tuple(row.family_id for row in six_workers) == family_ids
