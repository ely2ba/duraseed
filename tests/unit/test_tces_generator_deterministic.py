"""Focused deterministic and completeness tests for the TCES generator."""

from fractions import Fraction

import pytest

from duraseed.schemas import TCESTask
from duraseed.tasks.tces.ast import (
    BinaryOperator,
    leaf_values,
    node_count,
    tree_depth,
)
from duraseed.tasks.tces.enumerate import enumerate_task
from duraseed.tasks.tces.generator import (
    GeneratedTCESInstance,
    TCESGenerationError,
    TCESFamilyGenerator,
    TCESGenerator,
    TCESGeneratorConfig,
    render_prompt,
    render_teacher_completion,
    task_content_hash,
)
from duraseed.tasks.tces.teacher import build_teacher_trace, verify_teacher_trace
from duraseed.tasks.tces.verifier import verify_completion


def _small_config(**overrides: object) -> TCESGeneratorConfig:
    values: dict[str, object] = {
        "n_operands": 3,
        "operand_min": 2,
        "operand_max": 12,
        "max_tree_depth": 3,
        "max_ast_nodes": 5,
        "max_attempts": 128,
    }
    values.update(overrides)
    return TCESGeneratorConfig(**values)  # type: ignore[arg-type]


def _assert_latent_has_no_identity_steps(instance: GeneratedTCESInstance) -> None:
    for step in build_teacher_trace(instance.latent_expression).steps:
        if step.operator is BinaryOperator.ADD:
            assert step.left != 0 and step.right != 0
        elif step.operator is BinaryOperator.SUB:
            assert step.right != 0
        elif step.operator is BinaryOperator.MUL:
            assert step.left != 1 and step.right != 1
        else:
            assert step.right != 1


def test_primary_defaults_match_the_frozen_generation_contract() -> None:
    config = TCESGeneratorConfig()

    assert config.n_operands == 5
    assert (config.operand_min, config.operand_max) == (2, 25)
    assert config.require_distinct_operands
    assert (config.target_min, config.target_max) == (-250, 250)
    assert config.allowed_ops == tuple(BinaryOperator)
    assert config.allow_fractional_intermediates
    assert config.max_abs_intermediate == 10_000
    assert config.max_denominator == 1_000
    assert config.max_tree_depth == 5
    assert config.exclude_target_in_operands
    assert config.exclude_trivial_identity_steps
    assert config.min_valid_families == 1
    assert config.max_valid_families is None


def test_generation_is_deterministic_and_item_streams_are_independent() -> None:
    config = _small_config()
    generator = TCESGenerator(91_337, config)

    forward = tuple(generator.generate(index) for index in range(2))
    reverse = tuple(generator.generate(index) for index in reversed(range(2)))

    assert forward == tuple(reversed(reverse))
    assert forward == tuple(
        TCESGenerator(91_337, config).generate(index) for index in range(2)
    )
    assert forward[0].content_hash != forward[1].content_hash


def test_family_generation_varies_numbers_but_preserves_exact_family() -> None:
    template_config = _small_config(
        operand_max=20, max_attempts=256, split="a_candidate"
    )
    variant_config = _small_config(
        operand_max=20, max_attempts=256, split="a_seed_train"
    )
    template = TCESGenerator(7, template_config).generate(0)
    generator = TCESFamilyGenerator(99, template, variant_config)

    variants = generator.generate_many(4)

    assert variants == TCESFamilyGenerator(99, template, variant_config).generate_many(
        4
    )
    assert len({variant.content_hash for variant in variants}) == 4
    assert all(variant.content_hash != template.content_hash for variant in variants)
    assert all(variant.task.operands != template.task.operands for variant in variants)
    for index, variant in enumerate(variants):
        assert variant.item_index == index
        assert variant.task.split == "a_seed_train"
        assert variant.intended_family == template.intended_family
        assert variant.intended_family in variant.enumeration.complete_family_set
        verification = verify_completion(variant.teacher_trace, variant.task)
        assert verification.reward == 1.0
        assert verification.strategy_family_id == template.intended_family


def test_family_generation_rejects_incompatible_template_config() -> None:
    template_config = _small_config()
    template = TCESGenerator(7, template_config).generate(0)

    with pytest.raises(ValueError, match="operand counts differ"):
        TCESFamilyGenerator(
            99,
            template,
            TCESGeneratorConfig(
                n_operands=4,
                operand_min=2,
                operand_max=12,
                max_tree_depth=3,
                max_ast_nodes=7,
            ),
        )

    with pytest.raises(ValueError, match="requires distinct operands"):
        TCESFamilyGenerator(
            99,
            template,
            _small_config(require_distinct_operands=False),
        )

    with pytest.raises(ValueError, match="task constraints differ"):
        TCESFamilyGenerator(
            99,
            template,
            _small_config(max_denominator=99),
        )


def test_generated_item_is_full_tree_verified_enumerated_and_labeled() -> None:
    instance = TCESGenerator(1, _small_config()).generate(1)
    task = instance.task

    assert tuple(sorted(leaf_values(instance.latent_expression))) == task.operands
    assert node_count(instance.latent_expression) == 2 * len(task.operands) - 1
    assert tree_depth(instance.latent_expression) <= task.constraints.max_tree_depth
    assert task.target.denominator == 1
    assert -250 <= task.target.numerator <= 250
    assert task.target.as_fraction() not in map(Fraction, task.operands)
    _assert_latent_has_no_identity_steps(instance)

    assert instance.enumeration.complete
    assert instance.valid_family_count == len(instance.enumeration.families)
    assert instance.valid_expression_count == len(instance.enumeration.expressions)
    assert instance.intended_family in instance.enumeration.complete_family_set
    assert instance.intended_family_id == instance.intended_family

    assert verify_teacher_trace(instance.latent_expression, instance.teacher_trace)
    verification = verify_completion(instance.teacher_trace, task)
    assert verification.reward == 1.0
    assert verification.strategy_family_id == instance.intended_family
    assert render_teacher_completion(instance) == instance.teacher_trace


def test_exhaustive_labels_follow_public_verifier_not_latent_only_filters() -> None:
    config = _small_config(
        allow_fractional_intermediates=False,
        require_positive_intermediates=True,
        exclude_trivial_identity_steps=True,
    )
    instance = TCESGenerator(2, config).generate(2)

    # Latent construction obeys the stricter private sampling filters.
    for step in build_teacher_trace(instance.latent_expression).steps:
        assert step.result.denominator == 1
        assert step.result > 0
    _assert_latent_has_no_identity_steps(instance)

    # Family labels nevertheless include every solution accepted by the task's
    # public prompt and verifier, which do allow fractional, negative, and
    # identity intermediates within the exact rational guards.
    fresh = enumerate_task(
        instance.task,
        allow_fractional_intermediates=True,
        exclude_trivial_identity_steps=False,
        require_positive_intermediates=False,
        max_expressions_per_value=None,
    )
    assert instance.enumeration == fresh
    assert fresh.complete


def test_task_id_and_content_hash_cover_semantics_but_not_split() -> None:
    train = TCESGenerator(44, _small_config(split="a_seed_train")).generate(3)
    test = TCESGenerator(44, _small_config(split="a_test_single")).generate(3)

    assert train.task.split == "a_seed_train"
    assert test.task.split == "a_test_single"
    assert train.content_hash == test.content_hash
    assert train.task.task_id == test.task.task_id
    assert task_content_hash(train.task) == train.content_hash
    assert train.content_hash.startswith("sha256:")
    assert len(train.content_hash) == 71
    assert train.task.task_id == "tces-" + train.content_hash.removeprefix("sha256:")

    reordered = TCESTask(
        operands=tuple(reversed(train.task.operands)),
        target=train.task.target,
        allowed_ops=tuple(reversed(train.task.allowed_ops)),
        constraints=train.task.constraints,
    )
    assert task_content_hash(reordered) == train.content_hash


def test_family_count_filter_supports_single_family_generation() -> None:
    config = TCESGeneratorConfig(
        n_operands=2,
        operand_min=2,
        operand_max=10,
        max_tree_depth=2,
        max_ast_nodes=3,
        min_valid_families=1,
        max_valid_families=1,
        max_attempts=32,
    )
    instance = TCESGenerator(0, config).generate()

    assert instance.valid_family_count == 1
    assert instance.valid_expression_count >= 1


def test_impossible_family_filter_fails_after_fixed_attempt_budget() -> None:
    config = TCESGeneratorConfig(
        n_operands=2,
        operand_min=2,
        operand_max=10,
        max_tree_depth=2,
        max_ast_nodes=3,
        min_valid_families=2,
        max_valid_families=2,
        max_attempts=4,
    )

    with pytest.raises(TCESGenerationError, match="4 deterministic attempts"):
        TCESGenerator(0, config).generate()


def test_prompt_declares_the_exact_expression_contract() -> None:
    instance = TCESGenerator(7, _small_config()).generate()
    prompt = render_prompt(instance.task)

    assert f"Numbers: {list(instance.task.operands)}" in prompt
    assert f"Target: {instance.task.target.numerator}" in prompt
    assert "Allowed binary operations: +, -, *, /" in prompt
    assert "Use every listed number exactly once" in prompt
    assert "Division uses exact rational arithmetic" in prompt
    assert "<answer>EXPRESSION</answer>" in prompt


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_operands": 1},
        {"operand_min": -1},
        {"operand_max": 10**1024},
        {"n_operands": 5, "operand_min": 2, "operand_max": 4},
        {
            "n_operands": 3,
            "operand_min": 2,
            "operand_max": 12,
            "max_tree_depth": 3,
            "max_ast_nodes": 5,
            "max_answer_length": 8,
        },
        {"n_operands": 5, "max_tree_depth": 3},
        {"n_operands": 5, "max_ast_nodes": 8},
        {"min_valid_families": 3, "max_valid_families": 2},
        {"allowed_ops": ("%",)},
    ],
)
def test_invalid_generation_configs_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TCESGeneratorConfig(**kwargs)  # type: ignore[arg-type]
