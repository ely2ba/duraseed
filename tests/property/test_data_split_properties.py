from __future__ import annotations

from hypothesis import given, settings, strategies as st

from duraseed.data.leakage import audit_leakage
from duraseed.data.splits import TCESSplitBuilder, tces_numeric_key
from duraseed.tasks.tces import TCESGeneratorConfig


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


@given(
    root_seed=st.integers(min_value=0, max_value=10_000),
    split=st.sampled_from(("a_monitor", "a_validation", "a_seed_gate")),
    index=st.integers(min_value=0, max_value=2),
)
@settings(max_examples=8, deadline=None)
def test_lazy_indexed_generation_is_a_pure_function_of_seed_split_and_index(
    root_seed: int,
    split: str,
    index: int,
) -> None:
    first = TCESSplitBuilder(root_seed, _fast_config()).lazy_split(split, size=3)
    second = TCESSplitBuilder(root_seed, _fast_config()).lazy_split(split, size=3)

    # Different access histories must not alter the indexed result.
    _ = first[2]
    observed = first[index]
    repeated = second[index]

    assert observed == repeated
    assert observed.content_hash == repeated.content_hash
    assert observed.task.task_id == repeated.task.task_id


@given(root_seed=st.integers(min_value=0, max_value=2_000))
@settings(max_examples=6, deadline=None)
def test_rebuilt_teacher_and_evaluation_prefixes_are_identical_and_disjoint(
    root_seed: int,
) -> None:
    sizes = {"a_seed_train": 2, "a_validation": 2}
    first = TCESSplitBuilder(root_seed, _fast_config()).build_splits(sizes)
    second = TCESSplitBuilder(root_seed, _fast_config()).build_splits(
        dict(reversed(tuple(sizes.items())))
    )

    assert first == second
    teacher_keys = {tces_numeric_key(item) for item in first["a_seed_train"]}
    evaluation_keys = {tces_numeric_key(item) for item in first["a_validation"]}
    assert teacher_keys.isdisjoint(evaluation_keys)
    assert audit_leakage(first).clean
