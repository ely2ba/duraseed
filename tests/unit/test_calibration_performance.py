from __future__ import annotations

from types import SimpleNamespace

from duraseed.calibration_budget import stage_a_budget, teacher_dose_budget
from duraseed.data.manifests import build_tces_record
from duraseed.runners.solver_teacher_cache import solver_teacher_completion
from duraseed.runtime.data import SUPERVISED_MAX_LENGTH
from duraseed.tasks.tces import TCESGenerator, TCESGeneratorConfig


def _budget_inputs(records: tuple) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            teacher_dose=SimpleNamespace(
                calibration_updates=2,
                demonstrations_per_family=(1, 2),
                gate_samples_per_item=2,
            ),
            tinker=SimpleNamespace(
                learning_rates=SimpleNamespace(
                    teacher_seed_sft=SimpleNamespace(grid=(1e-4,))
                )
            ),
        ),
        teacher_sources=SimpleNamespace(
            target_train_manifest=SimpleNamespace(records=records)
        ),
        prompt_pools=SimpleNamespace(
            artifact=SimpleNamespace(bs_slot_order=(), bg_group_order=())
        ),
        max_tokens=SimpleNamespace(selected_max_tokens=8),
    )


def test_calibration_budgets_use_the_supervised_ceiling_without_solving(
    monkeypatch,
) -> None:
    import duraseed.calibration_budget as budget
    import duraseed.runners.solver_teacher_cache as cache

    records = tuple(
        SimpleNamespace(intended_family=family) for family in ("A", "A", "B", "B")
    )
    inputs = _budget_inputs(records)
    monkeypatch.setattr(budget, "TCESTaskManifestRecord", SimpleNamespace)
    monkeypatch.setattr(
        cache,
        "enumerate_task",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("budget calculation invoked the exact solver")
        ),
    )
    monkeypatch.setattr(budget, "teacher_families", lambda *_args: (("A",), ("B",)))
    monkeypatch.setattr(budget, "gate_records", lambda *_args: (object(),))
    monkeypatch.setattr(budget, "_monitor", lambda *_args: (object(),))
    monkeypatch.setattr(budget, "_prompt_length", lambda *_args: 5)
    monkeypatch.setattr(budget, "ordered_pools", lambda *_args: {})
    monkeypatch.setattr(
        budget, "scheduled_records", lambda *_args, **_kwargs: (object(), object())
    )

    teacher = teacher_dose_budget(inputs)
    stage = stage_a_budget(inputs)

    per_datum = SUPERVISED_MAX_LENGTH - 1
    assert teacher.tokens.train == 3 * 2 * 32 * per_datum
    assert stage.tokens.train == 50 * 2 * per_datum + 50 * 2 * (5 + 8 - 1) * 8
    assert stage.fixed_storage_usd == 2.7


def test_solver_teacher_completion_enumerates_once_per_task_family(monkeypatch) -> None:
    import duraseed.runners.solver_teacher_cache as cache

    record = build_tces_record(
        TCESGenerator(
            17,
            TCESGeneratorConfig(
                n_operands=3,
                operand_min=2,
                operand_max=12,
                max_tree_depth=3,
                max_ast_nodes=5,
                max_attempts=512,
                split="a_seed_train",
            ),
        ).generate(3)
    )
    cache._completion.cache_clear()
    original = cache.enumerate_task
    calls = 0

    def counted(task):
        nonlocal calls
        calls += 1
        return original(task)

    monkeypatch.setattr(cache, "enumerate_task", counted)

    first = solver_teacher_completion(record)
    second = solver_teacher_completion(record)

    assert first == second
    assert calls == 1
    assert cache._completion.cache_info().maxsize == 4_096
