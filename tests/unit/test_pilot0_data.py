from duraseed.data.manifests import TCESTaskManifestRecord, task_semantic_hash
from duraseed.pilot0_data import _tces_completion
from duraseed.schemas import ExactRational, TCESTask
from duraseed.tasks.tces import enumerate_task
from duraseed.training.reward import verify_task_completion


def test_solver_completion_rebuilds_frozen_family_without_search() -> None:
    provisional = TCESTask(
        operands=(1, 2, 3),
        target=ExactRational(numerator=6),
        split="a_rl_train",
    )
    task_id = task_semantic_hash(provisional)
    task = provisional.model_copy(update={"task_id": task_id})
    enumeration = enumerate_task(task)
    family_ids = tuple(sorted(enumeration.family_ids))
    record = TCESTaskManifestRecord(
        task_id=task_id,
        split=task.split,
        generator_version="1.0.0",
        generator_seed=11,
        item_index=0,
        accepted_attempt=0,
        prompt_template_id="tces_v1",
        content_hash=task_id,
        operands=task.operands,
        target=task.target,
        allowed_ops=task.allowed_ops,
        constraints=task.constraints,
        intended_family=family_ids[0],
        valid_family_ids=family_ids,
        valid_family_count=len(family_ids),
        valid_expression_count=len(enumeration.expressions),
        minimum_depth=enumeration.shortest_depth or 1,
    )

    result = verify_task_completion(_tces_completion(record), task)

    assert result.reward == 1.0
    assert result.strategy_family_id == record.intended_family
