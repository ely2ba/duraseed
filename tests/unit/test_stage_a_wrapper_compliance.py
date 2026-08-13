from types import SimpleNamespace

from duraseed.runners.stage_a_evidence import paired_items


def _observation(completion: str, *, sample_id: str):
    generation = SimpleNamespace(
        task_id="task-1",
        sampling_seed=17,
        sample_index=0,
        assigned_family_id="family-1",
        completion_text=completion,
        stop_reason="stop",
    )
    reward = SimpleNamespace(
        sample_id=sample_id,
        reward=1.0,
        exact_verification=SimpleNamespace(valid_answer_tag=True),
    )
    return SimpleNamespace(generation=generation, reward=reward)


def test_stage_a_wrapper_rejects_free_form_text_around_valid_answer_pair() -> None:
    origin = _observation("<answer>ADD(r1,r2)</answer>", sample_id="origin")
    current = _observation(
        "reasoning outside wrapper\n<answer>ADD(r1,r2)</answer>",
        sample_id="current",
    )

    item = paired_items((origin,), (current,))[0]

    assert item.origin_wrapper_compliance == (True,)
    assert item.current_wrapper_compliance == (False,)
