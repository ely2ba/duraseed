from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from duraseed.calibration_integrity import seal_calibration_action
from duraseed.provenance import canonical_json_bytes
from duraseed.run_records import GenerationRecord, RewardRecord, write_jsonl
from duraseed.runners.teacher_dose_arms import baseline_attempt
from duraseed.runners.teacher_dose_evidence import TEACHER_BASELINE, TeacherBaseline
from duraseed.schemas import VerificationFailure, VerificationResult


def _records() -> tuple[GenerationRecord, RewardRecord]:
    verification = VerificationResult(
        failure_code=VerificationFailure.MISSING_ANSWER_TAG
    )
    generation = GenerationRecord(
        sample_id="sample-1",
        sample_index=0,
        sampling_seed=17,
        purpose="evaluation",
        checkpoint_stage="m0",
        training_step=2,
        sampler_checkpoint_path="tinker://m0/sampler",
        task_manifest_id="sha256:" + "a" * 64,
        task_id="task-1",
        task_family="tces",
        source_split="a_seed_gate",
        prompt_text="prompt",
        completion_text="invalid",
        prompt_tokens=1,
        sampled_tokens=1,
        completion_token_ids=(1,),
        completion_logprobs=(-0.5,),
        reward=0.0,
    )
    reward = RewardRecord(
        reward_id="reward-1",
        sample_id=generation.sample_id,
        task_id=generation.task_id,
        reward=0.0,
        exact_verification=verification,
    )
    return generation, reward


def test_integrity_accepts_strict_tuple_fields_serialized_as_json_arrays(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "baseline-seed-17"
    attempt = arm / "attempt-0001"
    attempt.mkdir(parents=True)
    generation, reward = _records()
    write_jsonl(attempt / "generations.jsonl", (generation,))
    write_jsonl(attempt / "rewards.jsonl", (reward,))
    (attempt / "remote-call-state.json").write_bytes(
        canonical_json_bytes({"pending": None})
    )
    (arm / "coordinate.json").write_bytes(canonical_json_bytes({"arm_id": arm.name}))
    (arm / "completed.json").write_bytes(
        canonical_json_bytes({"arm_id": arm.name, "attempt": 1})
    )

    persisted = json.loads((attempt / "generations.jsonl").read_bytes())
    assert persisted["completion_token_ids"] == [1]
    integrity = seal_calibration_action("teacher-dose", tmp_path, teacher_updates=16)

    assert integrity["accepted_attempts"] == [
        {
            "arm_id": "baseline-seed-17",
            "attempt": "attempt-0001",
            "generation_count": 1,
            "reward_count": 1,
        }
    ]


def test_completed_baseline_replays_json_arrays_without_remote_work() -> None:
    generation, reward = _records()
    baseline = TeacherBaseline((generation,), (reward,))
    payload = json.loads(
        canonical_json_bytes(TEACHER_BASELINE.dump_python(baseline, mode="json"))
    )
    arm = SimpleNamespace(completed=True, completed_payload=payload)
    attempts = SimpleNamespace(open=lambda _arm_id: arm)

    observed = asyncio.run(baseline_attempt(SimpleNamespace(), attempts, seed=17))

    assert observed == baseline
