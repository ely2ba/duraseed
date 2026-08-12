from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from duraseed.run_records import (
    CheckpointRecord,
    GenerationRecord,
    RewardRecord,
    RunRecord,
    RunStatus,
    StageBEvaluationRecord,
    TrainingMetricRecord,
    append_jsonl,
    create_run_directory,
    read_run_record,
    write_jsonl,
    write_run_record,
)
from duraseed.schemas import VerificationFailure, VerificationResult


def _record() -> RunRecord:
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    return RunRecord(
        protocol_version="duraseed-prepilot-v1",
        git_commit="abc123",
        resolved_config_hash="sha256:" + "1" * 64,
        method="G-B",
        seed=17,
        model_id="Qwen/Qwen3.5-9B-Base",
        renderer=None,
        lora_rank=32,
        task_manifest_ids={"a_rl_train": "sha256:" + "2" * 64},
        parent_tinker_checkpoint_path=None,
        tinker_session_id=None,
        tinker_training_run_id=None,
        final_sampler_checkpoint_path=None,
        final_state_checkpoint_path=None,
        status=RunStatus.PLANNED,
        started_at=now,
        updated_at=now,
    )


def test_run_record_round_trip_and_atomic_update(tmp_path: Path) -> None:
    run_directory = create_run_directory(tmp_path, "run-001")
    record = _record()

    assert write_run_record(run_directory, record) == run_directory / "run.json"
    assert read_run_record(run_directory) == record

    updated = record.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "sampled_tokens": 120,
            "cost_usd": 1.25,
        }
    )
    write_run_record(run_directory, updated)
    assert read_run_record(run_directory) == updated
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "run-001")


def test_run_record_method_is_one_of_all_six_protocol_codes() -> None:
    for method in ("G-U", "G-B", "R-G", "B-S", "B-O", "B-G"):
        assert _record().model_copy(update={"method": method}).method == method

    payload = _record().model_dump(mode="python")
    payload["method"] = "unknown"
    with pytest.raises(ValidationError, match="method"):
        RunRecord.model_validate(payload)


def test_run_record_requires_canonical_task_manifest_ids() -> None:
    payload = _record().model_dump(mode="python")
    payload["task_manifest_ids"] = {"a_rl_train": "not-a-content-id"}

    with pytest.raises(ValidationError, match="canonical sha256"):
        RunRecord.model_validate(payload)


def test_engineering_smoke_run_records_capture_remote_contract_and_budget() -> None:
    smoke = _record().model_copy(
        update={
            "run_kind": "engineering_smoke",
            "method": None,
            "tinker_sdk_version": "0.23.3",
            "tinker_cookbook_version": "0.4.1",
            "project_id": "project-001",
            "authorized_cost_usd": 25.0,
            "reserved_cost_usd": 0.5,
            "price_snapshot_id": "qwen-9b-2026-08-09",
            "service_randomness_contract": "request seeds are best-effort",
        }
    )
    assert RunRecord.model_validate(smoke.model_dump()).method is None

    payload = smoke.model_dump(mode="python")
    payload["method"] = "G-B"
    with pytest.raises(ValidationError, match="engineering_smoke"):
        RunRecord.model_validate(payload)

    payload = _record().model_dump(mode="python")
    payload["method"] = None
    with pytest.raises(ValidationError, match="pilot_method"):
        RunRecord.model_validate(payload)

    payload = smoke.model_dump(mode="python")
    payload["reserved_cost_usd"] = 25.01
    with pytest.raises(ValidationError, match="must not exceed"):
        RunRecord.model_validate(payload)

    payload = smoke.model_dump(mode="python")
    payload["service_randomness_contract"] = " "
    with pytest.raises(ValidationError, match="must not be empty"):
        RunRecord.model_validate(payload)


def test_m0_calibration_run_records_are_methodless() -> None:
    payload = _record().model_dump(mode="python")
    payload.update(
        run_kind="m0_calibration",
        method=None,
        tinker_training_run_ids=("session:train:0", "session:train:1"),
    )

    calibration = RunRecord.model_validate(payload)
    assert calibration.run_kind == "m0_calibration"
    assert calibration.method is None
    assert calibration.tinker_training_run_ids == (
        "session:train:0",
        "session:train:1",
    )

    payload["method"] = "G-B"
    with pytest.raises(ValidationError, match="m0_calibration"):
        RunRecord.model_validate(payload)

    payload["method"] = None
    payload["tinker_training_run_ids"] = ("session:train:0", "session:train:0")
    with pytest.raises(ValidationError, match="must be unique"):
        RunRecord.model_validate(payload)


@pytest.mark.parametrize("run_kind", ["stage_a_calibration", "stage_b_calibration"])
def test_stage_calibration_run_records_are_methodless(run_kind: str) -> None:
    payload = _record().model_dump(mode="python")
    payload.update(run_kind=run_kind, method=None)

    calibration = RunRecord.model_validate(payload)
    assert calibration.run_kind == run_kind
    assert calibration.method is None

    payload["method"] = "B-S"
    with pytest.raises(ValidationError, match=run_kind):
        RunRecord.model_validate(payload)


def test_jsonl_writers_emit_plain_json_objects(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    write_jsonl(path, [{"step": 0, "loss": 1.0}])
    append_jsonl(path, {"step": 1, "loss": 0.5})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"loss": 1.0, "step": 0},
        {"loss": 0.5, "step": 1},
    ]


def test_checkpoint_records_retain_periodic_and_selected_paths() -> None:
    selected = CheckpointRecord(
        phase="stage_a",
        training_step=80,
        sampler_checkpoint_path="tinker://sampler/stage-a-80",
        state_checkpoint_path="tinker://state/stage-a-80",
        selected_for_stage_b=True,
    )
    stage_b = CheckpointRecord(
        phase="stage_b",
        training_step=20,
        sampler_checkpoint_path="tinker://sampler/stage-b-20",
        state_checkpoint_path="tinker://state/stage-b-20",
    )
    assert selected.selected_for_stage_b is True
    assert stage_b.phase == "stage_b"

    with pytest.raises(ValidationError, match="only a Stage-A checkpoint"):
        CheckpointRecord(
            phase="stage_b",
            training_step=20,
            sampler_checkpoint_path="tinker://sampler/stage-b-20",
            state_checkpoint_path="tinker://state/stage-b-20",
            selected_for_stage_b=True,
        )


def test_checkpoint_record_captures_m0_and_optimizer_inheritance() -> None:
    checkpoint = CheckpointRecord(
        phase="m0",
        training_step=0,
        sampler_checkpoint_path="tinker://sampler/m0",
        state_checkpoint_path="tinker://state/m0",
        m0_state_path="tinker://state/m0",
        parent_state_path="tinker://state/base",
        optimizer_inheritance="fresh",
        ttl_seconds=3600,
        selected_origin=True,
    )
    assert checkpoint.phase == "m0"
    assert checkpoint.selected_origin is True

    payload = checkpoint.model_dump(mode="python")
    payload["ttl_seconds"] = -1
    with pytest.raises(ValidationError, match="ttl_seconds"):
        CheckpointRecord.model_validate(payload)

    candidate = checkpoint.model_copy(
        update={"training_step": 5, "selected_origin": False}
    )
    assert CheckpointRecord.model_validate(candidate.model_dump()).phase == "m0"


def test_stage_b_records_retain_zero_shot_gain_curve_and_final_score() -> None:
    zero_shot = StageBEvaluationRecord(
        stage_b_step=0,
        sampler_checkpoint_path="tinker://sampler/selected-stage-a",
        score=0.2,
        gain_from_zero_shot=0.0,
    )
    final = StageBEvaluationRecord(
        stage_b_step=40,
        sampler_checkpoint_path="tinker://sampler/stage-b-40",
        score=0.55,
        gain_from_zero_shot=0.35,
        is_final=True,
    )
    assert zero_shot.score == 0.2
    assert final.gain_from_zero_shot == 0.35
    assert final.is_final is True

    with pytest.raises(ValidationError, match="zero-shot Stage-B point"):
        StageBEvaluationRecord(
            stage_b_step=0,
            sampler_checkpoint_path="tinker://sampler/selected-stage-a",
            score=0.2,
            gain_from_zero_shot=0.1,
        )


def test_typed_jsonl_rows_retain_history_item_outputs_and_reward_identity(
    tmp_path: Path,
) -> None:
    metric = TrainingMetricRecord(
        phase="stage_a",
        training_step=10,
        metrics={"loss": 0.75, "learning_rate": 1.0e-5},
    )
    generation = GenerationRecord(
        sample_id="sample-001",
        sample_index=0,
        sampling_seed=1234,
        purpose="evaluation",
        checkpoint_stage="stage_b",
        training_step=20,
        sampler_checkpoint_path="tinker://sampler/stage-b-20",
        task_manifest_id="sha256:" + "2" * 64,
        task_id="sha256:" + "3" * 64,
        task_family="maps",
        source_split="b_validation",
        prompt_text="A MAPS prompt",
        completion_text="<answer>NEG</answer>",
        prompt_tokens=12,
        sampled_tokens=5,
    )
    verification = VerificationResult(failure_code=VerificationFailure.WRONG_TARGET)
    reward = RewardRecord(
        reward_id="reward-001",
        sample_id=generation.sample_id,
        task_id=generation.task_id,
        reward=0.0,
        exact_verification=verification,
    )

    path = tmp_path / "records.jsonl"
    write_jsonl(path, (metric, generation, reward))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["metrics"]["loss"] == 0.75
    assert rows[1]["sample_id"] == "sample-001"
    assert rows[1]["sample_index"] == 0
    assert rows[1]["sampling_seed"] == 1234
    assert rows[1]["checkpoint_stage"] == "stage_b"
    assert rows[2]["reward_id"] == "reward-001"
    assert rows[2]["exact_verification"]["reward"] == 0.0

    without_sampling_seed = generation.model_dump(mode="python")
    without_sampling_seed.pop("sampling_seed")
    assert GenerationRecord.model_validate(without_sampling_seed).sampling_seed is None

    with pytest.raises(ValidationError, match="reward must match"):
        RewardRecord(
            reward_id="reward-002",
            sample_id=generation.sample_id,
            task_id=generation.task_id,
            reward=1.0,
            exact_verification=verification,
        )

    generation_payload = generation.model_dump(mode="python")
    generation_payload["sample_index"] = -1
    with pytest.raises(ValidationError, match="sample_index"):
        GenerationRecord.model_validate(generation_payload)

    generation_payload = generation.model_dump(mode="python")
    generation_payload["sampling_seed"] = -1
    with pytest.raises(ValidationError, match="sampling_seed"):
        GenerationRecord.model_validate(generation_payload)

    generation_payload = generation.model_dump(mode="python")
    generation_payload["task_manifest_id"] = "not-a-content-id"
    with pytest.raises(ValidationError, match="canonical sha256"):
        GenerationRecord.model_validate(generation_payload)


def test_common_records_support_m0_format_calibration() -> None:
    metric = TrainingMetricRecord(
        phase="m0",
        training_step=1,
        metrics={"loss": 0.5},
    )
    generation = GenerationRecord(
        sample_id="sample-format-001",
        sample_index=0,
        purpose="evaluation",
        checkpoint_stage="m0",
        training_step=1,
        sampler_checkpoint_path="tinker://sampler/m0-format-1",
        task_manifest_id="sha256:" + "2" * 64,
        task_id="sha256:" + "3" * 64,
        task_family="format",
        source_split="format_eval",
        prompt_text="Return one tagged token",
        completion_text="<answer>token</answer>",
        prompt_tokens=5,
        sampled_tokens=4,
    )

    assert TrainingMetricRecord.model_validate(metric.model_dump()).phase == "m0"
    assert GenerationRecord.model_validate(generation.model_dump()).task_family == (
        "format"
    )


def test_generation_record_captures_raw_tinker_smoke_evidence() -> None:
    generation = GenerationRecord(
        sample_id="sample-smoke-001",
        sample_index=0,
        sampling_seed=None,
        purpose="training",
        checkpoint_stage="stage_a",
        training_step=1,
        sampler_checkpoint_path="tinker://sampler/stage-a-1",
        task_manifest_id="sha256:" + "2" * 64,
        task_id="sha256:" + "3" * 64,
        task_family="tces",
        source_split="a_rl_train",
        prompt_text="A TCES prompt",
        completion_text="<answer>1 + 2</answer>",
        prompt_tokens=12,
        sampled_tokens=3,
        sampling_temperature=1.0,
        sampling_top_p=0.95,
        sampling_max_tokens=256,
        run_id="smoke-001",
        method=None,
        seed=17,
        origin_sampler_checkpoint_path="tinker://sampler/m0",
        item_index=0,
        assigned_family_id="family:assigned",
        family_id="family:test",
        panel_role="engineering-smoke",
        completion_token_ids=(101, 102, 103),
        completion_logprobs=(-0.1, None, -0.3),
        stop_reason="stop",
        reward=1.0,
        advantage=0.5,
    )
    assert generation.completion_token_ids == (101, 102, 103)
    assert generation.completion_logprobs == (-0.1, None, -0.3)
    assert generation.assigned_family_id == "family:assigned"
    assert generation.sampling_max_tokens == 256

    for field in ("seed", "item_index"):
        payload = generation.model_dump(mode="python")
        payload[field] = -1
        with pytest.raises(ValidationError, match=field):
            GenerationRecord.model_validate(payload)

    payload = generation.model_dump(mode="python")
    payload["completion_token_ids"] = (101, -1, 103)
    with pytest.raises(ValidationError, match="must be nonnegative"):
        GenerationRecord.model_validate(payload)

    payload = generation.model_dump(mode="python")
    payload["completion_logprobs"] = (-0.1, -0.2)
    with pytest.raises(ValidationError, match="length must match"):
        GenerationRecord.model_validate(payload)

    payload = generation.model_dump(mode="python")
    payload["completion_token_ids"] = None
    with pytest.raises(ValidationError, match="require completion_token_ids"):
        GenerationRecord.model_validate(payload)

    payload = generation.model_dump(mode="python")
    payload["completion_logprobs"] = (-0.1, float("inf"), -0.3)
    with pytest.raises(ValidationError, match="finite number"):
        GenerationRecord.model_validate(payload)

    payload = generation.model_dump(mode="python")
    payload["advantage"] = float("nan")
    with pytest.raises(ValidationError, match="finite number"):
        GenerationRecord.model_validate(payload)
