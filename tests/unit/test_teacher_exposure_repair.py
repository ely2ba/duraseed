from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.calibration_billing import reconcile_calibration_billing
from duraseed.calibration_billing_requirement import (
    calibration_billing_requirement,
)
from duraseed.calibration_parent import PARENT_BILLED_USD, PARENT_RUN_ID
from duraseed.config import load_pilot_config
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.run_records import (
    RunRecord,
    RunStatus,
    TrainingMetricRecord,
    read_run_record,
    write_run_record,
)
from duraseed.runners import RunnerGateError
from duraseed.runners.teacher_dose_evidence import teacher_families, teacher_records
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.teacher_exposure_spec import REPAIR_SPEC
from duraseed.training.teacher_exposure import (
    ExposurePointEvidence,
    ExposureTrajectoryEvidence,
    TeacherExposureEvidence,
    assess_exposure_point,
    select_teacher_exposure,
)
from tests.unit.teacher_dose_fixtures import _assessment, _raw_gate_fixture
from tests.unit.teacher_allocation_fixtures import freezer_sources


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _trajectory(seed: int, passing: tuple[bool, ...]) -> ExposureTrajectoryEvidence:
    updates = (4, 8, 12)[: len(passing)]
    points = tuple(
        ExposurePointEvidence(
            step,
            "tinker://dose-2/sampler",
            _assessment(2, seed=seed, passing=passed),
        )
        for step, passed in zip(updates, passing, strict=True)
    )
    metrics = tuple(
        TrainingMetricRecord(phase="stage_a", training_step=step, metrics={"loss": 0.5})
        for step in range(1, updates[-1] + 1)
    )
    return ExposureTrajectoryEvidence(seed, points, metrics)


def _evidence(*trajectories: ExposureTrajectoryEvidence) -> TeacherExposureEvidence:
    return TeacherExposureEvidence(REPAIR_SPEC, {"run_id": "parent"}, trajectories)


def test_progressive_selector_chooses_earliest_joint_pass_and_stays_compact() -> None:
    evidence = _evidence(
        _trajectory(17, (True, True)),
        _trajectory(37, (False, True)),
    )

    selected = select_teacher_exposure(evidence)

    assert selected.status == "selected"
    assert selected.selected_updates == 8
    assert selected.recipe is not None and selected.recipe.selected_updates == 8
    assert b"generations" not in canonical_json_bytes(evidence)


def test_progressive_selector_fails_closed_at_twelve_and_rejects_duplicates() -> None:
    first = _trajectory(17, (False, False, False))
    second = _trajectory(37, (False, False, False))
    assert select_teacher_exposure(_evidence(first, second)).status == (
        "no_stable_checkpoint"
    )

    with pytest.raises(RunnerGateError, match="duplicate trajectories"):
        select_teacher_exposure(_evidence(first, first, second))
    duplicate_point = ExposureTrajectoryEvidence(
        17,
        (first.points[0], first.points[0]),
        first.metrics[:4],
    )
    with pytest.raises(RunnerGateError, match="checkpoint ladder"):
        select_teacher_exposure(_evidence(duplicate_point, _trajectory(37, (False,))))


def test_exposure_point_rejects_a_mislabeled_training_step() -> None:
    fixture = _raw_gate_fixture()
    with pytest.raises(RunnerGateError, match="generation step differs"):
        assess_exposure_point(
            config=CONFIG.teacher_dose,
            panel=fixture["panel_artifact"],
            gate_manifest=fixture["gate_manifest"],
            m0_sampler_path=fixture["m0_sampler_checkpoint_path"],
            seed=17,
            updates=4,
            sampler_path=fixture["seeded_sampler_checkpoint_path"],
            target_generations=fixture["targeted_generations"],
            target_rewards=fixture["targeted_rewards"],
            baseline_generations=fixture["sentinel_m0_generations"],
            baseline_rewards=fixture["sentinel_m0_rewards"],
            sentinel_generations=fixture["sentinel_seeded_generations"],
            sentinel_rewards=fixture["sentinel_seeded_rewards"],
            metrics=(),
        )


def test_teacher_record_prefilter_is_identical_and_solves_only_selected_dose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from duraseed.runners import teacher_dose_evidence as module

    sources = freezer_sources(CONFIG)
    inputs = SimpleNamespace(teacher_sources=sources)
    families, _ = teacher_families(inputs, 17)
    original = module.solver_teacher_completion
    calls = []

    def counted(record):  # type: ignore[no-untyped-def]
        calls.append(record.task_id)
        return original(record)

    monkeypatch.setattr(module, "solver_teacher_completion", counted)
    records = teacher_records(inputs, families, 2)
    expected = tuple(
        row.task_id
        for family in sorted(families)
        for row in sorted(
            (
                value
                for value in sources.target_train_manifest.records
                if value.intended_family == family
            ),
            key=lambda value: (value.item_index, value.task_id),
        )[:2]
    )

    assert tuple(row.task_id for row in records) == expected
    assert tuple(calls) == expected


def _run(root: Path, status: RunStatus) -> RunRecord:
    finished = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    run = RunRecord(
        protocol_version="duraseed-prepilot-v1",
        git_commit="abc123",
        resolved_config_hash=SHA_A,
        run_kind="stage_a_calibration",
        method=None,
        seed=17,
        model_id="Qwen/Qwen3.5-9B-Base",
        renderer="qwen3",
        lora_rank=32,
        task_manifest_ids={"a_seed_gate": SHA_A},
        parent_tinker_checkpoint_path="tinker://m0/state",
        tinker_session_id="session-child",
        status=status,
        started_at=finished,
        updated_at=finished,
        finished_at=finished,
        cost_usd=3.0,
        project_id="project",
        authorized_cost_usd=199.36,
        reserved_cost_usd=199.36,
    )
    write_run_record(root, run)
    return run


def _required(run_status: str, **changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "duraseed-calibration-billing-required-v1",
        "status": "pending",
        "run_id": "child",
        "project_id": "project",
        "session_ids": ["session-child"],
        "execution_finished_at_utc": "2026-08-17T12:00:00+00:00",
        "run_status": run_status,
        "terminal_status": None,
        "terminal_sha256": None,
        "local_observed_cost_usd": 3.0,
        "action_caps_usd": {"teacher-dose": 44.27, "stage-a": 155.09},
        "aggregate_cap_usd": 199.36,
        "parent_run_id": PARENT_RUN_ID,
        "parent_billing_sha256": SHA_A,
        "parent_billed_usd": PARENT_BILLED_USD,
        "protected_reserve_usd": 984.46,
        "lifetime_calibration_cap_usd": 300,
    }
    value.update(changes)
    return value


def test_final_reconciler_accepts_child_cap_and_enforces_lifetime(
    tmp_path: Path,
) -> None:
    root = tmp_path / "child"
    root.mkdir()
    _run(root, RunStatus.COMPLETED)
    (root / "session-lineage.json").write_bytes(
        canonical_json_bytes(
            {"schema_version": "lineage", "session_ids": ["session-child"]}
        )
    )
    (root / "billing-reconciliation-required.json").write_bytes(
        canonical_json_bytes(_required("completed"))
    )
    (root / "preflight.json").write_bytes(
        canonical_json_bytes({"parent_calibration": {"billing_sha256": SHA_A}})
    )
    raw = tmp_path / "raw.json"
    raw.write_bytes(canonical_json_bytes({"data": [{"session_id": "session-child"}]}))
    reconciliation = tmp_path / "reconciliation.json"
    reconciliation.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "duraseed-calibration-final-reconciliation-v1",
                "status": "billing_reconciled",
                "run_id": "child",
                "project_id": "project",
                "session_ids": ["session-child"],
                "raw_billing_sha256": sha256_bytes(raw.read_bytes()),
                "raw_billing_entry_count": 1,
                "action_billed_usd": {"teacher-dose": 1, "stage-a": 2},
                "aggregate_billed_usd": 3,
                "remaining_balance_usd": 4000,
                "protected_reserve_usd": 984.46,
                "protected_reserve_survives": True,
                "raw_usage_cutoff_utc": "2026-08-17T13:00:00Z",
                "reconciled_at_utc": "2026-08-17T14:00:00Z",
            }
        )
    )

    result = reconcile_calibration_billing(root, reconciliation, raw)

    assert result["aggregate_billed_usd"] == 3
    assert read_run_record(root).cost_usd == 3
    required = _required("completed", parent_billing_sha256=SHA_B)
    (root / "billing-reconciliation-required.json").write_bytes(
        canonical_json_bytes(required)
    )
    with pytest.raises(RunnerGateError, match="incomplete"):
        reconcile_calibration_billing(root, reconciliation, raw)


def test_failed_handoff_is_emitted_only_for_exact_no_stable_terminal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "child"
    root.mkdir()
    (root / "session-lineage.json").write_text(
        json.dumps({"session_ids": ["session-child"]})
    )
    inputs = SimpleNamespace(
        run_id="child",
        project_id="project",
        teacher_ledger=TokenLedger(TokenBudget(0, 0, 0), 44.27),
        stage_a_ledger=TokenLedger(TokenBudget(0, 0, 0), 155.09),
        parent_teacher_evidence=SimpleNamespace(
            parent_run_id=PARENT_RUN_ID,
            parent_billing_sha256=SHA_A,
            parent_billed_usd=PARENT_BILLED_USD,
            protected_reserve_usd=984.46,
        ),
    )
    now = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
    assert (
        calibration_billing_requirement(inputs, root, RunStatus.FAILED, now, 1.0)
        is None
    )
    (root / "teacher-dose-terminal.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "duraseed-teacher-exposure-terminal-v2",
                "status": "no_stable_checkpoint",
                "run_id": "child",
            }
        )
    )

    requirement = calibration_billing_requirement(
        inputs, root, RunStatus.FAILED, now, 1.0
    )

    assert requirement is not None
    assert requirement["run_status"] == "failed"
    assert requirement["terminal_status"] == "no_stable_checkpoint"
