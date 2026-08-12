from __future__ import annotations

import pytest

from duraseed.training import TeacherDoseEvidenceError, summarize_teacher_dose_gate
from tests.unit.teacher_dose_fixtures import _raw_gate_fixture


def test_raw_teacher_dose_gate_replays_exact_evidence_and_panel_roles() -> None:
    evidence = _raw_gate_fixture()

    summary = summarize_teacher_dose_gate(**evidence)

    assert summary.targeted_family_count == 1
    assert summary.targeted_item_count == 8
    assert summary.mixed_group_rate == 1.0
    assert summary.all_zero_group_rate == 0.0
    assert summary.all_one_group_rate == 0.0
    assert summary.family_reachability == 1.0
    assert summary.format_compliance == 1.0
    assert summary.sentinel_control_change.mean_change == 0.0
    assert summary.sampled_token_mean_surprisal == 0.5
    assert summary.sampled_token_logprob_coverage == 1.0


def test_raw_teacher_dose_gate_rejects_unpaired_sentinel_sampling() -> None:
    evidence = _raw_gate_fixture()
    rows = list(evidence["sentinel_seeded_generations"])
    rows[0] = rows[0].model_copy(update={"sampling_seed": 999})
    evidence["sentinel_seeded_generations"] = tuple(rows)

    with pytest.raises(TeacherDoseEvidenceError, match="not paired"):
        summarize_teacher_dose_gate(**evidence)


def test_raw_teacher_dose_gate_rejects_wrong_exact_reward() -> None:
    evidence = _raw_gate_fixture()
    rows = list(evidence["targeted_generations"])
    rows[0] = rows[0].model_copy(update={"completion_text": "<answer>0</answer>"})
    evidence["targeted_generations"] = tuple(rows)

    with pytest.raises(TeacherDoseEvidenceError, match="authoritative"):
        summarize_teacher_dose_gate(**evidence)
