from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from duraseed.config import load_pilot_config
from duraseed.runners import RunnerGateError
from duraseed.runners.calibration import (
    CalibrationInputs,
    authorize_action,
    build_plan,
    reduce_calibration,
)
from duraseed.training import STAGE_A_LEARNING_RATE_GRIDS
from tests.unit.teacher_allocation_fixtures import (
    fixed_token_measurer,
    freezer_sources,
)
from tests.unit.teacher_dose_fixtures import _assessment
from tests.unit.test_stage_a_calibration_decision import _final, _screen


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _screens(method: str):
    return tuple(
        _screen(method, rate, set(range(40 - index * 10)))
        for index, rate in enumerate(STAGE_A_LEARNING_RATE_GRIDS[method])
    )


def test_calibration_real_mock_flow_invokes_all_carried_decisions(monkeypatch) -> None:
    import duraseed.training.teacher_allocation_freeze as freezer
    from duraseed.data.matching import FamilyBlockMatchStatus
    from duraseed.data.panels import PanelLabel
    from tests.unit.teacher_allocation_fixtures import _teacher_fixture

    sources = replace(freezer_sources(CONFIG), selected_dose=1)
    _, candidates = _teacher_fixture(
        protected_rows=24, root_seed=CONFIG.seed, operand_stride=1
    )
    panel_ids = frozenset(
        sources.panel.panel_a_family_ids + sources.panel.panel_b_family_ids
    )
    target_ids = {row.task_id for row in sources.target_train_manifest.records}
    target_candidates = tuple(
        row for row in candidates if row.source_record.task_id in target_ids
    )
    random_by_family = {}
    for row in candidates:
        if row.source_record.intended_family not in panel_ids:
            random_by_family.setdefault(row.source_record.intended_family, []).append(
                row
            )
    random_families = tuple(sorted(random_by_family))
    target_a = freezer.target_blocks(sources, target_candidates, PanelLabel.A)

    class Stream:
        def __init__(self, root_seed, generator_config):
            del root_seed, generator_config

        def lazy_split(self, split, *, size):
            del split, size
            return tuple(
                type("Template", (), {"intended_family": family})()
                for family in random_families
            )

    monkeypatch.setattr(freezer, "TCESSplitBuilder", Stream)
    monkeypatch.setattr(
        freezer, "family_structure_key", lambda *args: target_a[0].structure_key
    )
    monkeypatch.setattr(
        freezer,
        "generate_random_family",
        lambda source, template, *args: tuple(
            row.source_record for row in random_by_family[template.intended_family]
        ),
    )
    monkeypatch.setattr(
        freezer,
        "candidate_rows",
        lambda manifest, families, measurer: (
            target_candidates
            if manifest is sources.target_train_manifest
            else tuple(
                row
                for row in candidates
                if row.source_record.task_id
                in {record.task_id for record in manifest.records}
            )
        ),
    )
    inputs = CalibrationInputs(
        calibration_doses=(_assessment(1),),
        verification_dose=_assessment(1, seed=37),
        allocation_sources=sources,
        token_measurer=fixed_token_measurer,
        bs_screens=_screens("B-S"),
        bg_screens=_screens("B-G"),
        final_evidence=(_final("B-S", 1e-4), _final("B-G", 1e-5)),
    )
    result = reduce_calibration(CONFIG, inputs)

    assert result.dose.selected_dose == 1
    assert result.allocation.selected
    assert result.allocation.status is FamilyBlockMatchStatus.SELECTED
    assert tuple(row.selected_learning_rate for row in result.learning_rates) == (
        1e-4,
        1e-5,
    )
    assert result.duration.selected_max_updates == 50
    plan = build_plan(CONFIG)
    assert tuple(action.name for action in plan.actions) == (
        "teacher-dose",
        "teacher-allocation",
        "stage-a",
    )
    assert all("maps" not in action.name for action in plan.actions)


def test_calibration_authorization_and_maps_freeze() -> None:
    ready = dict(
        action="teacher-dose",
        prerequisite_selected=False,
        panel_frozen=True,
        phase6_smoke_passed=True,
        human_approval=True,
        remaining_balance_verified=True,
    )
    with pytest.raises(RunnerGateError, match=r"\$150"):
        authorize_action(CONFIG, execute=True, authorized_cost_usd="151", **ready)
    assert authorize_action(
        CONFIG, execute=True, authorized_cost_usd="150", **ready
    ).authorized_cost_usd == Decimal("150")
    assert (
        authorize_action(
            CONFIG,
            action="teacher-allocation",
            execute=True,
            authorized_cost_usd="0",
            prerequisite_selected=True,
            panel_frozen=True,
            phase6_smoke_passed=True,
            human_approval=True,
            remaining_balance_verified=True,
        ).authorized_cost_usd
        == 0
    )
    drifted = CONFIG.model_copy(
        update={
            "stage_b": CONFIG.stage_b.model_copy(update={"selected_max_updates": 640})
        }
    )
    with pytest.raises(RunnerGateError, match="step 480"):
        build_plan(drifted)
