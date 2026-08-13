from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.config import load_pilot_config
from duraseed.calibration_budget import (
    CalibrationBudget,
    calibration_allocation,
    require_remaining_budget,
)
from duraseed.runners import RunnerGateError
from duraseed.runtime import TokenBudget, TokenLedger
from duraseed.runners.calibration import (
    CalibrationInputs,
    authorize_calibration,
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
    assert tuple(action.name for action in plan.actions) == ("acquisition-calibration",)
    assert all("maps" not in action.name for action in plan.actions)


def test_calibration_authorization_and_maps_freeze() -> None:
    ready = dict(
        panel_frozen=True,
        live_smoke_passed=True,
        human_approval=True,
        remaining_balance_verified=True,
    )
    with pytest.raises(RunnerGateError, match=r"\$300"):
        authorize_calibration(CONFIG, execute=True, authorized_cost_usd="301", **ready)
    assert authorize_calibration(
        CONFIG, execute=True, authorized_cost_usd="300", **ready
    ).authorized_cost_usd == Decimal("300")
    drifted = CONFIG.model_copy(
        update={
            "stage_b": CONFIG.stage_b.model_copy(update={"selected_max_updates": 640})
        }
    )
    with pytest.raises(RunnerGateError, match="step 480"):
        build_plan(drifted)


def test_calibration_cap_is_allocated_from_complete_local_bounds(monkeypatch) -> None:
    import duraseed.calibration_budget as budget

    teacher = CalibrationBudget(TokenBudget(1, 2, 3), 1.0, 80.0)
    stages = {
        1: CalibrationBudget(TokenBudget(4, 5, 6), 2.0, 190.0),
        2: CalibrationBudget(TokenBudget(7, 8, 9), 3.0, 220.0),
    }
    inputs = SimpleNamespace(
        config=SimpleNamespace(
            teacher_dose=SimpleNamespace(demonstrations_per_family=(1, 2))
        )
    )
    monkeypatch.setattr(budget, "teacher_dose_budget", lambda _inputs: teacher)
    monkeypatch.setattr(budget, "stage_a_budget", lambda _inputs, dose: stages[dose])

    allocation = calibration_allocation(inputs)
    expected_stage = 3.01
    assert (allocation.teacher_cap_usd, allocation.stage_a_cap_usd) == (
        80,
        expected_stage,
    )
    assert allocation.stage_a_tokens == TokenBudget(7, 8, 9)
    assert allocation.aggregate_cap_usd == 300
    monkeypatch.setattr(
        budget,
        "teacher_dose_budget",
        lambda _inputs: CalibrationBudget(TokenBudget(0, 0, 0), 0.8, 0.8),
    )
    fixed_cap = calibration_allocation(inputs).teacher_cap_usd
    fixed = TokenLedger(TokenBudget(0, 0, 0), fixed_cap)
    for _ in range(16):
        fixed.reserve_call(TokenBudget(0, 0, 0), fixed_usd=0.05)
        fixed.settle_call(TokenBudget(0, 0, 0))
    assert fixed.committed_cost_usd <= fixed.authorized_usd
    dynamic = TokenLedger(TokenBudget(0, 0, 0), 220)
    assert (
        require_remaining_budget(
            CalibrationBudget(TokenBudget(0, 0, 0), 0, 200),
            dynamic,
            prior_billed_usd=0,
        )["action_cap_usd"]
        == 220
    )

    monkeypatch.setattr(
        budget,
        "teacher_dose_budget",
        lambda _inputs: CalibrationBudget(TokenBudget(1, 2, 3), 1.0, 298.0),
    )
    with pytest.raises(RunnerGateError, match=r"aggregate \$300"):
        calibration_allocation(inputs)
