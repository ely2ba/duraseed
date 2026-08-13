from pathlib import Path
from types import SimpleNamespace

import pytest

from duraseed.config import load_pilot_config
from duraseed.runners.teacher_dose_live import (
    _safe_arm,
    _validate_completed_prefix,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_pilot_config(ROOT / "duraseed_pilot_config.yaml")


def _completed(*, verification_arm: bool) -> frozenset[str]:
    dose = CONFIG.teacher_dose.demonstrations_per_family[0]
    rates = sorted(CONFIG.tinker.learning_rates.teacher_seed_sft.grid)
    values = {
        "baseline-seed-17",
        "baseline-seed-37",
        *(_safe_arm(f"seed-17-dose-{dose}-lr-{rate:.0e}") for rate in rates),
    }
    if verification_arm:
        values.add(_safe_arm(f"seed-37-dose-{dose}-lr-{rates[0]:.0e}"))
    return frozenset(values)


@pytest.mark.parametrize("verification_arm", [False, True])
def test_teacher_resume_accepts_verification_prefixes(verification_arm: bool) -> None:
    _validate_completed_prefix(
        SimpleNamespace(config=CONFIG), _completed(verification_arm=verification_arm)
    )
