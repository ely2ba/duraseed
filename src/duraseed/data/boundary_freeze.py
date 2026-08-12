"""Three-cohort boundary reduction, disabled until Phase-7 equivalence."""

from __future__ import annotations

from collections.abc import Sequence
from duraseed.data.panel_matching import FamilyPanelCandidate


BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES = 36
BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS = "pending_phase7_three_cohort_check"


class BoundaryFreezeUnverifiedError(RuntimeError):
    """The freeze output cannot be computed before Phase-7 equivalence."""


def freeze_three_cohort_panels(
    cohorts: Sequence[object],
    candidates: Sequence[FamilyPanelCandidate],
    *,
    panel_size: int,
    allocation_seed: int,
    training_seeds: Sequence[int],
    m0_checkpoint_path: str,
) -> None:
    """Freeze panels only after Phase 7 records exact three-cohort equivalence.

    This public path deliberately fails closed today.  Phase 7 may change the
    required status only after comparing this reduction with the archived
    three-cohort implementation on the completed real evidence.  Fixture or
    Extension-1 checks do not authorize its output.
    """

    del cohorts, candidates, panel_size, allocation_seed, training_seeds
    del m0_checkpoint_path
    raise BoundaryFreezeUnverifiedError(
        "three-cohort freeze is pending the Phase-7 old-vs-new exact check"
    )


__all__ = [
    "BOUNDARY_PANEL_FREEZE_EQUIVALENCE_STATUS",
    "BOUNDARY_PANEL_FREEZE_MINIMUM_CANDIDATES",
    "BoundaryFreezeUnverifiedError",
    "freeze_three_cohort_panels",
]
