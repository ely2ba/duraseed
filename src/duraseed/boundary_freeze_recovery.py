"""One-use recovery for the completed three-cohort reducer comparison."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path
import subprocess
from typing import Any

import duraseed.boundary_capacity as boundary_capacity
from duraseed.boundary_teacher_tokens import archived_token_counter
from duraseed.data.boundary_freeze import _reduce_three_cohort_panels
from duraseed.data.boundary_freeze_contracts import (
    BoundaryFreezeResult,
    BoundaryFreezeSettings,
)
from duraseed.provenance import canonical_json_bytes, canonical_json_value
from duraseed.runners import RunnerGateError


RECOVERY_RUN_ID = "boundary-panel-freeze-20260815T032009Z"
RECOVERY_CURRENT_V1_COMMIT = "37c67b1ce8caf7c83cb6ef5cba45e12ab46675f7"
RECOVERY_ARCHIVED_V0_COMMIT = "a9c65f92c857a9f51e26fdfe08b7ef6e30ee9597"
RECOVERY_TRANSCRIPT_EVENT_SHA256 = (
    "sha256:1fa8fb1e2d0bfcd72558493b55a0d25d1965d97e450288e634dfa210e6a7b037"
)
RECOVERY_TRANSCRIPT_EVENT_LINE = 64918
RECOVERY_SOURCE_SHA256 = (
    "sha256:673a42d8ba47a30743a6c54079bbdf53822641aa4e396ae5e57930c50bba2043"
)
RECOVERY_SETTINGS_SHA256 = (
    "sha256:b1ddb88ff9ef31a0f0396b31c87e6d089c7b973705ae42615589c21a66fe6361"
)
RECOVERY_V0_SOURCE_SHA256 = {
    "src/duraseed/tinker_boundary_panel_freeze.py": (
        "sha256:55820094e4d99143fa37ec926ec15afdfd402fabb14021093f676919aa4e1432"
    ),
    "src/duraseed/tinker_boundary_confirmation.py": (
        "sha256:b1fa035143f2d035304b95e477f917132eb4bf276f9604aadad8bab199e43a24"
    ),
}
RECOVERY_V1_SOURCE_SHA256 = {
    "src/duraseed/boundary_capacity.py": (
        "sha256:e6ccc4118eb1eb68db9a4a90dc4c81d71e4630150acccb62d176a858e50c77de"
    ),
    "src/duraseed/data/boundary_freeze.py": (
        "sha256:4d7925c74f7b55ca804493a67f0fa7a908027cb9f3d34f21e7bcc243b479d1b3"
    ),
    "src/duraseed/data/boundary_freeze_contracts.py": (
        "sha256:ffbca63afbbee6125fa907f2ade9c5dc902caf2aed321394dfdcaa1b1e3742d6"
    ),
    "src/duraseed/boundary_teacher_tokens.py": (
        "sha256:00cf6712e558fa6477c64eb56bedd75b03a4768ca294bb91cd14d9021222dc44"
    ),
}
RECOVERY_WORKERS = 6


class BoundaryFreezeRecoveryError(RunnerGateError):
    """The failed comparison incident cannot be recovered safely."""


def source_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {
        name: "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        raise BoundaryFreezeRecoveryError("cannot authenticate git commit") from error


def clean_checkout_identity(root: Path) -> dict[str, Any]:
    """Require a clean committed recovery descended from the initial code."""

    try:
        head = _git_head(root)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "merge-base",
                "--is-ancestor",
                RECOVERY_CURRENT_V1_COMMIT,
                head,
            ],
            check=True,
            capture_output=True,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise BoundaryFreezeRecoveryError(
            "cannot authenticate the recovery checkout"
        ) from error
    if status or len(head) != 40:
        raise BoundaryFreezeRecoveryError(
            "recovery checkout must be clean and committed"
        )
    return {
        "initial_current_v1_git_commit": RECOVERY_CURRENT_V1_COMMIT,
        "recovery_git_commit": head,
        "recovery_checkout_clean": True,
    }


def authenticate_recovery(
    *,
    repository_root: Path,
    archived_v0_root: Path,
    run_id: str,
    source_input_sha256: str,
    settings_sha256: str,
) -> dict[str, Any]:
    """Authenticate every executable input to the one recovery."""

    return validate_recovery(
        run_id=run_id,
        source_input_sha256=source_input_sha256,
        settings_sha256=settings_sha256,
        archived_v0_commit=_git_head(archived_v0_root),
        archived_v0_source_sha256=source_hashes(
            archived_v0_root, RECOVERY_V0_SOURCE_SHA256
        ),
        current_v1_source_sha256=source_hashes(
            repository_root, RECOVERY_V1_SOURCE_SHA256
        ),
        checkout_identity=clean_checkout_identity(repository_root),
    )


def validate_recovery(
    *,
    run_id: str,
    source_input_sha256: str,
    settings_sha256: str,
    archived_v0_commit: str,
    archived_v0_source_sha256: dict[str, str],
    current_v1_source_sha256: dict[str, str],
    checkout_identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind recovery to the exact inputs and sources used by the failed run."""

    if (
        run_id != RECOVERY_RUN_ID
        or source_input_sha256 != RECOVERY_SOURCE_SHA256
        or settings_sha256 != RECOVERY_SETTINGS_SHA256
        or archived_v0_commit != RECOVERY_ARCHIVED_V0_COMMIT
        or archived_v0_source_sha256 != RECOVERY_V0_SOURCE_SHA256
        or current_v1_source_sha256 != RECOVERY_V1_SOURCE_SHA256
        or checkout_identity.get("initial_current_v1_git_commit")
        != RECOVERY_CURRENT_V1_COMMIT
        or checkout_identity.get("recovery_checkout_clean") is not True
        or not isinstance(checkout_identity.get("recovery_git_commit"), str)
        or len(checkout_identity["recovery_git_commit"]) != 40
    ):
        raise BoundaryFreezeRecoveryError(
            "post-comparison recovery input or reducer source changed"
        )
    return {
        "mode": "current_v1_rematerialization_after_exact_comparison",
        "initial_run_id": RECOVERY_RUN_ID,
        **checkout_identity,
        "initial_archived_v0_git_commit": RECOVERY_ARCHIVED_V0_COMMIT,
        "initial_exact_old_v0_current_v1_comparison_completed": True,
        "initial_failure_stage": "run_record_publication_after_comparison",
        "initial_failure_transcript_event_sha256": RECOVERY_TRANSCRIPT_EVENT_SHA256,
        "initial_failure_transcript_event_line": RECOVERY_TRANSCRIPT_EVENT_LINE,
        "archived_v0_reexecuted": False,
        "archived_v0_scientific_hash_basis": (
            "derived_transitively_from_prior_exact_equality_and_"
            "deterministic_current_v1_rematerialization"
        ),
        "current_v1_reexecuted": True,
        "current_v1_capacity_workers": RECOVERY_WORKERS,
        "source_input_sha256": source_input_sha256,
    }


def rematerialize_current(
    cohorts, settings: BoundaryFreezeSettings
) -> tuple[BoundaryFreezeResult, bytes]:
    """Rerun deterministic v1 only, at the already-proven total CPU load."""

    previous_workers = boundary_capacity._CAPACITY_AUDIT_WORKERS
    boundary_capacity._CAPACITY_AUDIT_WORKERS = RECOVERY_WORKERS
    try:
        result = _reduce_three_cohort_panels(
            cohorts,
            settings=settings,
            teacher_trace_token_counter=archived_token_counter(),
        )
    finally:
        boundary_capacity._CAPACITY_AUDIT_WORKERS = previous_workers
    return result, canonical_json_bytes(canonical_json_value(result))


def mark_recovered_equivalence(
    equivalence: dict[str, Any], recovery: dict[str, Any]
) -> None:
    equivalence["old_new_identical_basis"] = (
        "historical_exact_byte_comparison_before_publication_failure"
    )
    archived = equivalence["archived_v0"]
    archived["scientific_outputs_sha256_basis"] = recovery[
        "archived_v0_scientific_hash_basis"
    ]
    archived["scientific_outputs_sha256_observed_in_recovery"] = False
    equivalence["comparison_execution"] = recovery


__all__ = [
    "BoundaryFreezeRecoveryError",
    "authenticate_recovery",
    "clean_checkout_identity",
    "mark_recovered_equivalence",
    "rematerialize_current",
    "source_hashes",
    "validate_recovery",
]
