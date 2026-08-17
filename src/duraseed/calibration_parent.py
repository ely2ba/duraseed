"""Read-only authentication of the failed calibration parent and its billing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any

from duraseed.calibration_prior_repair import load_prior_repair
from duraseed.provenance import canonical_json_bytes, sha256_bytes
from duraseed.runners import RunnerGateError
from duraseed.runtime import PRICE_SNAPSHOT, UsageQuantities, parse_billing_usage
from duraseed.teacher_exposure_spec import (
    PRIOR_REPAIR_CHILD_AUTHORIZATION_USD,
    PRIOR_REPAIR_TEACHER_CAP_USD,
)


PARENT_RUN_ID = "acquisition-calibration-20260816T082800Z"
PARENT_PREFLIGHT_SHA256 = (
    "sha256:83fce64fce5d27a438408ddc6e377298944471ef4a9e464fa5ddd4250c30d42d"
)
PARENT_TERMINAL_SHA256 = (
    "sha256:5ade05b71fe576d399e131206331bc73d936194e27f5a589cad1d3fe8369bdb7"
)
PARENT_INTEGRITY_SHA256 = (
    "sha256:58efd025f435532f7c0cdb1bb4119fcb0bd0c8e48883fb71770981ebe3477def"
)
PARENT_RAW_BILLING_SHA256 = (
    "sha256:9dcb9c6083226d9363fe5f46c50c20f041645bd15ddbae1e58827ca7e93b96bc"
)
PARENT_BILLED_USD = 11.510623512
PROTECTED_RESERVE_USD = 984.46
PARENT_BILLING_CUTOFF = "2026-08-17T08:00:00Z"
PARENT_SESSION_ID = "9e923e2f-e347-54b9-8787-111aa907dd39"
PARENT_USAGE = UsageQuantities(
    prefill_tokens=163_103,
    cached_prefill_tokens=660_992,
    sample_tokens=5_144_874,
    train_tokens=718_866,
)
_BASELINE_HASHES = {
    17: {
        "completed": "sha256:8b89f3da773da2234749247f37cfe86ecb40d5f7a70fb32db30500532ad99925",
        "generations": "sha256:a5bad2201fd1ada528a4505a780ed44fe8fc7985dc2ad0dd9a78eb2bdcc7cf09",
        "rewards": "sha256:60f88564bddfec5de8bd3853348c14efaebc94f33e64b1846b8ebf1b92ba310b",
    },
    37: {
        "completed": "sha256:67c670b2802c251e678238f307ac8bc12c13822200b5893e78877e48ed4ddc54",
        "generations": "sha256:27c2afc69b4b376af9efe5dbe75cfb5ddc750661a3067ef41d6ea91218843e90",
        "rewards": "sha256:761677b177180ad2a616ad75d28fb2957c178b48374502fd4cb448a19df6fa85",
    },
}


@dataclass(frozen=True, slots=True)
class ParentBaseline:
    generations: tuple[Any, ...]
    rewards: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class CalibrationParentEvidence:
    parent_run_id: str
    parent_preflight_sha256: str
    parent_terminal_sha256: str
    parent_integrity_sha256: str
    parent_baseline_hashes: dict[int, dict[str, str]]
    parent_billing_sha256: str
    parent_raw_billing_sha256: str
    parent_billed_usd: float
    remaining_balance_usd: float
    protected_reserve_usd: float
    prior_repair_lineage: dict[str, object]
    prior_repair_teacher_cap_usd: float
    baselines: tuple[tuple[int, ParentBaseline], ...]

    def baseline(self, seed: int) -> ParentBaseline:
        try:
            return dict(self.baselines)[seed]
        except KeyError as error:
            raise RunnerGateError(
                "parent baseline omits a repair orientation"
            ) from error

    @property
    def lineage(self) -> dict[str, object]:
        return {
            "run_id": self.parent_run_id,
            "preflight_sha256": self.parent_preflight_sha256,
            "terminal_sha256": self.parent_terminal_sha256,
            "integrity_sha256": self.parent_integrity_sha256,
            "baseline_hashes": {
                str(seed): hashes
                for seed, hashes in self.parent_baseline_hashes.items()
            },
            "billing_sha256": self.parent_billing_sha256,
            "raw_billing_sha256": self.parent_raw_billing_sha256,
            "billed_usd": self.parent_billed_usd,
        }

    @property
    def lifetime_sunk_usd(self) -> float:
        return float(
            Decimal(str(self.parent_billed_usd))
            + Decimal(str(self.prior_repair_teacher_cap_usd))
        )


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerGateError(f"invalid repair parent {label}") from error
    if not isinstance(value, dict):
        raise RunnerGateError(f"repair parent {label} is not an object")
    return value, raw


def _jsonl(path: Path, model: type) -> tuple[Any, ...]:
    try:
        return tuple(
            model.model_validate_json(line) for line in path.read_bytes().splitlines()
        )
    except (OSError, ValueError) as error:
        raise RunnerGateError(f"invalid repair parent stream: {path.name}") from error


def _baseline(root: Path, seed: int) -> ParentBaseline:
    from duraseed.run_records import GenerationRecord, RewardRecord

    arm = root / "teacher-dose-arms" / f"baseline-seed-{seed}"
    completed, completed_raw = _object(arm / "completed.json", "baseline")
    attempt = arm / "attempt-0001"
    generations_path = attempt / "generations.jsonl"
    rewards_path = attempt / "rewards.jsonl"
    expected = _BASELINE_HASHES[seed]
    if (
        sha256_bytes(completed_raw) != expected["completed"]
        or sha256_bytes(generations_path.read_bytes()) != expected["generations"]
        or sha256_bytes(rewards_path.read_bytes()) != expected["rewards"]
        or completed.get("arm_id") != f"baseline-seed-{seed}"
        or completed.get("preflight_sha256") != PARENT_PREFLIGHT_SHA256
        or completed.get("attempt") != 1
    ):
        raise RunnerGateError("repair parent baseline hash or identity differs")
    baseline = ParentBaseline(
        _jsonl(generations_path, GenerationRecord),
        _jsonl(rewards_path, RewardRecord),
    )
    if canonical_json_bytes(completed.get("evidence")) != canonical_json_bytes(
        baseline
    ):
        raise RunnerGateError(
            "repair parent completed baseline differs from raw streams"
        )
    return baseline


def _billing(
    reconciliation_path: Path, raw_path: Path, *, project_id: str
) -> tuple[str, str, float]:
    reconciliation, reconciliation_raw = _object(
        reconciliation_path, "billing reconciliation"
    )
    raw, raw_bytes = _object(raw_path, "raw billing")
    rows = raw.get("data")
    if not isinstance(rows, list):
        raise RunnerGateError("repair parent raw billing omits events")
    parent_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("session_id") == PARENT_SESSION_ID
    ]
    bucket_ends = tuple(
        row.get("bucket_end")
        for row in parent_rows
        if isinstance(row.get("bucket_end"), str)
    )
    usage = parse_billing_usage(
        {"data": parent_rows}, session_id=PARENT_SESSION_ID, project_id=project_id
    )
    billed = PRICE_SNAPSHOT.cost(usage)
    try:
        cutoff = datetime.fromisoformat(PARENT_BILLING_CUTOFF.replace("Z", "+00:00"))
        reconciled = datetime.fromisoformat(
            str(reconciliation.get("reconciled_at_utc", "")).replace("Z", "+00:00")
        )
    except ValueError as error:
        raise RunnerGateError(
            "repair parent billing chronology is malformed"
        ) from error
    console_before = Decimal(str(reconciliation.get("pre_run_console_spend_usd")))
    console_after = Decimal(str(reconciliation.get("post_run_console_spend_usd")))
    if (
        sha256_bytes(raw_bytes) != PARENT_RAW_BILLING_SHA256
        or len(rows) != 90
        or len(parent_rows) != 35
        or not bucket_ends
        or max(bucket_ends) > PARENT_BILLING_CUTOFF
        or usage != PARENT_USAGE
        or not math.isclose(billed, PARENT_BILLED_USD, rel_tol=0, abs_tol=1e-12)
        or reconciliation.get("schema_version")
        != "duraseed-teacher-exposure-parent-billing-v1"
        or reconciliation.get("status") != "billing_reconciled"
        or reconciliation.get("parent_run_id") != PARENT_RUN_ID
        or reconciliation.get("parent_preflight_sha256") != PARENT_PREFLIGHT_SHA256
        or reconciliation.get("parent_terminal_sha256") != PARENT_TERMINAL_SHA256
        or reconciliation.get("parent_integrity_sha256") != PARENT_INTEGRITY_SHA256
        or reconciliation.get("project_id") != project_id
        or reconciliation.get("parent_session_id") != PARENT_SESSION_ID
        or reconciliation.get("raw_billing_sha256") != PARENT_RAW_BILLING_SHA256
        or reconciliation.get("raw_billing_entry_count") != 90
        or reconciliation.get("parent_session_entry_count") != 35
        or reconciliation.get("raw_usage_cutoff_utc") != PARENT_BILLING_CUTOFF
        or reconciliation.get("raw_quantities")
        != {
            "prefill_tokens": PARENT_USAGE.prefill_tokens,
            "cached_prefill_tokens": PARENT_USAGE.cached_prefill_tokens,
            "sample_tokens": PARENT_USAGE.sample_tokens,
            "train_tokens": PARENT_USAGE.train_tokens,
        }
        or reconciliation.get("raw_derived_billed_usd") != PARENT_BILLED_USD
        or console_before != Decimal("66.19")
        or console_after != Decimal("77.70")
        or console_after - console_before != Decimal("11.51")
        or reconciliation.get("remaining_balance_usd") != 4922.30
        or reconciliation.get("protected_reserve_usd") != PROTECTED_RESERVE_USD
        or reconciliation.get("protected_reserve_survives") is not True
        or reconciliation.get("new_run_authorization_usd")
        != PRIOR_REPAIR_CHILD_AUTHORIZATION_USD
        or reconciliation.get("lifetime_calibration_cap_usd") != 300
        or not math.isclose(
            reconciliation.get("lifetime_worst_case_usd", -1),
            PARENT_BILLED_USD + PRIOR_REPAIR_CHILD_AUTHORIZATION_USD,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or 4922.30 - PRIOR_REPAIR_CHILD_AUTHORIZATION_USD < PROTECTED_RESERVE_USD
        or not isinstance(reconciliation.get("authorizer"), str)
        or not reconciliation["authorizer"].strip()
        or reconciled.tzinfo is None
        or reconciled.astimezone(UTC) < cutoff
    ):
        raise RunnerGateError("repair parent billing evidence differs")
    return (
        sha256_bytes(reconciliation_raw),
        sha256_bytes(raw_bytes),
        PARENT_BILLED_USD,
    )


def load_calibration_parent(
    parent_directory: str | Path,
    reconciliation_path: str | Path,
    raw_billing_path: str | Path,
    *,
    project_id: str,
) -> CalibrationParentEvidence:
    """Bind the child launch to the immutable failed parent and lag-cleared spend."""

    from duraseed.run_records import RunStatus, read_run_record

    root = Path(parent_directory)
    preflight, preflight_raw = _object(root / "preflight.json", "preflight")
    terminal, terminal_raw = _object(root / "teacher-dose-terminal.json", "terminal")
    integrity, integrity_raw = _object(
        root / "teacher-dose-arms/integrity.json", "integrity"
    )
    run = read_run_record(root)
    if (
        root.name != PARENT_RUN_ID
        or sha256_bytes(preflight_raw) != PARENT_PREFLIGHT_SHA256
        or sha256_bytes(terminal_raw) != PARENT_TERMINAL_SHA256
        or sha256_bytes(integrity_raw) != PARENT_INTEGRITY_SHA256
        or preflight.get("run_id") != PARENT_RUN_ID
        or preflight.get("project_id") != project_id
        or terminal.get("status") != "verification_failed"
        or terminal.get("preflight_sha256") != PARENT_PREFLIGHT_SHA256
        or terminal.get("integrity", {}).get("artifact_sha256")
        != PARENT_INTEGRITY_SHA256
        or terminal.get("integrity", {})
        != {**integrity, "artifact_sha256": PARENT_INTEGRITY_SHA256}
        or run.status is not RunStatus.FAILED
        or run.project_id != project_id
        or run.tinker_session_id != PARENT_SESSION_ID
        or run.finished_at is None
        or terminal.get("run_git_commit") != run.git_commit
    ):
        raise RunnerGateError("repair parent terminal lineage differs")
    billing_hash, raw_hash, billed = _billing(
        Path(reconciliation_path), Path(raw_billing_path), project_id=project_id
    )
    parent_lineage = {
        "run_id": PARENT_RUN_ID,
        "preflight_sha256": PARENT_PREFLIGHT_SHA256,
        "terminal_sha256": PARENT_TERMINAL_SHA256,
        "integrity_sha256": PARENT_INTEGRITY_SHA256,
        "baseline_hashes": {
            str(seed): hashes for seed, hashes in _BASELINE_HASHES.items()
        },
        "billing_sha256": billing_hash,
        "raw_billing_sha256": raw_hash,
        "billed_usd": billed,
    }
    return CalibrationParentEvidence(
        PARENT_RUN_ID,
        PARENT_PREFLIGHT_SHA256,
        PARENT_TERMINAL_SHA256,
        PARENT_INTEGRITY_SHA256,
        _BASELINE_HASHES,
        billing_hash,
        raw_hash,
        billed,
        4922.30,
        PROTECTED_RESERVE_USD,
        load_prior_repair(root, project_id=project_id, parent_lineage=parent_lineage),
        PRIOR_REPAIR_TEACHER_CAP_USD,
        tuple((seed, _baseline(root, seed)) for seed in (17, 37)),
    )


__all__ = [
    "CalibrationParentEvidence",
    "PARENT_BILLED_USD",
    "PARENT_RUN_ID",
    "load_calibration_parent",
]
