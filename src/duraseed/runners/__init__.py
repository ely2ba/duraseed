"""Shared, side-effect-free contracts for the two next-gate runners."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Mapping

from duraseed.config import PilotConfig
from duraseed.data.boundary_sources import (
    BoundarySourceContract,
    validate_confirmation_source_contract,
)
from duraseed.data.manifests import DatasetManifest
from duraseed.run_records import GenerationRecord, RunRecord


class RunnerGateError(RuntimeError):
    """A launch or mock-safety condition was not satisfied."""


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    cost_cap_usd: Decimal
    requires: tuple[str, ...] = ()
    remote: bool = True


@dataclass(frozen=True, slots=True)
class RunPlan:
    name: str
    actions: tuple[Action, ...]
    launch_preconditions: tuple[str, ...]
    dry_run_command: str
    mock_command: str
    authorization_command: str

    @property
    def remote_cost_cap_usd(self) -> Decimal:
        return sum(
            (action.cost_cap_usd for action in self.actions if action.remote),
            start=Decimal("0"),
        )


@dataclass(frozen=True, slots=True)
class LaunchAuthorization:
    plan_name: str
    authorized_cost_usd: Decimal


def authenticate_extension1_source(
    config: PilotConfig,
    broad_root: str | Path,
    refine_root: str | Path,
    *,
    expected_parent_manifest_id: str,
) -> BoundarySourceContract:
    """Load and authenticate a completed broad/refinement handoff."""

    broad, refine = Path(broad_root).resolve(), Path(refine_root).resolve()
    if (
        broad.name != "boundary-broad-extension1-20260810T223000Z"
        or refine.name != "boundary-refine-extension1-20260811T104725Z"
    ):
        raise RunnerGateError("source must be the frozen Extension-1 handoff")

    def manifest(root: Path) -> DatasetManifest:
        return DatasetManifest.model_validate_json(
            (root / "a_candidate_manifest.json").read_text()
        )

    def generations(root: Path) -> tuple[GenerationRecord, ...]:
        return tuple(
            GenerationRecord.model_validate_json(row)
            for row in (root / "generations.jsonl").read_text().splitlines()
        )

    broad_manifest = manifest(broad)
    return validate_confirmation_source_contract(
        config=config,
        broad_run=RunRecord.model_validate_json((broad / "run.json").read_text()),
        refinement_run=RunRecord.model_validate_json((refine / "run.json").read_text()),
        broad_plan=json.loads((broad / "preflight.public.json").read_text()),
        refinement_plan=json.loads((refine / "preflight.public.json").read_text()),
        broad_manifest=broad_manifest,
        refinement_manifest=manifest(refine),
        broad_generations=generations(broad),
        refinement_generations=generations(refine),
        expected_parent_manifest_id=expected_parent_manifest_id,
    )


def validate_mock_output_root(path: str | Path | None) -> None:
    """Reject scientific-directory names even though mocks currently write nothing."""

    if path is None:
        return
    resolved = Path(path).expanduser().resolve()
    forbidden = {"runs", "artifacts", "frozen"}
    if forbidden.intersection(resolved.parts):
        raise RunnerGateError(
            "mock output root cannot be inside runs/, artifacts/, or frozen/"
        )


def authorize_launch(
    plan: RunPlan,
    *,
    execute: bool,
    authorized_cost_usd: str | float | Decimal | None,
    preconditions: Mapping[str, bool],
) -> LaunchAuthorization:
    """Fail closed unless the explicit flag, exact cap, and every gate agree."""

    if execute is not True:
        raise RunnerGateError("remote authorization requires explicit --authorize")
    try:
        authorized = Decimal(str(authorized_cost_usd))
    except (InvalidOperation, ValueError) as error:
        raise RunnerGateError(
            "authorized cost must be an exact decimal value"
        ) from error
    if authorized != plan.remote_cost_cap_usd:
        raise RunnerGateError(
            f"authorized cost must equal ${plan.remote_cost_cap_usd} exactly"
        )
    missing = tuple(
        name
        for name in plan.launch_preconditions
        if preconditions.get(name) is not True
    )
    if missing:
        raise RunnerGateError(
            "launch preconditions not satisfied: " + ", ".join(missing)
        )
    return LaunchAuthorization(plan.name, authorized)


def render_preflight(title: str, plan: RunPlan, *, suffix: tuple[str, ...]) -> str:
    return "\n".join(
        (
            f"{title} (no Tinker calls, no writes):",
            *(f"- {a.name}: ${a.cost_cap_usd}" for a in plan.actions),
            f"Remote cost cap: ${plan.remote_cost_cap_usd}",
            f"Dry-run: {plan.dry_run_command}",
            f"Mock: {plan.mock_command}",
            f"Authorization only (does not execute): {plan.authorization_command}",
            *suffix,
        )
    )


__all__ = [
    "Action",
    "LaunchAuthorization",
    "RunPlan",
    "RunnerGateError",
    "authorize_launch",
    "authenticate_extension1_source",
    "render_preflight",
    "validate_mock_output_root",
]
