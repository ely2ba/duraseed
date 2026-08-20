"""CLI surface for the single frozen capability-dose launch."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from duraseed.runners import RunnerGateError
from duraseed.runners.capability_dose_launch import run_remote_capability_dose


def capability_dose_live(
    run_id: str = typer.Option(...),
    boundary_directory: Path = typer.Option(...),
    source_directory: Path = typer.Option(...),
    smoke_acceptance: Path = typer.Option(...),
    m0_selection: Path = typer.Option(...),
    m0_ttl: Path = typer.Option(...),
    panel_split_authorization: Path = typer.Option(...),
    panel_split_equivalence: Path = typer.Option(...),
    max_token_specification: Path = typer.Option(...),
    max_token_authorization: Path = typer.Option(...),
    max_token_evidence: Path = typer.Option(...),
    lifetime_console_evidence: Path = typer.Option(...),
    output_root: Path = typer.Option(Path("runs/acquisition-calibration")),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_human_launch: bool = typer.Option(False),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Execute the authenticated six-epoch B-S capability-dose path."""

    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required")
    try:
        result = asyncio.run(
            run_remote_capability_dose(
                run_id=run_id,
                output_root=output_root,
                config_path=config,
                boundary_directory=boundary_directory,
                source_directory=source_directory,
                smoke_acceptance_path=smoke_acceptance,
                m0_selection_path=m0_selection,
                m0_ttl_path=m0_ttl,
                panel_split_authorization_path=panel_split_authorization,
                panel_split_equivalence_path=panel_split_equivalence,
                max_token_specification_path=max_token_specification,
                max_token_authorization_path=max_token_authorization,
                max_token_evidence_path=max_token_evidence,
                lifetime_console_evidence_path=lifetime_console_evidence,
                project_id=project_id,
                authorized_cost_usd=authorized_cost_usd,
                human_approval=confirm_human_launch,
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"capability-dose artifacts: {result}")


__all__ = ["capability_dose_live"]
