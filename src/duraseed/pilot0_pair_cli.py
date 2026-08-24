"""CLI surface for one separately authorized Pilot-0 seed pair."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from duraseed.config import load_pilot_config
from duraseed.pilot0_source_build import (
    build_pilot_seed_sources,
    write_pilot_seed_source_set,
)
from duraseed.runners import RunnerGateError
from duraseed.runners.pilot0_pair_launch import run_remote_pilot0_pair


def pilot0_sources_build(
    boundary_directory: Path = typer.Option(...),
    output: Path = typer.Option(...),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
) -> None:
    """Build and persist both frozen Pilot source bundles without remote access."""

    try:
        sources = build_pilot_seed_sources(
            load_pilot_config(config), boundary_directory
        )
        result = write_pilot_seed_source_set(output, sources)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Pilot-0 prepared sources: {result}")


def pilot0_pair_live(
    run_id: str = typer.Option(...),
    pair_index: int = typer.Option(..., min=1, max=2),
    prepared_sources: Path = typer.Option(...),
    boundary_directory: Path = typer.Option(...),
    source_directory: Path = typer.Option(...),
    smoke_acceptance: Path = typer.Option(...),
    m0_selection: Path = typer.Option(...),
    m0_ttl: Path = typer.Option(...),
    panel_split_authorization: Path = typer.Option(...),
    panel_split_equivalence: Path = typer.Option(...),
    dose_run: Path = typer.Option(...),
    stage_b_recipe: Path = typer.Option(...),
    lifetime_console_evidence: Path = typer.Option(...),
    prior_pair_root: Path | None = typer.Option(None),
    output_root: Path = typer.Option(Path("runs/pilot0")),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_human_launch: bool = typer.Option(False),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Execute exactly one frozen B-S/B-G paired Pilot seed."""

    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required")
    try:
        result = asyncio.run(
            run_remote_pilot0_pair(
                run_id=run_id,
                pair_index=pair_index,
                prepared_sources=prepared_sources,
                output_root=output_root,
                config_path=config,
                boundary_directory=boundary_directory,
                source_directory=source_directory,
                smoke_acceptance_path=smoke_acceptance,
                m0_selection_path=m0_selection,
                m0_ttl_path=m0_ttl,
                panel_split_authorization_path=panel_split_authorization,
                panel_split_equivalence_path=panel_split_equivalence,
                dose_root=dose_run,
                stage_b_recipe_path=stage_b_recipe,
                lifetime_console_evidence_path=lifetime_console_evidence,
                prior_pair_root=prior_pair_root,
                project_id=project_id,
                authorized_cost_usd=authorized_cost_usd,
                human_approval=confirm_human_launch,
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Pilot-0 pair artifacts: {result}")


__all__ = ["pilot0_pair_live", "pilot0_sources_build"]
