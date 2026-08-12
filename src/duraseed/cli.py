"""Side-effect-free next-gate preflights and explicit launch authorization."""

from __future__ import annotations

from pathlib import Path

import typer

from duraseed.config import load_pilot_config
from duraseed.runners import RunnerGateError, authorize_launch
from duraseed.runners import boundary_extension as boundary
from duraseed.runners import calibration


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("validate")
def validate(
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
) -> None:
    """Validate the local frozen config and both next-gate plans."""

    resolved = load_pilot_config(config)
    boundary.build_plan()
    calibration.build_plan(resolved)
    typer.echo(f"local validation passed: {resolved.resolved_config_hash()}")


@app.command("boundary-extension")
def boundary_extension(
    dry_run: bool = typer.Option(False, "--dry-run"),
    execute: bool = typer.Option(False, "--authorize"),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_live_smoke: bool = typer.Option(False),
    confirm_boundary_extension_approval: bool = typer.Option(False),
    confirm_source_authenticated: bool = typer.Option(False),
    confirm_remaining_balance: bool = typer.Option(False),
) -> None:
    """Print the fixed plan or validate a non-executing authorization envelope."""

    if dry_run and not execute:
        typer.echo(boundary.preflight_text())
        return
    try:
        authorization = authorize_launch(
            boundary.build_plan(),
            execute=execute,
            authorized_cost_usd=authorized_cost_usd,
            preconditions={
                "live_smoke_passed": confirm_live_smoke,
                "boundary_extension_human_approval": (
                    confirm_boundary_extension_approval
                ),
                "extension1_source_authenticated": confirm_source_authenticated,
                "remaining_balance_verified": confirm_remaining_balance,
            },
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"authorized {authorization.plan_name}: ${authorization.authorized_cost_usd}"
    )


@app.command("calibration")
def calibration_runner(
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    action: str | None = typer.Option(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
    execute: bool = typer.Option(False, "--authorize"),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_prerequisite_selected: bool = typer.Option(False),
    confirm_panel_frozen: bool = typer.Option(False),
    confirm_live_smoke: bool = typer.Option(False),
    confirm_human_approval: bool = typer.Option(False),
    confirm_remaining_balance: bool = typer.Option(False),
) -> None:
    """Print calibration gates or validate one evidence-dependent action."""

    resolved = load_pilot_config(config)
    if dry_run and not execute:
        typer.echo(calibration.preflight_text(resolved))
        return
    if action is None:
        raise typer.BadParameter("--action is required with --authorize")
    try:
        authorization = calibration.authorize_action(
            resolved,
            action=action,
            execute=execute,
            authorized_cost_usd=authorized_cost_usd,
            prerequisite_selected=confirm_prerequisite_selected,
            panel_frozen=confirm_panel_frozen,
            live_smoke_passed=confirm_live_smoke,
            human_approval=confirm_human_approval,
            remaining_balance_verified=confirm_remaining_balance,
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        f"authorized {authorization.plan_name}: ${authorization.authorized_cost_usd}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
