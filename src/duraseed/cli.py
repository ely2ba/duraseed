"""Side-effect-free next-gate preflights and explicit launch authorization."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from duraseed.config import load_pilot_config
from duraseed.calibration_billing import reconcile_calibration_billing
from duraseed.runners import RunnerGateError, authorize_launch
from duraseed.runners import boundary_extension as boundary
from duraseed.runners import calibration
from duraseed.runners import calibration_launch
from duraseed.runners import boundary_confirm_launch
from duraseed.runners import boundary_launch
from duraseed.runners import live_smoke as smoke


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("live-smoke")
def live_smoke(
    run_id: str | None = typer.Option(None),
    output_root: Path = typer.Option(Path("runs/live-smoke")),
    dry_run: bool = typer.Option(False, "--dry-run"),
    execute: bool = typer.Option(False, "--execute"),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_billing_reconciled: bool = typer.Option(False),
    confirm_human_launch: bool = typer.Option(False),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Preflight or execute the exact $25 live smoke gate."""

    if dry_run and execute:
        raise typer.BadParameter("--dry-run and --execute are mutually exclusive")
    if not execute:
        typer.echo(smoke.preflight_text())
        return
    if run_id is None:
        raise typer.BadParameter("--run-id is required with --execute")
    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required with --execute")
    try:
        authorization = smoke.authorize(
            execute=execute,
            authorized_cost_usd=authorized_cost_usd,
            billing_reconciled=confirm_billing_reconciled,
            human_approval=confirm_human_launch,
        )
        result = asyncio.run(
            smoke.run_remote(
                smoke.SmokeSettings(run_id=run_id, output_root=output_root),
                authorization,
                project_id=project_id,
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"live smoke artifacts: {result}")


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


@app.command("boundary-live")
def boundary_live(
    run_id: str = typer.Option(...),
    source_root: Path = typer.Option(
        Path("frozen/v0/runs/tinker-calibration/boundary")
    ),
    smoke_acceptance: Path = typer.Option(...),
    billing_reconciliation: Path = typer.Option(...),
    extension1_confirmation: Path = typer.Option(...),
    output_root: Path = typer.Option(Path("runs/boundary-extension")),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_human_launch: bool = typer.Option(False),
    refine_retry_trace: Path | None = typer.Option(None),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Execute the authenticated fixed $120 boundary continuation."""

    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required")
    try:
        authorization, _ = boundary_launch.authorize_boundary(
            authorized_cost_usd=authorized_cost_usd,
            smoke_acceptance=smoke_acceptance,
            billing_reconciliation=billing_reconciliation,
            human_approval=confirm_human_launch,
            project_id=project_id,
        )
        result = asyncio.run(
            boundary_launch.run_remote_boundary(
                authorization=authorization,
                project_id=project_id,
                run_id=run_id,
                source_root=source_root,
                output_root=output_root,
                config_path=config,
                extension1_confirmation_path=extension1_confirmation,
                refine_retry_trace=refine_retry_trace,
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"boundary artifacts: {result}")


@app.command("boundary-confirm-resume")
def boundary_confirm_resume(
    run_id: str = typer.Option(...),
    source_root: Path = typer.Option(
        Path("frozen/v0/runs/tinker-calibration/boundary")
    ),
    output_root: Path = typer.Option(Path("runs/boundary-extension")),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    confirm_human_launch: bool = typer.Option(False),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Continue the authenticated paused run at Extension-2 confirmation only."""

    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required")
    try:
        result = asyncio.run(
            boundary_confirm_launch.run_remote_boundary_confirmation_resume(
                project_id=project_id,
                run_id=run_id,
                source_root=source_root,
                output_root=output_root,
                config_path=config,
                human_approval=confirm_human_launch,
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"boundary artifacts: {result}")


@app.command("calibration")
def calibration_runner(
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    dry_run: bool = typer.Option(False, "--dry-run"),
    execute: bool = typer.Option(False, "--authorize"),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_panel_frozen: bool = typer.Option(False),
    confirm_live_smoke: bool = typer.Option(False),
    confirm_human_approval: bool = typer.Option(False),
    confirm_remaining_balance: bool = typer.Option(False),
) -> None:
    """Print calibration gates or authorize the one `$300` acquisition launch."""

    resolved = load_pilot_config(config)
    if dry_run and not execute:
        typer.echo(calibration.preflight_text(resolved))
        return
    try:
        authorization = calibration.authorize_calibration(
            resolved,
            execute=execute,
            authorized_cost_usd=authorized_cost_usd,
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


@app.command("calibration-live")
def calibration_live(
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
    billing_reconciliation: Path = typer.Option(...),
    raw_billing: Path = typer.Option(...),
    output_root: Path = typer.Option(Path("runs/acquisition-calibration")),
    config: Path = typer.Option(Path("duraseed_pilot_config.yaml")),
    authorized_cost_usd: str | None = typer.Option(None),
    confirm_human_launch: bool = typer.Option(False),
    restart_reconciliation: list[Path] = typer.Option([]),
    restart_raw_billing: list[Path] = typer.Option([]),
    project_id: str | None = typer.Option(None, envvar="TINKER_PROJECT_ID"),
) -> None:
    """Execute one authenticated `$300` acquisition-calibration launch."""

    if not project_id or not project_id.strip():
        raise typer.BadParameter(
            "an explicit --project-id/TINKER_PROJECT_ID is required"
        )
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise typer.BadParameter("TINKER_API_KEY is required")
    if len(restart_reconciliation) != len(restart_raw_billing):
        raise typer.BadParameter("restart reconciliation/raw billing counts differ")
    try:
        result = asyncio.run(
            calibration_launch.run_remote_calibration(
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
                billing_reconciliation_path=billing_reconciliation,
                raw_billing_path=raw_billing,
                project_id=project_id,
                authorized_cost_usd=authorized_cost_usd,
                human_approval=confirm_human_launch,
                restart_evidence=tuple(
                    zip(
                        restart_reconciliation,
                        restart_raw_billing,
                        strict=True,
                    )
                ),
            )
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"calibration artifacts: {result}")


@app.command("calibration-reconcile")
def calibration_reconcile(
    run_directory: Path = typer.Option(...),
    reconciliation: Path = typer.Option(...),
    raw_billing: Path = typer.Option(...),
) -> None:
    """Finalize post-run billing only after lag-cleared raw usage is present."""

    try:
        result = reconcile_calibration_billing(
            run_directory, reconciliation, raw_billing
        )
    except RunnerGateError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"calibration billing reconciled: {result['aggregate_billed_usd']}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
