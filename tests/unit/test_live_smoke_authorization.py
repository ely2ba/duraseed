from pathlib import Path

import pytest
from typer.testing import CliRunner

from duraseed.cli import app
from duraseed.live_smoke_gate import authorize
from duraseed.runners import RunnerGateError


def test_cli_preflight_is_credential_free_and_execute_is_exactly_gated(
    monkeypatch,
) -> None:
    runner = CliRunner()
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_PROJECT_ID", raising=False)
    result = runner.invoke(app, ["live-smoke"])
    assert result.exit_code == 0
    assert "Remote cost cap: $25" in result.stdout
    blocked = runner.invoke(
        app,
        ["live-smoke", "--execute", "--run-id", "live", "--authorized-cost-usd", "25"],
    )
    assert blocked.exit_code != 0
    assert "project-id" in blocked.output


def test_cli_execute_dispatches_real_runner_only_after_all_gates(
    monkeypatch, tmp_path: Path
) -> None:
    called: dict[str, object] = {}

    async def fake_remote(settings, authorization, *, project_id):  # type: ignore[no-untyped-def]
        called.update(
            settings=settings, authorization=authorization, project_id=project_id
        )
        return tmp_path / "remote-run"

    monkeypatch.setenv("TINKER_API_KEY", "secret-for-test")
    monkeypatch.setattr("duraseed.cli.smoke.run_remote", fake_remote)
    result = CliRunner().invoke(
        app,
        [
            "live-smoke",
            "--execute",
            "--run-id",
            "live",
            "--authorized-cost-usd",
            "25",
            "--confirm-billing-reconciled",
            "--confirm-human-launch",
            "--project-id",
            "project-1",
        ],
    )
    assert result.exit_code == 0
    assert called["project_id"] == "project-1"
    assert str(getattr(called["authorization"], "authorized_cost_usd")) == "25"


def test_authorization_requires_exact_cap_and_both_confirmations() -> None:
    with pytest.raises(RunnerGateError, match=r"equal \$25"):
        authorize(
            execute=True,
            authorized_cost_usd="24.99",
            billing_reconciled=True,
            human_approval=True,
        )
    with pytest.raises(RunnerGateError, match="billing_reconciled"):
        authorize(
            execute=True,
            authorized_cost_usd="25",
            billing_reconciled=False,
            human_approval=True,
        )
    granted = authorize(
        execute=True,
        authorized_cost_usd="25",
        billing_reconciled=True,
        human_approval=True,
    )
    assert str(granted.authorized_cost_usd) == "25"
