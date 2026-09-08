"""The local CLI cannot turn a check or incomplete preflight into spending."""

import json

import pytest

from duraseed import replay_cli
from duraseed.provenance import sha256_bytes


def fail_if_called(*args, **kwargs):
    pytest.fail("a local or unauthorized path attempted paid execution")


def test_check_never_approves_or_creates_remote(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--repo",
            str(tmp_path),
            "--config",
            str(tmp_path / "config.json"),
            "--check",
        ],
    )
    monkeypatch.setattr(replay_cli, "load_config", lambda *a: {})
    monkeypatch.setattr(replay_cli, "block_inputs", lambda *a: None)
    monkeypatch.setattr(replay_cli, "approve", fail_if_called)
    monkeypatch.setattr(replay_cli, "ReplayRemote", fail_if_called)
    replay_cli.main()
    assert "no service was created" in capsys.readouterr().out


def test_unresolved_liability_blocks_even_exact_approval(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    preflight = {
        "approval_ceiling_usd": "2125.11",
        "blockers": ["BACKUP_LIABILITY_UNVERIFIED"],
        "all_in_maximum_liability_usd": None,
    }
    (tmp_path / "preflight.json").write_text(json.dumps(preflight))
    config = {
        "protocol": {"sha256": "sha256:protocol"},
        "preflight": {"path": "preflight.json"},
    }
    with pytest.raises(ValueError, match="explicit launch approval required"):
        replay_cli.approve(tmp_path, config_path, config, "")
    phrase = f"AUTHORIZE DURASEED REPLAY sha256:protocol {sha256_bytes(config_path.read_bytes())} CEILING $2125.11"
    monkeypatch.setattr(replay_cli.subprocess, "check_output", fail_if_called)
    with pytest.raises(ValueError, match="unresolved service/billing"):
        replay_cli.approve(tmp_path, config_path, config, phrase)
