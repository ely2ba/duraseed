"""Repository identity guards shared by paid launchers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from duraseed.runners import RunnerGateError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def require_clean_worktree(*, gate_name: str) -> None:
    """Refuse paid execution when recorded HEAD would omit local changes."""

    try:
        status = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise RunnerGateError(
            f"{gate_name} cannot authenticate the git worktree"
        ) from error
    if status:
        raise RunnerGateError(f"{gate_name} requires a clean git worktree")


__all__ = ["require_clean_worktree"]
