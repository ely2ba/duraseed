from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "tools" / "check_size.py"


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "src" / "duraseed").mkdir(parents=True)
    shutil.copyfile(SOURCE, repo / "tools" / "check_size.py")
    return repo


def run(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(repo / "tools" / "check_size.py"), *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def hash_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(repo: Path, lines: list[str]) -> None:
    destination = repo / "provenance" / "immutable-carryover.sha256"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")


def test_absent_manifest_means_no_exemptions(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "src" / "duraseed" / "small.py").write_text("value = 1\n")
    result = run(repo)
    assert result.returncode == 0
    assert "fresh source/test 1 files / 1 lines" in result.stdout
    assert "carry-over 0 files" in result.stdout


def test_over_limit_unlisted_file_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "src" / "duraseed" / "large.py").write_text("x = 1\n" * 401)
    result = run(repo)
    assert result.returncode == 1
    assert "401 lines exceeds fresh-code cap 400" in result.stdout


def test_exact_manifest_hash_exempts_immutable_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    carried = repo / "src" / "duraseed" / "legacy.py"
    carried.write_text("x = 1\n" * 401)
    write_manifest(repo, [f"{hash_of(carried)}  src/duraseed/legacy.py"])
    result = run(repo)
    assert result.returncode == 0
    assert "carry-over 1 files (1 Python / 401 lines)" in result.stdout


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("bad  src/duraseed/x.py", "expected '<64 lowercase hex>"),
        (f"{'0' * 64}  ../x.py", "unsafe path"),
        (f"{'0' * 64}  tools/x.py", "ineligible path"),
        (f"{'0' * 64}  src/duraseed/runtime/x.py", "ineligible path"),
        (f"{'0' * 64}  src/duraseed/config.py", "ineligible path"),
        (f"{'0' * 64}  docs/x.md", "ineligible path"),
    ],
)
def test_invalid_manifest_entry_fails(tmp_path: Path, line: str, message: str) -> None:
    repo = make_repo(tmp_path)
    write_manifest(repo, [line])
    result = run(repo)
    assert result.returncode == 1
    assert message in result.stdout


def test_manifest_rejects_missing_hash_mismatch_duplicate_and_unsorted(
    tmp_path: Path,
) -> None:
    repo = make_repo(tmp_path)
    first = repo / "src" / "duraseed" / "a.py"
    second = repo / "src" / "duraseed" / "b.py"
    first.write_text("a = 1\n")
    second.write_text("b = 1\n")
    cases = [
        ([f"{'0' * 64}  src/duraseed/missing.py"], "missing regular file"),
        ([f"{'0' * 64}  src/duraseed/a.py"], "SHA-256 mismatch"),
        (
            [
                f"{hash_of(first)}  src/duraseed/a.py",
                f"{hash_of(first)}  src/duraseed/a.py",
            ],
            "duplicate path",
        ),
        (
            [
                f"{hash_of(second)}  src/duraseed/b.py",
                f"{hash_of(first)}  src/duraseed/a.py",
            ],
            "must be sorted",
        ),
    ]
    for lines, expected in cases:
        write_manifest(repo, lines)
        result = run(repo)
        assert result.returncode == 1
        assert expected in result.stdout


def test_legacy_allowlist_and_cli_arguments_are_rejected(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    (repo / "tools" / "size_allowlist.txt").write_text("")
    assert "is forbidden" in run(repo).stdout
    argument_result = run(repo, "--skip")
    assert argument_result.returncode == 2
    assert "accepts no command-line options" in argument_result.stderr
