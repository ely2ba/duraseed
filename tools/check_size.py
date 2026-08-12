#!/usr/bin/env python3
"""Enforce DuraSeed's fresh-code limits and immutable carry-over hashes."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "duraseed"
TESTS = ROOT / "tests"
TOOLS = ROOT / "tools"
MANIFEST = ROOT / "provenance" / "immutable-carryover.sha256"
LEGACY_ALLOWLIST = TOOLS / "size_allowlist.txt"

MAX_FILE_LINES = 400
MAX_RUNNER_LINES = 300
MAX_RUNTIME_LINES = 1_200
MAX_DEPTH = 3
ENTRY = re.compile(r"^([0-9a-f]{64})  ([^\t\r\n]+)$")


class GuardrailError(ValueError):
    """Raised when immutable carry-over metadata is invalid."""


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible_prefix(relative: str) -> bool:
    return relative.startswith("src/duraseed/") or relative.startswith("tests/")


def _ineligible(relative: str) -> bool:
    return (
        relative == "src/duraseed/config.py"
        or relative.startswith("src/duraseed/runtime/")
        or relative.startswith("src/duraseed/runners/")
        or relative.startswith("tools/")
    )


def load_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        return {}
    entries: dict[str, str] = {}
    ordered_paths: list[str] = []
    root = ROOT.resolve()
    for number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        match = ENTRY.fullmatch(raw)
        if match is None:
            raise GuardrailError(
                f"{MANIFEST.relative_to(ROOT)}:{number}: expected "
                "'<64 lowercase hex><two spaces><POSIX relative path>'"
            )
        expected, relative = match.groups()
        posix = PurePosixPath(relative)
        if (
            relative != relative.strip()
            or relative != posix.as_posix()
            or posix.is_absolute()
            or not posix.parts
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise GuardrailError(f"manifest line {number}: unsafe path {relative!r}")
        if relative in entries:
            raise GuardrailError(f"manifest line {number}: duplicate path {relative!r}")
        if not _eligible_prefix(relative) or _ineligible(relative):
            raise GuardrailError(
                f"manifest line {number}: ineligible path {relative!r}"
            )
        target = ROOT / relative
        if target.is_symlink() or not target.is_file():
            raise GuardrailError(
                f"manifest line {number}: missing regular file {relative!r}"
            )
        try:
            target.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise GuardrailError(
                f"manifest line {number}: path escapes repository {relative!r}"
            ) from error
        actual = file_hash(target)
        if actual != expected:
            raise GuardrailError(
                f"manifest line {number}: SHA-256 mismatch for {relative!r}: "
                f"expected {expected}, got {actual}"
            )
        entries[relative] = expected
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise GuardrailError("carry-over manifest paths must be sorted")
    return entries


def python_files() -> list[Path]:
    roots = (SRC, TESTS, TOOLS)
    return sorted(
        path for root in roots if root.exists() for path in root.rglob("*.py")
    )


def check() -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    if LEGACY_ALLOWLIST.exists():
        errors.append(
            "tools/size_allowlist.txt is forbidden; use immutable carry-over hashes"
        )
    try:
        carried = load_manifest()
    except GuardrailError as error:
        return [str(error)], {}

    totals = {
        "fresh_files": 0,
        "fresh_lines": 0,
        "fresh_tool_files": 0,
        "fresh_tool_lines": 0,
        "carried_files": len(carried),
        "carried_python_files": 0,
        "carried_python_lines": 0,
        "runtime_lines": 0,
    }
    for path in python_files():
        relative = path.relative_to(ROOT).as_posix()
        lines = line_count(path)
        if relative in carried:
            totals["carried_python_files"] += 1
            totals["carried_python_lines"] += lines
            continue

        if path.is_relative_to(TOOLS):
            totals["fresh_tool_files"] += 1
            totals["fresh_tool_lines"] += lines
        else:
            totals["fresh_files"] += 1
            totals["fresh_lines"] += lines

        is_runner = path.is_relative_to(SRC / "runners")
        limit = MAX_RUNNER_LINES if is_runner else MAX_FILE_LINES
        if lines > limit:
            errors.append(f"{relative}: {lines} lines exceeds fresh-code cap {limit}")
        if path.is_relative_to(SRC):
            depth = len(path.relative_to(SRC).parts) - 1
            if depth > MAX_DEPTH:
                errors.append(
                    f"{relative}: depth {depth} exceeds fresh-code cap {MAX_DEPTH}"
                )
        if path.is_relative_to(SRC / "runtime"):
            totals["runtime_lines"] += lines

    if totals["runtime_lines"] > MAX_RUNTIME_LINES:
        errors.append(
            f"src/duraseed/runtime/: {totals['runtime_lines']} lines exceeds "
            f"fresh-code cap {MAX_RUNTIME_LINES}"
        )
    return errors, totals


def main() -> int:
    if len(sys.argv) != 1:
        print("size guardrail accepts no command-line options", file=sys.stderr)
        return 2
    errors, totals = check()
    if errors:
        print("size guardrail FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(
        "size guardrail OK: "
        f"fresh source/test {totals['fresh_files']} files / {totals['fresh_lines']} lines; "
        f"fresh tools {totals['fresh_tool_files']} files / "
        f"{totals['fresh_tool_lines']} lines; "
        f"carry-over {totals['carried_files']} files "
        f"({totals['carried_python_files']} Python / "
        f"{totals['carried_python_lines']} lines); "
        f"runtime {totals['runtime_lines']}/{MAX_RUNTIME_LINES} lines"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
