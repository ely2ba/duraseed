from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from duraseed.data.sealing import (
    ExecutionContext,
    FinalTestAccessDenied,
    SealError,
    inspect_seal,
    open_final_test,
    seal_file,
)


KEY = bytes(range(32))


def _sealed_fixture(tmp_path: Path) -> tuple[Path, bytes]:
    plaintext = b'{"split":"a_test_single","task_id":"fixture"}\n'
    source = tmp_path / "final.jsonl"
    sealed = tmp_path / "final.sealed"
    source.write_bytes(plaintext)
    seal_file(source, sealed, key=KEY, declared_split="a_test_single")
    return sealed, plaintext


def test_seal_inspect_and_open_final_test(tmp_path: Path) -> None:
    sealed, plaintext = _sealed_fixture(tmp_path)

    info = inspect_seal(sealed)

    assert info.declared_split == "a_test_single"
    assert len(info.plaintext_sha256) == 64
    assert (
        open_final_test(
            sealed,
            key=KEY,
            command=ExecutionContext.FINAL_EVALUATION,
            expected_split="a_test_single",
        )
        == plaintext
    )


def test_wrong_key_is_rejected(tmp_path: Path) -> None:
    sealed, _ = _sealed_fixture(tmp_path)

    with pytest.raises(SealError, match="authentication failed"):
        open_final_test(
            sealed,
            key=b"x" * 32,
            command=ExecutionContext.FINAL_EVALUATION,
        )


def test_tampered_ciphertext_is_rejected(tmp_path: Path) -> None:
    sealed, _ = _sealed_fixture(tmp_path)
    envelope = json.loads(sealed.read_bytes())
    ciphertext = bytearray(base64.b64decode(envelope["ciphertext"]))
    ciphertext[0] ^= 1
    envelope["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
    sealed.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(SealError, match="authentication failed"):
        open_final_test(
            sealed,
            key=KEY,
            command=ExecutionContext.FINAL_EVALUATION,
        )


def test_wrong_declared_split_is_rejected(tmp_path: Path) -> None:
    sealed, _ = _sealed_fixture(tmp_path)

    with pytest.raises(SealError, match="expected 'a_test_ood'"):
        open_final_test(
            sealed,
            key=KEY,
            command=ExecutionContext.FINAL_EVALUATION,
            expected_split="a_test_ood",
        )


def test_stage_b_final_split_is_sealed_and_guarded(tmp_path: Path) -> None:
    source = tmp_path / "b-test.jsonl"
    sealed = tmp_path / "b-test.sealed"
    source.write_bytes(b'{"split":"b_test","task_id":"fixture"}\n')

    seal_file(source, sealed, key=KEY, declared_split="b_test")

    assert inspect_seal(sealed).declared_split == "b_test"
    with pytest.raises(FinalTestAccessDenied):
        open_final_test(sealed, key=KEY, command=ExecutionContext.SELECTION)


@pytest.mark.parametrize(
    "command", [ExecutionContext.SELECTION, ExecutionContext.TRAINING]
)
def test_selection_and_training_access_are_denied(
    tmp_path: Path, command: ExecutionContext
) -> None:
    sealed, _ = _sealed_fixture(tmp_path)

    with pytest.raises(FinalTestAccessDenied):
        open_final_test(sealed, key=KEY, command=command)
