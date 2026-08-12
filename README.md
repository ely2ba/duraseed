# DuraSeed

DuraSeed studies the stability–plasticity trade-off induced by different post-training acquisition paths: after methods reach matched capability on one task family, do they retain it differently while learning the next one? The study uses deterministic, exact-verifier TCES and MAPS environments with `Qwen/Qwen3.5-9B-Base`, rank-32 LoRA adapters, and Tinker for explicitly authorized remote execution.

**Status:** Calibration and the v1 rebuild are complete through the next-gate runners; boundary extension and Stage-A calibration remain pending, and Pilot 0 has not started.

## Repository layout

Start with [`docs/MAP.md`](docs/MAP.md) for the component map, scientific data flow, and provenance locations. Current gate status is tracked in [`docs/STATUS.md`](docs/STATUS.md).

- `src/duraseed/tasks/`: TCES and MAPS generators, parsers, solvers, and exact verifiers.
- `src/duraseed/data/`: split, manifest, leakage, boundary, panel, and matching logic.
- `src/duraseed/training/`: verified-data builders and calibration decision reducers.
- `src/duraseed/evaluation/`: item-level transition and frontier analysis.
- `src/duraseed/runtime/`: bounded Tinker client, sampling, checkpoint, ledger, and billing code.
- `src/duraseed/runners/`: orchestration for the next authorized experimental gates.
- `frozen/v0/` and `provenance/`: preserved calibration evidence, hashes, and replay records.
- `PROTOCOL.md`: public scientific contract and calibration record.

## Local checks

Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
python tools/check_size.py
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
uv run duraseed validate
```

These checks are local and credential-free. Remote Tinker access requires a separate, explicit launch authorization and spend cap.

## Research scope

The design compares acquisition paths under one base model, one adapter rank, synthetic exact tasks, and one frozen supervised Stage-B probe. Any eventual inference is conditional on those choices. Calibration outcomes and engineering checks are not results for the stability–plasticity hypothesis.

## License

[Apache License 2.0](LICENSE)
