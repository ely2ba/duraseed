# Repository map

This is the starting point for readers and contributors. The public scientific contract is [`PROTOCOL.md`](../PROTOCOL.md); current progress is in [`STATUS.md`](STATUS.md).

## Components

- `src/duraseed/tasks/`: immutable TCES and MAPS task definitions, generators, parsers, solvers, and exact verifiers.
- `src/duraseed/data/`: manifests, deterministic splits, leakage and sealing checks, boundary reducers, panel construction, and matching.
- `src/duraseed/training/`: verified datum/reward builders plus teacher-dose, allocation, and Stage-A calibration reducers.
- `src/duraseed/evaluation/analysis.py`: item-level acquisition, retention, transfer, and frontier estimands.
- `src/duraseed/runtime/`: small shared Tinker boundary for clients, sampling, data conversion, checkpoints, spend accounting, and billing.
- `src/duraseed/runners/`: thin orchestration for the next boundary and calibration gates; scientific rules remain in the modules above.
- `duraseed_pilot_config.yaml`: machine-readable frozen configuration.
- `tools/`: size/hash enforcement and local replay equivalence.

## Scientific data flow

1. **M0:** select one common format-capable adapter origin for all methods.
2. **Panels:** scan TCES families, confirm boundary difficulty on held-out items, and freeze crossed target/sentinel panels.
3. **Stage A:** acquire TCES capability through the compared post-training paths under matched information and update budgets.
4. **Stage B:** apply the common frozen supervised MAPS probe (`shortest2_cap2`, `3e-4`, step 480) to every selected Stage-A origin.
5. **Frontier:** combine Stage-A retention, Stage-B raw gain AUC, and specificity into the stability–future-learnability comparison.

No later stage may rescan an earlier task distribution using method outcomes. Pilot and confirmatory generations use explicit seeds and authenticated manifests.

## Evidence and provenance

- `frozen/v0/`: compact calibration and replay evidence carried from v0; runtime code never writes here.
- `provenance/v0-artifacts.sha256`: hashes for byte-identical preserved artifacts.
- `provenance/public-projections.sha256`: hashes for path-free public views of three privately preserved source records.
- `provenance/immutable-carryover.sha256`: hashes for immutable scientific source and tests.
- `provenance/replay-equivalence.md`: exact v0 → v1 verifier, boundary, MAPS, and analysis replay.
- `provenance/v0-map.md`: archive, checkpoint, billing, ratification, and public-projection lineage.

Generated top-level `runs/`, `artifacts/`, local datasets, credentials, and private research notes are excluded from Git.
