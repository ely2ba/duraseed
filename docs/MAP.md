# Repository map

This is the starting point for readers and contributors. The public scientific contract is [`PROTOCOL.md`](../PROTOCOL.md); current progress is in [`STATUS.md`](STATUS.md).

## Components

- `src/duraseed/tasks/`: immutable TCES and MAPS task definitions, generators, parsers, solvers, and exact verifiers.
- `src/duraseed/data/`: manifests, deterministic splits, leakage and sealing checks, boundary reducers, panel construction, and matching.
- `src/duraseed/training/`: verified datum/reward builders, Stage-A calibration reducers, and preserved retired warm-start reducers.
- `src/duraseed/evaluation/analysis.py`: item-level acquisition, endpoint and AUC retention, transfer, and frontier estimands.
- `src/duraseed/runtime/`: small shared Tinker boundary for clients, sampling, data conversion, checkpoints, spend accounting, and billing.
- `src/duraseed/runners/`: thin orchestration for the next boundary and calibration gates; scientific rules remain in the modules above.
- `duraseed_pilot_config.yaml`: machine-readable frozen configuration.
- `tools/`: size/hash enforcement and local replay equivalence.

## Scientific data flow

1. **M0:** select one common format-capable adapter origin for all methods.
2. **Panels:** scan TCES families, confirm boundary difficulty on held-out items, and freeze equal crossed target/sentinel panels. The accepted local amendment collapses the fixed 49 candidates into 37 exact algebraic representatives, applies the unchanged 22/22 single-family gate before panel matching, and selects the separate intermediate pool from ordinary-capacity representatives outside both panels. The current design prefers 12/12, but strict separation, adequate breadth and precision, and no use of method outcomes are the invariants.
3. **Stage A:** branch B-S and B-G weights-only from the same M0 with fresh optimizers. B-S uses fixed verified solver-teacher SFT; B-G uses boundary-weighted verifier-scored group-relative RL. Compare both at fixed budgets and later at matched targeted-panel capability, without claiming their data or resource use is identical. Plot capability against cumulative cost and the relevant token axes, and retain the separate teacher-example, prefill, sample, training-token, update, and storage resource vector.
4. **Matching:** after fixed-budget Pilot 0, select real checkpoints matched specifically on targeted-panel exact-success and publish the full descriptive pre-B profile rather than claiming global behavioral equivalence.
5. **Stage B:** apply the common frozen supervised MAPS probe (`shortest2_cap2`, `3e-4`, step 480) to every selected Stage-A origin.
6. **Frontier:** combine full-`a_validation` Stage-A fixed-budget retention, secondary `a_monitor` RetentionAUC, Stage-B raw gain AUC, and specificity into the stability–future-learnability comparison.

No later stage may rescan an earlier task distribution using method outcomes. Pilot and confirmatory generations use explicit seeds and authenticated manifests.

The pre-B profile includes target, sentinel, and per-family accuracy, the Cover curve, invalid-output rate, completion length, supported Pass@k, verified strategy diversity, and token surprisal. Pass@k is emitted only for `k` no larger than the collected independent draw count. These quantities describe residual differences; they are not additional matching gates.

After the two-seed Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds for variance reconnaissance. If a reproducible difference appears, prioritize supervised replay of a frozen verifier-correct subset of B-G rollouts before broad method expansion. B-O remains conditional and G-U optional; the duplicate G-B and teacher-allocation R-G arms are retired. Only after the mechanism decision determines the confirmatory matrix do we freeze and power the design, preregister it, and run fresh confirmatory seeds. A same-model higher-rank or full-weight replication is conditional on a result worth following up and tests whether it is specific to rank-32 LoRA.

## Evidence and provenance

- `frozen/v0/`: compact calibration and replay evidence carried from v0; runtime code never writes here.
- `provenance/v0-artifacts.sha256`: hashes for byte-identical preserved artifacts.
- `provenance/public-projections.sha256`: hashes for path-free public views of three privately preserved source records.
- `provenance/immutable-carryover.sha256`: hashes for immutable scientific source and tests.
- `provenance/replay-equivalence.md`: exact v0 → v1 verifier, boundary, MAPS, and analysis replay.
- `provenance/v0-map.md`: archive, checkpoint, billing, ratification, and public-projection lineage.

Generated top-level `runs/`, `artifacts/`, local datasets, credentials, and private research notes are excluded from Git.
