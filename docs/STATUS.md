# Project status

_Last updated: 2026-08-30._

## Current

Pilot 0 pair 1 (seed 11) completed at `2026-08-30T19:18:10Z` with
`evidence_collected` and exit code 0. Both 480-update Stage-B schedules and final
evaluations are complete; the F1/F2 cells, F3 profiles, matching record, billing
lineage, and all 34 local LoRA archives passed terminal verification.

Pair 2 (seed 29) is running under launchd as
`pilot0-pair2-seed29-20260830T204211Z`, confirmed at `2026-08-30T20:45:22Z`,
with its exact `$772.40` preflight ceiling and the existing heartbeat monitoring.
The local launchd check at `2026-08-30T22:02:53Z` confirmed the same job running;
that status check did not read pair-2 outcome artifacts.

The [pair-1 scientific results](results/pilot0-pair1.md) are available, including
F1 retention, F2 absolute and baseline-relative learning, both F3 profiles, and
the descriptive, post-hoc offline analyses. This remains one paired seed.

**Pair-2 launch commitment (2026-08-23, before any pair-1 scientific result
was read):** Pair-2 launch authorization depends only on: (a) pair-1 terminal
with durable artifacts and clean integrity/billing lineage, (b) no
infrastructure or protocol-execution failure, and (c) lifetime spend plus the
pair-2 preflight remaining within accepted ceilings. It does not depend on the
direction, size, or significance of any pair-1 scientific contrast; a
matching-unavailable outcome does not block pair 2.

**Human authorization clarification (2026-08-30):** Values were machine-read
and recomputed during terminal verification; no contrast was reported to or
consulted by the human before authorization; pair-2 matching is mechanically
blind to pair-1 contrasts. The documented, authorized infrastructure recovery
that ended in full verified evidence satisfies condition (b) and is not a
launch blocker.

As authorized on 2026-08-30, account-wide lifetime formulas are retired: launch
requires the sum of DuraSeed run-ledger actuals plus pair-2's exact preflight
ceiling to be within `$1,548.08`, with the saved `$3,964.56` grant balance as a
separate sanity check; CSV attribution is deferred to end-of-project accounting.

Final test data remain sealed.

## Completed foundations

- The common format-capable M0 origin is frozen.
- TCES and MAPS generators, solvers, and exact verifiers are frozen.
- Two disjoint matched 12-family panels and a separate intermediate pool are
  authenticated; target and sentinel roles cross across paired seeds.
- The MAPS Stage-B recipe is frozen at `shortest2_cap2`, LR `3e-4`, and 480
  updates.
- Replay equivalence and the live engineering smoke passed.
- The task-specific teacher warm start was retired before Pilot outcomes, and
  both methods now branch directly from M0.
- The Stage-A answer-tag contract was corrected before Pilot outcomes while the
  original screen evidence was retained.
- The bounded B-S capability-dose run reached the planned overlap zone at
  update 90, passed the frozen viability gates, and validated trainable-state
  save followed by weights-only restore.
- Pilot pair execution, cadence state checkpointing, post-hoc matching, F3
  profiles, Stage B, billing gates, and no-reroll failure handling pass locally.
- Pair 1 completed both Stage-A schedules, with a documented authorized
  infrastructure recovery; all 34 retained cadence checkpoints are archived locally.
- Matching selected B-S update 140 with B-G update 30; both selected F3 profiles,
  both 480-update Stage-B probes, and their final evaluations are complete.
- The pair-1 local F1/F2/F3 readout was produced only after pair 2 was confirmed
  running. The [public results](results/pilot0-pair1.md) include the observed
  trajectories, conditional item-level uncertainty, and descriptive profiles.
- Supplementary, post-hoc adapter geometry covers the 34 retained pair-1
  Stage-A checkpoints. It is outside F3 and all matching or gate decisions.
- A prospective mechanical prediction was recorded at `2026-08-30T21:11:34Z`,
  before pair-2 outcome inspection. The prediction and subsequent gauge-rescaling
  candidate are [non-binding proposals](../README.md#what-comes-next), not changes
  to the running experiment or authorization for a follow-up.

## Pair-2 terminal read order

The authorized read order is unchanged:

1. First report completion, integrity and matching status plus selected checkpoint
   indices only, without F1/F2/F3 values, contrasts or directional language.
2. Then issue a separate geometry-first report for the selected checkpoints:
   A- and B-factor Frobenius norms, BA Frobenius norm, largest singular value,
   and stable rank. End by recording the prospective prediction that the
   larger-B-norm arm has faster F1 decay and faster early F2 gain. Stop until
   the human replies **PREDICTION RECORDED**.
3. Only after that reply deliver the full pair-2 scientific readout.

Checkpoint archival continues before expiry. Machine integrity checks may
recompute values internally, but scientific outcomes are not consulted to form
the geometry-first prediction. This documentation update read no pair-2
scientific outcomes or contrasts.

## Next

1. Complete pair 2 under the unchanged scientific recipe and verify its durable evidence.
2. After the terminal read order and human release above, use both pairs to
   assess feasibility and effect direction, with a first, crude look at
   seed-to-seed variability.
3. Seek separate authorization for fresh paired seeds before powering and
   preregistering confirmation; no further launch is authorized by this list.

For the scientific overview, return to the [README](../README.md). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
