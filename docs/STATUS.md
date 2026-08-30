# Project status

_Last updated: 2026-08-30._

## Current

Pilot 0 pair 1 (seed 11) completed at `2026-08-30T19:18:10Z` with
`evidence_collected` and exit code 0. Both 480-update Stage-B schedules and final
evaluations are complete; the F1/F2 cells, F3 profiles, matching record, billing
lineage, and all 34 local LoRA archives passed terminal verification.

Pair 2 (seed 29) is separately authorized at its exact `$772.40` preflight
ceiling and is being prepared for launch.

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
- Matching selected B-S update 140 with B-G update 30, both selected F3 profiles
  completed, and the common Stage-B probe began.

## Next

1. Launch the separately authorized pair 2 under the unchanged scientific recipe.
2. After launch is confirmed running, report pair-1 F1/F2/F3 from local artifacts.
3. Use both pairs to assess feasibility and effect direction, with a first,
   crude look at seed-to-seed variability.
4. Extend to fresh paired seeds before powering and preregistering confirmation.

For the scientific overview, return to the [README](../README.md). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
