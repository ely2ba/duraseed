# Project status

_Last updated: 2026-08-28._

## Current

Pilot 0 pair 1 uses seed 11 and is running under its separate `$772.40`
authorization. Both Stage-A trajectories are complete. The frozen overlap rule
selected the real B-S update-140 checkpoint and B-G update-30 checkpoint, and
both full pre-Stage-B F3 profiles are durable. Stage B is now running on B-S:
five of 480 updates are committed, the update-5 MAPS evaluation is complete,
and its retention evaluation is in progress. B-G follows under the identical
frozen Stage-B schedule.

Pair 2 remains unlaunched. It requires durable pair-1 F1/F2 cells, F3 profiles,
a matching record, billing lineage, and a separate human authorization.

**Pair-2 launch commitment (2026-08-23, before any pair-1 scientific result
was read):** Pair-2 launch authorization depends only on: (a) pair-1 terminal
with durable artifacts and clean integrity/billing lineage, (b) no
infrastructure or protocol-execution failure, and (c) lifetime spend plus the
pair-2 preflight remaining within accepted ceilings. It does not depend on the
direction, size, or significance of any pair-1 scientific contrast; a
matching-unavailable outcome does not block pair 2.

No retention or future-learning contrast is claimed yet. Final test data remain
sealed.

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
- Pair 1 completed both Stage-A schedules without scientific or infrastructure
  failure; all 34 retained cadence checkpoints are archived locally.
- Matching selected B-S update 140 with B-G update 30, both selected F3 profiles
  completed, and the common Stage-B probe began.

## Next

1. Finish pair 1 Stage B and inspect its durable F1/F2/F3 evidence.
2. Launch pair 2 only after its separate authorization.
3. Use both pairs to assess feasibility and effect direction, with a first,
   crude look at seed-to-seed variability.
4. Extend to fresh paired seeds before powering and preregistering confirmation.

For the scientific overview, return to the [README](../README.md). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
