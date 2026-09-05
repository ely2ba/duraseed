# Project status

_Last updated: 2026-09-05._

## Current

**Pilot 0 is complete and frozen.** Pair 1 (seed 11) and Pair 2 (seed 29)
both finished with `evidence_collected`; both 480-update Stage-B schedules and
final evaluations are complete. Their F1/F2 cells, F3 profiles, frozen matching
records, billing lineages, and local LoRA geometry archives passed the required
terminal checks.

The matching rule selected B-S@140 with B-G@30 at `31/96` for Pair 1, and
B-S@40 with B-G@20 at `17/96` for Pair 2. The complete public state is linked
from the [README](../README.md), including both readouts, figures, profiles,
paired offline analyses, geometry reports, and the dated
[prospective Pair-2 prediction record](results/pilot0-pair2-prediction.md).

The Pair-2 prediction was released from geometry before its F1/F2/F3 outcomes
were opened. Its two registered legs both held, producing the mechanical score
`PREDICTION: CONFIRMED`. Final test data remain sealed.

**The supervised trace-replay follow-up is now running.** Both acquisition
arms use supervised training on shared prompts: R-S learns solver traces and
R-P learns archived, verifier-correct Pilot-0 B-G traces. In the first source
block, R-S acquisition is complete and R-P acquisition is in progress. The
second block, fixed-rule matching, and eligible matched MAPS continuations
follow. There are no completed follow-up contrasts yet. The follow-up has a
separate frozen design; its aim is explained in the
[current overview](../README.md#what-were-testing-now). Pilot-0 settings and
evidence are unchanged. Earlier gauge work is dropped.

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
- Pair 2 completed both Stage-A schedules, matched B-S update 40 with B-G
  update 20, completed both Stage-B probes, and retained both F3 profiles and
  all 34 local Stage-A geometry archives.
- A prospective mechanical prediction was recorded at `2026-08-30T21:11:34Z`,
  before pair-2 outcome inspection. Before outcome release, B-S was selected
  mechanically from the Pair-2 geometry and the scoring rule was fixed. Both
  registered outcome legs held.

## Pair-2 terminal read order — completed

The required read order was completed:

1. First report completion, integrity and matching status plus selected checkpoint
   indices only, without F1/F2/F3 values, contrasts or directional language.
2. Then issue a separate geometry-first report for the selected checkpoints:
   A- and B-factor Frobenius norms, BA Frobenius norm, largest singular value,
   and stable rank. End by recording the prospective prediction that the
   larger-B-norm arm has faster F1 decay and faster early F2 gain. Stop until
   the human replies **PREDICTION RECORDED**.
3. Only after the human replied **PREDICTION RECORDED**, deliver the full Pair-2
   scientific readout and apply the registered score.

The archived record is [public](results/pilot0-pair2-prediction.md). No Pair-2
outcome was consulted to form the geometry-first prediction.

## Next

1. Preserve Pilot 0 as the frozen two-pair result.
2. Complete the authorized two-block supervised trace-replay comparison,
   retaining any unavailable matching outcome without replacement.
3. Report the follow-up alongside Pilot 0 and complete the manuscript. No
   additional seed, model, task, or gauge experiment is part of this package.

For the scientific overview, return to the [README](../README.md). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
