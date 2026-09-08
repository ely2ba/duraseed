# Project status

_Last updated: 2026-09-08._

## Current

**The experimental phase is complete.** The study comprises two Pilot pairs,
four replay acquisition runs, and the only matched replay continuation pair,
in block 11. The paper is being revised; it has not been posted or submitted.
No additional experiments are planned or authorized.

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

**The bounded trace-replay package is complete.** All four acquisition runs and
all twelve candidate assessments finished. The selected seed-11 checkpoints,
R-S@220 and R-P@20, both completed the unchanged 480-update Stage-B recipe:
960 updates, 48 evaluation panels, and 246,784 completions. Seed 29 remains
`NO_MATCH` and received no Stage-B training. The saved readout reproduces exactly;
panel evidence, checkpoint/optimizer continuity, and all 1,100 logical call
records passed terminal checks, with no pending request. The completed launchd
job is unloaded and its monitor deleted. Original evidence and Pilot-0 settings are unchanged; gauge
work is dropped.

The [follow-up results and reproducible figures](../artifacts/replay-v1/followup/README.md)
report R-P minus R-S targeted retention AUC0–20 of +0.061263
(conditional paired-item 95% interval [0.042904, 0.079395]). This is not uniformly
slower forgetting: R-P starts higher, crosses its own half-baseline earlier,
and rebounds at update 10. Absolute MAPS AUC0–480 instead favors R-S; endpoints
are similar. The [detailed readout](../artifacts/replay-v1/followup/readout.md)
and [supplement](../artifacts/replay-v1/followup/supplement.md) report the
trajectories, starting-score sensitivity, and unavailable block.

Local accounting totals $341.504818818 in observed-token costs plus $35.25 in
storage allowances ($376.754818818 combined), not settled provider invoices.
Final provider attribution, external account commitments, and current reserve
remain unverified. The 20 Stage-B checkpoint pairs record a 30-day TTL; absolute
expiry timestamps and current provider retention were not independently checked.

The original matching design unnecessarily required agreement with historical
Pilot-0 scores in addition to agreement between the new arms. We changed the
design to remove the historical-score requirement.

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

## Manuscript preparation and closeout

1. Preserve Pilot 0 as the frozen two-pair result and retain the completed replay records.
2. Continue manuscript revision without publishing the paper. Any later posting
   or submission requires owner authorization; neither has occurred.
3. Finish provider invoice attribution during end-of-project accounting. No
   additional seed, model, task, or gauge experiment is part of this package.

For the scientific overview, return to the [README](../README.md#results). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
