# Project status

_Last updated: 2026-09-10._

## Current

**The final endpoint-clone comparison is running: RL teacher acquisition.**
The owner approved the revised [specification](experiments/endpoint-clone.md)
on 2026-09-10, including the revised continuation matrix and use of the old
reserve. Launch was confirmed at 16:39 UTC on 2026-09-10: M0 restored, initial
sampler creation pending, and no teacher update committed at that check. The
exact full-cap preflight is **$2,148.44**, $1.44 above the approximate planning
estimate because the final preflight includes the complete fixed schedule.
No endpoint-clone outcome exists yet. The experiment uses a fresh
30-update RL teacher, unfiltered endpoint samples, arithmetic-only clone
selection and independent-item confirmation, then two teacher and two student
continuations through update 20 if confirmation passes. One predetermined run
per checkpoint continues to update 480. The primary analysis compares dense
cross-checkpoint trajectory distances with within-checkpoint repeat distances;
it carries no binary equivalence score.

**Completed evidence remains frozen:** two Pilot pairs, the original four
replay acquisition runs, the matched block-11 replay continuation, a dense
retention repeat from those same origins, and a separate acquisition-order
replication on block 11's existing replay corpus. The paper is being revised;
it has not been posted or submitted.

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

## Completed September 9–10 follow-ups

- **Stored update-10 re-evaluation:** targeted success was 229/768 against
  241/768 originally; sentinel success was 282/768 against 281/768. The
  [computational handoff](../artifacts/replay-v1/publication-checks-20260909/COMPUTATIONAL-HANDOFF.md)
  also reports half-life uncertainty and archived replay adapter geometry.
- **Dense retention:** both original selected replay checkpoints were continued
  again through 20 MAPS updates, with arithmetic evaluated after every update.
  The targeted 0–20 mean scores were 0.111035 for R-S and 0.116634 for R-P;
  the paired-item difference was +0.005599 [−0.005762, 0.016829]. First-crossing
  half-lives were 3.823529 and 1.551724 updates. The rebound spans updates 9–11
  in this run. The [readout](../artifacts/replay-v1/dense-retention-20260909/README.md)
  retains both roles, full trajectories, and the reused original update-0
  observations; it does not replace the original coarse-grid result.
- **Acquisition-order replication:** order seed 47 completed both 294-update
  acquisition runs on the same 579-example block-11 corpus. Prospective direct
  between-arm matching selected R-S@200 and R-P@140, and both 480-update
  continuations finished. Targeted 0–20 mean scores were 0.149935 and 0.025977,
  a difference of −0.123958 [−0.146745, −0.102507]. The first-crossing half-lives
  were 4.344828 and 0.878947 updates. The [readout](../artifacts/replay-v1/order-seed47-20260909/readout.md)
  contains the full retention, learning, and profile results. This is another
  acquisition-order realization, not a new trace corpus or source block.

Intervals above are paired item-bootstrap intervals conditional on the observed
training realization. The [acquisition-order specification](experiments/replay-order-replication.md)
remains separate from the original replay design.

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

## Manuscript preparation and final comparison

1. Preserve Pilot 0 and all completed replay/follow-up records unchanged.
2. Continue manuscript revision without publishing the paper. Any later posting
   or submission requires owner authorization; neither has occurred.
3. Execute only the authorized endpoint-clone comparison. No additional model,
   task, teacher, gauge intervention, or outcome-dependent rescue is included.
4. Finish provider invoice attribution during end-of-project accounting.

For the scientific overview, return to the [README](../README.md#results). For exact
schedules, gates, calibration history, budgets, and provenance, see the
[technical appendix](TECHNICAL.md), frozen [capability-targeted
amendment](amendment-capability-targeted-acquisition.md), and
[protocol](../PROTOCOL.md).
