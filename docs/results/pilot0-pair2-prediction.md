# Pilot 0 Pair-2 prospective prediction and outcome

This record preserves the read order for Pair 2, seed `29`. It contains no
metric or analysis defined after the outcomes were opened.

## Prospective prediction

_Initial mechanical prediction recorded 2026-08-30 at 21:11:34 UTC, before
Pair-2 outcome inspection._

Pair-1 adapter geometry suggested that inherited LoRA factor scale might act as
an effective step-size multiplier under the fresh Stage-B optimizer. The
prospective prediction for the next pair was that the selected arm with the
larger aggregate B-factor Frobenius norm would show both faster F1 decay and
faster early F2 gain.

After Pair 2 completed, its integrity and matching status were reported first.
The frozen rule selected B-S update `40` and B-G update `20`, each with `17/96`
targeted successes on the matching panel. Archived geometry was then read
without consulting F1/F2/F3 outcomes:

| Selected checkpoint | ‖A‖F | ‖B‖F | ‖BA‖F | sigma1 | Stable rank |
|---|---:|---:|---:|---:|---:|
| B-S @40 | 51.943300 | 7.274779 | 4.475286 | 2.463769 | 3.299449 |
| B-G @20 | 51.545578 | 1.332343 | 0.772266 | 0.297040 | 6.759322 |

The B-factor norm ratio was `5.46×`, mechanically selecting B-S. The recorded
prediction was therefore: **B-S shows faster F1 decay and faster early F2
gain.**

## Scoring rule

_Recorded 2026-09-03 at 09:22:40 UTC, after the geometry-first prediction and
before opening any Pair-2 F1/F2/F3 outcome._

1. **Faster forgetting under B-S.** B-S has the shorter targeted raw-Pass@1
   retention half-life: the first downward crossing of 50% of that arm's own
   update-0 Pass@1, linearly interpolated between registered checkpoints. If
   one arm never crosses, it is treated as longer than an arm that does cross.
   If neither crosses, this leg is inconclusive.
2. **Faster early Stage-B learning under B-S.** B-S has the larger
   baseline-relative raw-Pass@1 gain AUC over the fixed grid
   `[0, 1, 2, 5, 10, 20, 40]`, trapezoidally integrated and divided by `40`,
   where `gain(t) = Pass@1(t) − Pass@1(0)`.

Scoring was fixed as:

- both predictions hold: `CONFIRMED`;
- exactly one holds: `PARTIAL MISS`;
- neither holds: `FAILED`; and
- an inconclusive F1 leg does not count as holding.

Targeted relative retention at update 2, raw Pass@1 gain at update 40, the
stored 0–480 raw-gain AUC, the F2 endpoint, F3 profiles, geometry, and other
diagnostics were explicitly excluded from scoring.

## Outcome

_Outcome released and recorded 2026-09-03 at 10:07:08 UTC, after the human
confirmed `PREDICTION RECORDED`._

**PREDICTION: CONFIRMED**

| Registered leg | B-S | B-G | Result |
|---|---:|---:|:---:|
| Targeted raw-Pass@1 retention half-life, updates | 1.136364 | 3.343284 | held |
| Baseline-relative raw-Pass@1 gain AUC, updates 0–40 | 0.069696045 | 0.018778992 | held |

Pair 2's `5.46×` B-factor ratio is smaller than Pair 1's `7.36×`. This is a
descriptive comparison only; no magnitude or dose-response relationship is
scored or inferred from two pairs.

The [complete Pair-2 outcome report](../../artifacts/pilot0-pair2-readout/outcome-report.md)
contains the F1/F2 trajectories, F3 profiles, paired uncertainty, diagnostics,
and the Pair-1/Pair-2 comparison.
