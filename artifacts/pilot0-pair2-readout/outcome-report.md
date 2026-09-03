PREDICTION: CONFIRMED

# Pair-2 outcome report

Run: `pilot0-pair2-seed29-20260830T204211Z`; seed `29`; local artifacts
only. Rates below are proportions. In two-arm cells, values are `B-S / B-G`.

Matching selected B-S update `40` and B-G update `20`. Each checkpoint had
`17/96 = 0.177083333333` targeted exact success on the frozen cadence panel;
the absolute matching difference was `0`.

## Prospective mechanical score

| Registered leg | B-S | B-G | Holds |
|---|---:|---:|:---:|
| Targeted raw-Pass@1 retention half-life, updates | 1.136364 | 3.343284 | yes |
| Baseline-relative raw-Pass@1 gain AUC, updates 0–40 | 0.069696045 | 0.018778992 | yes |

The F1 crossings are between updates `1–2` for B-S and `2–5` for B-G. The
F2 AUC uses exactly `[0, 1, 2, 5, 10, 20, 40]`, trapezoidal integration, and
division by `40`.

## Pair-2 F1: old-skill retention

Stage-B update grid: `[0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480]`.
Each monitor point contains 192 items per role × 4 draws = 768 completions.

| Arm / role | Raw Pass@1 across the update grid |
|---|---|
| B-S / targeted | `[0.175781, 0.097656, 0.026042, 0.005208, 0.003906, 0, 0, 0, 0, 0, 0]` |
| B-G / targeted | `[0.197917, 0.200521, 0.138021, 0.050781, 0.024740, 0, 0, 0.003906, 0, 0, 0]` |
| B-S / sentinel | `[0.059896, 0.046875, 0.022135, 0.015625, 0.003906, 0, 0, 0, 0, 0, 0]` |
| B-G / sentinel | `[0.197917, 0.201823, 0.158854, 0.039062, 0.024740, 0.001302, 0, 0.006510, 0, 0, 0]` |

| Raw-Pass@1 statistic | B-S | B-G |
|---|---:|---:|
| Targeted relative retention, update 2 | 0.148148 | 0.697368 |
| Sentinel relative retention, update 2 | 0.369565 | 0.802632 |
| Targeted half-life, updates | 1.136364 | 3.343284 |
| Sentinel half-life, updates | 1.684211 | 3.500000 |

On the 256-item × 16-draw A-validation panel, targeted raw Pass@1 was
`0.177490234375 / 0.201904296875` before Stage B and `0 / 0` after update 480.
Sentinel raw Pass@1 was `0.052246093750 / 0.211425781250` before Stage B and
`0 / 0` after update 480. Final targeted and sentinel Pass@1, Pass@4, and
Pass@16 were all `0` for both arms.

Paired item-bootstrap raw-Pass@1 contrasts (`B-S − B-G`, 50,000 paired item
resamples; pointwise 95% percentile intervals) were:

| Update | Difference | 95% interval |
|---:|---:|---:|
| 1 | -0.10286458 | [-0.14062500, -0.06510417] |
| 2 | -0.11197917 | [-0.13802083, -0.08723958] |
| 5 | -0.04557292 | [-0.06119792, -0.02994792] |
| 10 | -0.02083333 | [-0.03255208, -0.00911458] |

## Pair-2 F2: new-skill learning

Each point contains 512 MAPS items × 16 draws = 8,192 completions.

| Arm | Raw Pass@1 across the update grid |
|---|---|
| B-S | `[0, 0.003174, 0.055298, 0.069458, 0.072998, 0.071655, 0.077393, 0.072632, 0.073730, 0.137329, 0.268555]` |
| B-G | `[0.053467, 0.076904, 0.049927, 0.068481, 0.070190, 0.074951, 0.076172, 0.138550, 0.171387, 0.330566, 0.368652]` |

| Statistic | B-S | B-G |
|---|---:|---:|
| Raw Pass@1 baseline | 0 | 0.053466797 |
| Raw Pass@1 gain, update 40 | 0.077392578 | 0.022705078 |
| Raw gain AUC, updates 0–40 | 0.069696045 | 0.018778992 |
| Stored posterior raw-gain AUC, updates 0–480 | 0.119604492 | 0.176492968 |
| Raw Pass@1 endpoint, update 480 | 0.268554688 | 0.368652344 |
| Posterior endpoint, update 480 | 0.282169118 | 0.376378676 |

The paired item-bootstrap update-480 raw-Pass@1 difference was
`-0.10009766`, interval `[-0.11828613, -0.08264160]`. The stored posterior
raw-gain AUC difference was `-0.05688848`, interval
`[-0.06862360, -0.04530086]`.

At updates `1, 2, 5, 10, 20, 40, 80`, the B-G cap/format-failure share of
all failures was respectively
`[0, 0.009379, 0.034989, 0, 0, 0, 0]`. At update 40 and update 80, its cap,
invalid-tag, and invalid-syntax counts were all `0`; raw successes were
`624/8192` and `1135/8192`, respectively. Full per-update failure codes and
length summaries are in the linked diagnostics package.

At update 0, B-G solved at least once in 16 draws on `249/512` MAPS items
(`438/8192` successful completions). Overlap of that item set with solved
sets at updates 80 and 480 was `192/215` for B-S and `211/227` for B-G,
respectively.

## Pair-2 F3: pre-Stage-B fingerprints

Each role contains 256 items × 16 draws = 4,096 completions.

| Targeted-panel metric | B-S @40 | B-G @20 |
|---|---:|---:|
| Raw Pass@1 | 0.177490234375 | 0.201904296875 |
| Pass@4 | 0.440021892170 | 0.566313959478 |
| Pass@16 | 0.796875000000 | 0.902343750000 |
| Syntactically invalid rate | 0.531250000000 | 0.639160156250 |
| Length-stop rate | 0.190185546875 | 0.286865234375 |
| Loops / length stops | 96/779 = 0.123234917 | 169/1175 = 0.143829787 |
| Mean / median response tokens | 885.973877 / 96 | 1793.645752 / 1097.5 |
| Unique completion texts | 3555 | 4021 |
| Verified strategy families | 82 | 435 |
| Mean sampled-token surprisal | 0.065836468125 | 0.252446657266 |

| Sentinel-panel metric | B-S @40 | B-G @20 |
|---|---:|---:|
| Raw Pass@1 | 0.052246093750 | 0.211425781250 |
| Pass@4 | 0.166447029533 | 0.572394402473 |
| Pass@16 | 0.398437500000 | 0.906250000000 |
| Syntactically invalid rate | 0.603027343750 | 0.623291015625 |
| Length-stop rate | 0.208740234375 | 0.293212890625 |
| Loops / length stops | 99/855 = 0.115789474 | 138/1201 = 0.114904246 |
| Mean / median response tokens | 965.663330 / 104 | 1795.084229 / 1041.5 |
| Unique completion texts | 3925 | 4045 |
| Verified strategy families | 84 | 437 |
| Mean sampled-token surprisal | 0.070446379921 | 0.259446434691 |

The separate 16-draw targeted pre-B raw-Pass@1 difference was `-0.02441406`,
with paired item-bootstrap interval `[-0.05322266, 0.00512695]`.

## Pair-2 selected-checkpoint geometry

Factor norms are pooled Frobenius norms across 249 modules. BA quantities use
the sorted union of per-module singular values: a block-diagonal bookkeeping
aggregate, not a composed network-level matrix.

| Metric | B-S @40 | B-G @20 |
|---|---:|---:|
| ‖A‖F | 51.943300220 | 51.545578405 |
| ‖B‖F | 7.274778829 | 1.332342982 |
| ‖BA‖F | 4.475285660 | 0.772266396 |
| Largest singular value | 2.463769427 | 0.297040441 |
| Stable rank | 3.299448850 | 6.759321664 |

Pair 2's ‖B‖F ratio is `5.46×`; Pair 1's is `7.36×`. This comparison is
descriptive only and is not scored or interpreted as a magnitude/dose-response
relationship from two pairs.

## Pair-1 / Pair-2 comparison

Every two-arm value is `B-S / B-G`.

| Metric | Pair 1 · seed 11 | Pair 2 · seed 29 |
|---|---:|---:|
| Selected checkpoints | 140 / 30 | 40 / 20 |
| Matched cadence score | 31/96 / 31/96 | 17/96 / 17/96 |
| ‖B‖F ratio, B-S ÷ B-G | 7.36× | 5.46× |
| Targeted raw-Pass@1 half-life | 2.664286 / 4.104651 | 1.136364 / 3.343284 |
| Sentinel raw-Pass@1 half-life | 6.315789 / 4.110714 | 1.684211 / 3.500000 |
| Targeted relative retention, update 2 | 0.567686 / 0.881857 | 0.148148 / 0.697368 |
| Sentinel relative retention, update 2 | 0.794118 / 0.933921 | 0.369565 / 0.802632 |
| F2 raw-Pass@1 baseline | 0 / 0.049926758 | 0 / 0.053466797 |
| F2 raw gain AUC, updates 0–40 | 0.076927185 / 0.017735291 | 0.069696045 / 0.018778992 |
| F2 raw gain, update 40 | 0.116577148 / 0.022338867 | 0.077392578 / 0.022705078 |
| Stored posterior raw-gain AUC, updates 0–480 | 0.246866264 / 0.189929798 | 0.119604492 / 0.176492968 |
| F2 raw-Pass@1 endpoint | 0.378173828 / 0.403686523 | 0.268554688 / 0.368652344 |

### Core F3 comparison

| Targeted-panel metric | P1 B-S@140 | P1 B-G@30 | P2 B-S@40 | P2 B-G@20 |
|---|---:|---:|---:|---:|
| Raw Pass@1 | 0.336181641 | 0.309814453 | 0.177490234 | 0.201904297 |
| Sentinel raw Pass@1 | 0.041748047 | 0.316406250 | 0.052246094 | 0.211425781 |
| Median response tokens | 78 | 1464 | 96 | 1097.5 |
| Syntactically invalid rate | 0.308105469 | 0.573974609 | 0.531250000 | 0.639160156 |
| Length-stop rate | 0.019775391 | 0.303222656 | 0.190185547 | 0.286865234 |
| Loop fraction among stops | 0.283950617 | 0.101449275 | 0.123234917 | 0.143829787 |
| Unique completion texts | 2430 | 4057 | 3555 | 4021 |
| Verified strategy families | 40 | 542 | 82 | 435 |
| Mean sampled-token surprisal | 0.047192886 | 0.245732785 | 0.065836468 | 0.252446657 |
| New-task raw Pass@1 before Stage B | 0 | 0.049926758 | 0 | 0.053466797 |

## Complete local package

- [Full F1/F2 trajectories and exact counts](README.md)
- [Full F3 profiles](F3.md)
- [Paired uncertainty and early-window F1](../pilot0-pair2-offline-analysis/uncertainty.md)
- [F2 failure, length, and solved-item diagnostics](../pilot0-pair2-offline-analysis/f2_diagnostics.md)
- [All adapter-geometry summaries and cadence trajectories](../pilot0-pair2-offline-analysis/geometry.md)
- [Executed descriptive notebook and combined report](../pilot0-pair2-offline-analysis/README.md)
- [Prospective rule recorded before outcome inspection](../../docs/confirmatory-design-notes.md)

No remote calls, new sampling, training, or metric definitions were used.
