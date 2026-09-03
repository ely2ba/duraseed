# Paired item-level uncertainty and early-window F1

Descriptive, post-hoc, seed 29 only. No result enters a gate or changes the running experiment.

## Paired uncertainty

All contrasts are B-S minus B-G. Intervals are 95% paired item-cluster percentile bootstrap CIs with 50,000 replicates (NumPy PCG64, base seed 314159; seeds 314160 and 314161 for the monitor and pre-B panels). Sampling the same item index in both arms preserves pairing; for AUC, that index retains the entire trajectory and all within-item draws. No individual completions are treated as independent items.

The sampling unit is an item, conditional on these fitted models, their selected checkpoints, the fixed family panel, and the observed per-item draw records. Items are treated as exchangeable independent clusters within each analyzed population. This estimates variability over empirical item clusters with realized draws retained, not fresh-completion uncertainty for the exact fixed panel. This is not a family-cluster or training-seed CI; correlation across items in a family could make intervals too narrow. The monitor has 12 assigned families per role. Intervals do not include training-seed variance, checkpoint-selection uncertainty, uncertainty from choosing these post-hoc windows, or a separate within-item resampling model. They are pointwise, without multiplicity adjustment. A posterior-score bootstrap CI is not a Bayesian credible interval.

| Contrast | Paired items | Estimate | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|---:|
| F2 update-480 Pass@1 | 512 | -0.10009766 | -0.11828613 | -0.08264160 |
| F2 update-480 posterior mean | 512 | -0.09420956 | -0.11132813 | -0.07778033 |
| F2 raw-gain AUC (stored posterior-score estimand) | 512 | -0.05688848 | -0.06862360 | -0.04530086 |
| F2 Pass@1 gain AUC (supplementary raw-rate estimand) | 512 | -0.06044401 | -0.07291257 | -0.04813217 |
| F1 targeted monitor update-1 Pass@1 | 192 | -0.10286458 | -0.14062500 | -0.06510417 |
| F1 targeted monitor update-1 posterior mean | 192 | -0.08229167 | -0.11250000 | -0.05208333 |
| F1 targeted monitor update-2 Pass@1 | 192 | -0.11197917 | -0.13802083 | -0.08723958 |
| F1 targeted monitor update-2 posterior mean | 192 | -0.08958333 | -0.11041667 | -0.06979167 |
| F1 targeted monitor update-5 Pass@1 | 192 | -0.04557292 | -0.06119792 | -0.02994792 |
| F1 targeted monitor update-5 posterior mean | 192 | -0.03645833 | -0.04895833 | -0.02395833 |
| F1 targeted monitor update-10 Pass@1 | 192 | -0.02083333 | -0.03255208 | -0.00911458 |
| F1 targeted monitor update-10 posterior mean | 192 | -0.01666667 | -0.02604167 | -0.00729167 |
| Pre-B 16-draw targeted Pass@1 (matching imperfection) | 256 | -0.02441406 | -0.05322266 | 0.00512695 |
| Pre-B 16-draw targeted posterior mean (matching imperfection) | 256 | -0.02297794 | -0.05009191 | 0.00482537 |

Units are proportions (multiply by 100 for percentage points). F2 raw-gain AUC here is the existing posterior-score estimand: trapezoidal absolute AUC over updates 0–480, normalized by 480, minus that arm's own update-0 posterior score. The supplementary Pass@1-gain AUC applies the same operation to raw rates. With 16 draws, posterior differences and gain differences equal 16/17 times their raw-rate counterparts.

Pre-B uses the separate 256-targeted-item × 16-draw A-validation panel, not the one-draw cadence matching panel (17/96 for each selected checkpoint). Its CI is conditional on B-S@40 and B-G@20 having been selected; it does not certify an equivalence margin or adjust for selection.

## Early-window retention

Relative retention is the ratio of the panel's mean score at t to its own update-0 mean; it is not a mean of itemwise ratios and is not clipped at 1. Both raw Pass@1 and the stored Jeffreys posterior score are shown. Half-life is the first downward crossing of half the initial score, linearly interpolated between the adjacent observed checkpoints on the update axis; it is not an exponential-fit parameter or an observed intermediate checkpoint. All 11 available points (0–480) are searched.

AUC(0–20) uses only the observed grid [0,1,2,5,10,20], trapezoidal integration, divided by 20. Relative AUC additionally divides by that arm/role's initial score; the unnormalized absolute area is in the JSON. These are point estimates, without bootstrap intervals.

| Arm / role / score | Relative u1 | u2 | u5 | u10 | Half-life (updates) | Absolute AUC 0–20 | Relative AUC 0–20 |
|---|---:|---:|---:|---:|---|---:|---:|
| B-S / targeted / Pass@1 | 0.555556 | 0.148148 | 0.029630 | 0.022222 | 1.136364 [1, 2] | 0.01438802 | 0.08185185 |
| B-S / targeted / posterior mean | 0.740260 | 0.502165 | 0.432900 | 0.428571 | 2.093750 [2, 5] | 0.11151042 | 0.46341991 |
| B-G / targeted / Pass@1 | 1.013158 | 0.697368 | 0.256579 | 0.125000 | 3.343284 [2, 5] | 0.04820964 | 0.24358553 |
| B-G / targeted / posterior mean | 1.008065 | 0.814516 | 0.544355 | 0.463710 | 7.750000 [5, 10] | 0.13856771 | 0.53639113 |
| B-S / sentinel / Pass@1 | 0.782609 | 0.369565 | 0.260870 | 0.065217 | 1.684211 [1, 2] | 0.01064453 | 0.17771739 |
| B-S / sentinel / posterior mean | 0.929577 | 0.795775 | 0.760563 | 0.697183 | Not crossed (0–480) | 0.10851563 | 0.73362676 |
| B-G / sentinel / Pass@1 | 1.019737 | 0.802632 | 0.197368 | 0.125000 | 3.500000 [2, 5] | 0.04833984 | 0.24424342 |
| B-G / sentinel / posterior mean | 1.012097 | 0.879032 | 0.508065 | 0.463710 | 5.909091 [5, 10] | 0.13867188 | 0.53679435 |

For B-S sentinel posterior retention, half the initial score lies below the four-draw Jeffreys floor of 0.1, so the requested posterior half-life is unattainable under this score definition; the raw-rate half-life is reported separately. Relative retention does not subtract M0 ability and does not measure survival restricted to newly acquired items.

Sources: [pair-2 result](/Users/elyb/Documents/DuraSeed-v1/runs/pilot0/pilot0-pair2-seed29-20260830T204211Z/result.json); [matching](/Users/elyb/Documents/DuraSeed-v1/runs/pilot0/pilot0-pair2-seed29-20260830T204211Z/seed-29/matching.json); [all exact estimates and evaluation paths](uncertainty.json).

Checks: identical paired item-ID and draw populations at every included checkpoint; existing stored F2 curves/AUC contrast reproduced; raw-to-posterior 16/17 identity checked. No remote calls or new samples.
