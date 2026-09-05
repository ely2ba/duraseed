# Paired item-level uncertainty and early-window F1

Descriptive, post-hoc, seed 11 only. No result enters a gate or changes the running experiment.

## Paired uncertainty

All contrasts are B-S minus B-G. Intervals are 95% paired item-cluster percentile bootstrap CIs with 50,000 replicates (NumPy PCG64, base seed 314159; seeds 314160 and 314161 for the monitor and pre-B panels). Sampling the same item index in both arms preserves pairing; for AUC, that index retains the entire trajectory and all within-item draws. No individual completions are treated as independent items.

The sampling unit is an item, conditional on these fitted models, their selected checkpoints, the fixed family panel, and the observed per-item draw records. Items are treated as exchangeable independent clusters within each analyzed population. This estimates variability over empirical item clusters with realized draws retained, not fresh-completion uncertainty for the exact fixed panel. This is not a family-cluster or training-seed CI; correlation across items in a family could make intervals too narrow. The monitor has 12 assigned families per role. Intervals do not include training-seed variance, checkpoint-selection uncertainty, uncertainty from choosing these post-hoc windows, or a separate within-item resampling model. They are pointwise, without multiplicity adjustment. A posterior-score bootstrap CI is not a Bayesian credible interval.

| Contrast | Paired items | Estimate | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|---:|
| F2 update-480 Pass@1 | 512 | -0.02551270 | -0.04345703 | -0.00769043 |
| F2 update-480 posterior mean | 512 | -0.02401195 | -0.04090074 | -0.00723805 |
| F2 raw-gain AUC (stored posterior-score estimand) | 512 | 0.05693647 | 0.04660165 | 0.06723609 |
| F2 Pass@1 gain AUC (supplementary raw-rate estimand) | 512 | 0.06049500 | 0.04951425 | 0.07143835 |
| F1 targeted monitor update-1 Pass@1 | 192 | -0.03906250 | -0.08593750 | 0.00781250 |
| F1 targeted monitor update-1 posterior mean | 192 | -0.03125000 | -0.06875000 | 0.00625000 |
| F1 targeted monitor update-2 Pass@1 | 192 | -0.10286458 | -0.14973958 | -0.05729167 |
| F1 targeted monitor update-2 posterior mean | 192 | -0.08229167 | -0.11979167 | -0.04583333 |
| F1 targeted monitor update-5 Pass@1 | 192 | -0.02604167 | -0.05859375 | 0.00651042 |
| F1 targeted monitor update-5 posterior mean | 192 | -0.02083333 | -0.04687500 | 0.00520833 |
| F1 targeted monitor update-10 Pass@1 | 192 | -0.00260417 | -0.01041667 | 0.00520833 |
| F1 targeted monitor update-10 posterior mean | 192 | -0.00208333 | -0.00833333 | 0.00416667 |
| Pre-B 16-draw targeted Pass@1 (matching imperfection) | 256 | 0.02636719 | -0.00390625 | 0.05712891 |
| Pre-B 16-draw targeted posterior mean (matching imperfection) | 256 | 0.02481618 | -0.00367647 | 0.05376838 |

Units are proportions (multiply by 100 for percentage points). F2 raw-gain AUC here is the existing posterior-score estimand: trapezoidal absolute AUC over updates 0–480, normalized by 480, minus that arm's own update-0 posterior score. The supplementary Pass@1-gain AUC applies the same operation to raw rates. With 16 draws, posterior differences and gain differences equal 16/17 times their raw-rate counterparts.

Pre-B uses the separate 256-targeted-item × 16-draw A-validation panel, not the one-draw cadence matching panel (31/96 for each selected checkpoint). Its CI is conditional on B-S@140 and B-G@30 having been selected; it does not certify an equivalence margin or adjust for selection.

## Early-window retention

Relative retention is the ratio of the panel's mean score at t to its own update-0 mean; it is not a mean of itemwise ratios and is not clipped at 1. Both raw Pass@1 and the stored Jeffreys posterior score are shown. Half-life is the first downward crossing of half the initial score, linearly interpolated between the adjacent observed checkpoints on the update axis; it is not an exponential-fit parameter or an observed intermediate checkpoint. All 11 available points (0–480) are searched.

AUC(0–20) uses only the observed grid [0,1,2,5,10,20], trapezoidal integration, divided by 20. Relative AUC additionally divides by that arm/role's initial score; the unnormalized absolute area is in the JSON. These are point estimates, without bootstrap intervals.

| Arm / role / score | Relative u1 | u2 | u5 | u10 | Half-life (updates) | Absolute AUC 0–20 | Relative AUC 0–20 |
|---|---:|---:|---:|---:|---|---:|---:|
| B-S / targeted / Pass@1 | 0.825328 | 0.567686 | 0.262009 | 0.017467 | 2.664286 [2, 5] | 0.05458984 | 0.18307860 |
| B-S / targeted / posterior mean | 0.876923 | 0.695385 | 0.480000 | 0.307692 | 4.721429 [2, 5] | 0.14367188 | 0.42438462 |
| B-G / targeted / Pass@1 | 0.924051 | 0.881857 | 0.337553 | 0.025316 | 4.104651 [2, 5] | 0.07425130 | 0.24061181 |
| B-G / targeted / posterior mean | 0.945946 | 0.915916 | 0.528529 | 0.306306 | 5.641892 [5, 10] | 0.15940104 | 0.45953453 |
| B-S / sentinel / Pass@1 | 0.794118 | 0.794118 | 0.647059 | 0.088235 | 6.315789 [5, 10] | 0.01357422 | 0.30661765 |
| B-S / sentinel / posterior mean | 0.946154 | 0.946154 | 0.907692 | 0.761538 | Not crossed (0–480) | 0.11085938 | 0.81865385 |
| B-G / sentinel / Pass@1 | 1.057269 | 0.933921 | 0.317181 | 0.044053 | 4.110714 [2, 5] | 0.07587891 | 0.25671806 |
| B-G / sentinel / posterior mean | 1.040248 | 0.953560 | 0.520124 | 0.328173 | 5.524194 [5, 10] | 0.16070313 | 0.47763158 |

For B-S sentinel posterior retention, half the initial score lies below the four-draw Jeffreys floor of 0.1, so the requested posterior half-life is unattainable under this score definition; the raw-rate half-life is reported separately. Relative retention does not subtract M0 ability and does not measure survival restricted to newly acquired items.

Raw source references ([data download and layout](../../docs/pilot0-data.md)): pair-1 result (`runs/pilot0/pilot0-pair1-seed11-20260825T125100Z/result.json`); matching (`runs/pilot0/pilot0-pair1-seed11-20260825T125100Z/seed-11/matching.json`); [all exact estimates and evaluation paths](uncertainty.json).

Checks: identical paired item-ID and draw populations at every included checkpoint; existing stored F2 curves/AUC contrast reproduced; raw-to-posterior 16/17 identity checked. No remote calls or new samples.
