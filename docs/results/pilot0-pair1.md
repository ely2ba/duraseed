# Pilot 0 — first paired-seed results

Seed `11`; run `pilot0-pair1-seed11-20260825T125100Z`.
B-S denotes supervised acquisition; B-G denotes reinforcement-learning acquisition.
Both selected checkpoints subsequently received the same 480-update Stage-B recipe.
This page reports that one completed pair. It does not combine training seeds.

The frozen summaries are F1 retention, F2 subsequent learning, and F3 pre-Stage-B
profiles. The item-bootstrap intervals, early-window summaries, failure-code
cross-tabs, and adapter-geometry analysis below are descriptive, post-hoc analyses.
None changes a checkpoint selection, gate, or training recipe.

## Checkpoint selection and starting scores

The frozen matching rule selected **B-S update 140** and **B-G update 30**.
Both scored **31/96** on the one-draw targeted cadence panel.

On the separate targeted A-validation panel, with 256 items and 16 draws per item:

- B-S: **1,377/4,096**, or **33.618164%** exact success.
- B-G: **1,269/4,096**, or **30.981445%** exact success.
- B-S minus B-G: **2.636719 percentage points**; paired item-bootstrap
  95% CI **[−0.390625, 5.712891] points**.

The matching-panel equality is not an equivalence test on the higher-draw panel.
The interval above is conditional on these checkpoints having been selected.

## Reading the measurements

All trajectory arrays use the same Stage-B update grid:

```text
updates = [0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480]
```

- **A monitor:** 192 targeted items and 192 sentinel items, four draws each.
  Each panel therefore has 768 completions at each update.
- **A validation:** 256 targeted and 256 sentinel items, 16 draws each;
  4,096 completions per panel at the selected checkpoint and final endpoint.
- **B validation:** 512 MAPS items, 16 draws each; 8,192 completions per update.
- **Raw Pass@1:** exact successes divided by completions. Counts below retain
  all verification failures in the denominator.
- **Posterior score:** mean across items of `(successes + 0.5)/(draws + 1)`.
  Its zero-success floor is `0.1` with four draws and `1/34` with 16 draws;
  neither floor is an observed success rate.
- **Pass@k:** mean across items of the unbiased estimator
  `1 − choose(n−c,k)/choose(n,k)`, using the already-collected draws.

See the tracked [analysis definitions](../../src/duraseed/evaluation/analysis.py)
and [Pilot reducers](../../src/duraseed/pilot0_analysis.py).

## F1 — retention during Stage B

These are exact-success counts on A monitor; divide each entry by 768 for Pass@1.

```text
B-S targeted = [229, 189, 130, 60, 4, 1, 1, 1, 2, 0, 0]
B-G targeted = [237, 219, 209, 80, 6, 4, 6, 3, 0, 0, 0]
B-S sentinel = [34, 27, 27, 22, 3, 0, 0, 0, 0, 0, 0]
B-G sentinel = [227, 240, 212, 72, 10, 5, 2, 2, 0, 0, 0]
denominator   = 768 at every point in every array
```

Targeted monitor Pass@1 starts at **29.817708% / 30.859375%** for B-S / B-G.
At update 2 it is **16.927083% / 27.213542%**; at update 10,
**0.520833% / 0.781250%**.

On the separate A-validation panel, selected checkpoint → Stage-B update 480:

- B-S targeted: **1,377/4,096 → 0/4,096**; sentinel: **171/4,096 → 0/4,096**.
- B-G targeted: **1,269/4,096 → 0/4,096**; sentinel: **1,296/4,096 → 0/4,096**.
- Final Pass@1, Pass@4 and Pass@16 are all zero on both panels for both arms.
  Final posterior scores are `0.029411764705882353` for all four panels.

These endpoints are zero observed successes, not a measurement that latent
success probabilities or all underlying abilities are exactly zero.

### Exploratory early-window summaries

Relative retention divides each panel's mean score by its own **monitor update-0**
mean. It is not a mean of itemwise ratios, is not clipped at one, and does not
subtract M0 ability. It does not isolate items whose ability was newly acquired.

For raw Pass@1, ratios at updates `[1, 2, 5, 10]` are:

```text
B-S targeted = [0.825328, 0.567686, 0.262009, 0.017467]
B-G targeted = [0.924051, 0.881857, 0.337553, 0.025316]
B-S sentinel = [0.794118, 0.794118, 0.647059, 0.088235]
B-G sentinel = [1.057269, 0.933921, 0.317181, 0.044053]
```

Raw-rate half-life is the first downward crossing of half the update-0 score,
linearly interpolated between observed checkpoints:

- Targeted: B-S **2.664286 updates**, bracket `[2,5]`; B-G **4.104651**, `[2,5]`.
- Sentinel: B-S **6.315789 updates**, bracket `[5,10]`; B-G **4.110714**, `[2,5]`.

These are interpolations, not measured intermediate checkpoints or exponential
decay fits. The search uses all 11 available checkpoints through update 480.

AUC over updates 0–20 integrates `[0,1,2,5,10,20]` with the trapezoidal rule
and divides by 20. The following pairs are **absolute AUC / own-baseline-relative AUC**:

- Raw targeted: B-S `0.05458984 / 0.18307860`; B-G `0.07425130 / 0.24061181`.
- Raw sentinel: B-S `0.01357422 / 0.30661765`; B-G `0.07587891 / 0.25671806`.
- Posterior targeted: B-S `0.14367188 / 0.42438462`; B-G `0.15940104 / 0.45953453`.
- Posterior sentinel: B-S `0.11085938 / 0.81865385`; B-G `0.16070313 / 0.47763158`.

Posterior-score half-lives are **4.721429 `[2,5]` / 5.641892 `[5,10]`** for
targeted B-S / B-G, and **not crossed / 5.524194 `[5,10]`** for sentinel B-S / B-G.
Half of B-S's initial sentinel posterior score lies below the four-draw floor
of `0.1`; that posterior half-life cannot cross its threshold under this definition.
These early-window statistics have no bootstrap intervals and were chosen post hoc.

## F2 — subsequent MAPS learning

Exact-success counts on B validation, on the same 11-point update grid:

```text
B-S successes = [0, 24, 292, 513, 583, 575, 955, 1336, 1935, 2668, 3098]
B-G successes = [409, 534, 443, 503, 584, 549, 592, 1407, 1469, 2691, 3307]
denominator    = 8192 at every point
```

Raw Pass@1 baseline → endpoint is **0.000000% → 37.817383%** for B-S and
**4.992676% → 40.368652%** for B-G. Endpoint minus own baseline is
**37.817383 / 35.375977 percentage points**, respectively.

The frozen AUC uses posterior scores, trapezoidal integration on the **linear**
update axis, and division by 480. The recorded `raw_gain_auc` means absolute
posterior-score AUC minus the arm's own update-0 posterior score; “raw” here
does not mean raw Pass@1.

- Absolute posterior AUC: B-S **0.2762780283**; B-G **0.2663314520**.
- Posterior gain AUC: B-S **0.2468662636**; B-G **0.1899297976**.
- Headroom-normalized gain AUC: B-S **0.2543470595**; B-G **0.2056411193**.
  This divides gain AUC by `1 − baseline + 1e-12`.
- Posterior endpoint: B-S **0.3853400735**; B-G **0.4093520221**.

Endpoint, absolute AUC and own-baseline-relative AUC are distinct summaries.
The paired endpoint and gain-AUC contrasts below have opposite signs.

### Failure-code and baseline-item checks

At B-G updates 40 → 80, successes change **592 → 1,407** and wrong-target
failures **7,600 → 6,785**, all out of 8,192 completions. Cap, invalid-tag and
invalid-syntax counts are **0 → 0**. Across updates 1–40, the union of cap/format
failures accounts for **0%–0.2323%** of all failures. These counts do not identify
a causal learning mechanism. The sampling cap is 128 tokens; instruction-count
failures are separate from token-cap stops.

Define a baseline-solved item as an item with at least one exact success in its
16 B-G update-0 draws. There are **223/512** such items, with **409/8,192**
successful completions. At subsequent checkpoints:

- B-S update 80: 336 solved items; 186 overlap the baseline set;
  768/1,336 successful completions are on baseline-set items.
- B-S update 480: 406 solved items; overlap 197;
  1,689/3,098 successful completions are on baseline-set items.
- B-G update 80: 306 solved items; overlap 179;
  771/1,407 successful completions are on baseline-set items.
- B-G update 480: 400 solved items; overlap 206;
  1,898/3,307 successful completions are on baseline-set items.

Set membership reflects these finite observed draws, not a latent ability label.
Conditioning on observed baseline successes can introduce selection and
regression-to-the-mean effects. See the [MAPS verifier](../../src/duraseed/tasks/maps/verifier.py).

## Paired uncertainty

All contrasts below are **B-S minus B-G**, in **proportion units**; multiply by
100 for percentage points. Each line gives **estimate; [95% CI lower, upper]**.

- F2 update-480 Pass@1: `−0.02551270; [−0.04345703, −0.00769043]`.
- F2 update-480 posterior score: `−0.02401195; [−0.04090074, −0.00723805]`.
- F2 frozen posterior gain AUC: `0.05693647; [0.04660165, 0.06723609]`.
- F2 supplementary raw-Pass@1 gain AUC: `0.06049500; [0.04951425, 0.07143835]`.
- F1 targeted monitor u1 Pass@1: `−0.03906250; [−0.08593750, 0.00781250]`.
- F1 targeted monitor u1 posterior: `−0.03125000; [−0.06875000, 0.00625000]`.
- F1 targeted monitor u2 Pass@1: `−0.10286458; [−0.14973958, −0.05729167]`.
- F1 targeted monitor u2 posterior: `−0.08229167; [−0.11979167, −0.04583333]`.
- F1 targeted monitor u5 Pass@1: `−0.02604167; [−0.05859375, 0.00651042]`.
- F1 targeted monitor u5 posterior: `−0.02083333; [−0.04687500, 0.00520833]`.
- F1 targeted monitor u10 Pass@1: `−0.00260417; [−0.01041667, 0.00520833]`.
- F1 targeted monitor u10 posterior: `−0.00208333; [−0.00833333, 0.00416667]`.
- Pre-B targeted validation Pass@1: `0.02636719; [−0.00390625, 0.05712891]`.
- Pre-B targeted validation posterior: `0.02481618; [−0.00367647, 0.05376838]`.

These are paired item-cluster percentile bootstrap intervals with 50,000 replicates:
512 paired items for F2, 192 for targeted F1 monitor, and 256 for pre-B validation.
One resampled item retains both arms, all time points and its complete observed
draw counts. NumPy PCG64 seeds are 314159, 314160 and 314161, respectively.

Items are treated as exchangeable independent clusters within each analyzed
population. These are empirical item-cluster intervals conditional on one fitted
seed pair, selected checkpoints and realized draws—not uncertainty across training
seeds or a separate model for fresh completions on the exact fixed panel.
They are not family-cluster intervals; within-family dependence may make them
too narrow. They exclude checkpoint-selection uncertainty and are pointwise,
without multiplicity adjustment. Posterior-score bootstrap intervals are not
Bayesian credible intervals. With 16 draws, posterior differences and gain
differences are `16/17` of the corresponding raw-rate differences.

## F3 — profiles before Stage B

Each role below has 256 items × 16 draws = 4,096 completions. Each has 12 assigned
families. Pass@k lists are ordered `[1,4,16]`; all use existing draws.

### B-S update 140

- **Targeted:** success `1377/4096` (`0.336181640625`); Pass@k
  `[0.33618164, 0.71421274, 0.92578125]`; syntactic-invalid `1262/4096` (`0.30810547`).
  Token length minimum/mean/median/maximum: `44 / 175.860840 / 78 / 4096`.
  Cap stops `81/4096` (`0.01977539`); detected loops among those stops `23/81`
  (`0.28395062`). Unique texts `2430`; verified strategy families `40`.
  Mean sampled-token surprisal `0.04719289`; logprob coverage `720326/720326`.
- **Sentinel:** success `171/4096` (`0.041748046875`); Pass@k
  `[0.04174805, 0.11793441, 0.24218750]`; syntactic-invalid `1965/4096` (`0.47973633`).
  Token length: `52 / 222.393311 / 87 / 4096`.
  Cap stops `110/4096` (`0.02685547`); loops `28/110` (`0.25454545`).
  Unique texts `3109`; verified strategy families `43`.
  Mean sampled-token surprisal `0.04810765`; coverage `910923/910923`.

### B-G update 30

- **Targeted:** success `1269/4096` (`0.309814453125`); Pass@k
  `[0.30981445, 0.72900498, 0.98437500]`; syntactic-invalid `2351/4096` (`0.57397461`).
  Token length: `3 / 1985.206055 / 1464 / 4096`.
  Cap stops `1242/4096` (`0.30322266`); loops `126/1242` (`0.10144928`).
  Unique texts `4057`; verified strategy families `542`.
  Mean sampled-token surprisal `0.24573279`; coverage `8131404/8131404`.
- **Sentinel:** success `1296/4096` (`0.31640625`); Pass@k
  `[0.31640625, 0.73000086, 0.95312500]`; syntactic-invalid `2328/4096` (`0.56835938`).
  Token length: `3 / 1947.673828 / 1408.5 / 4096`.
  Cap stops `1188/4096` (`0.29003906`); loops `114/1188` (`0.09595960`).
  Unique texts `4070`; verified strategy families `563`.
  Mean sampled-token surprisal `0.24858168`; coverage `7977672/7977672`.

Syntactic invalidity is not answer-tag invalidity. Cap stops mean sampled tokens
reach the sampling cap. Loops use the final-1,024-token lag-agreement detector,
only among cap-stopped completions. Unique texts are exact text identities;
verified strategy families count only reward-1 completions, not assigned families.
Surprisal is the negative mean recorded sampled-token logprob, not full-vocabulary
entropy. Definitions: [profile reducer](../../src/duraseed/pilot0_profiles.py),
[loop detector](../../src/duraseed/runners/capability_dose_evidence.py), and
[TCES verifier](../../src/duraseed/tasks/tces/verifier.py).

## Adapter geometry — supplementary and descriptive

The local archive contains 34 Stage-A checkpoints: B-S updates 10–290 every 10,
and B-G updates 10–50 every 10. It includes the selected checkpoints but no M0
or B-S update-294 adapter. Each checkpoint has 249 adapted modules: 248 within
transformer layers 0–31 and the output projection. LoRA rank and alpha are both 32.

At selected B-S u140 / B-G u30:

- Aggregate A-factor Frobenius norm: **52.232121 / 51.551249**.
- Aggregate B-factor Frobenius norm: **10.226575 / 1.389553**, ratio **7.359615**.
- Block-diagonal BA Frobenius norm: **6.436389 / 0.805484**.
- Largest module singular value: **3.446728 / 0.312937**, ratio **11.014117**;
  both are in the output projection.
- Block-diagonal entropy effective rank: **5178.854881 / 5916.142512**;
  stable rank: **3.487149 / 6.625200**.
- Output-projection stable rank: **1.223585 / 1.416887**; its leading direction
  contains **81.727057% / 70.577243%** of squared spectral mass. Numerical rank is 32.

Factor norms depend on LoRA parameterization: compensating changes to A and B
can leave BA unchanged. Aggregate spectra are the block-diagonal union of module
matrices, not the composition, Jacobian or effective rank of the whole network.
These are full saved adapters, not differences from M0, and do not establish a
causal explanation for F1 or F2. Entropy rank uses normalized singular values;
stable rank is squared Frobenius norm divided by the largest singular value squared.

## Local source locations

Exact unrounded values, item/family records and full spectra are retained locally
at the following repository-relative locations; these generated artifacts are
not required to be present in a public checkout:

```text
runs/pilot0/pilot0-pair1-seed11-20260825T125100Z/result.json
runs/pilot0/pilot0-pair1-seed11-20260825T125100Z/seed-11/matching.json
runs/pilot0/pilot0-pair1-seed11-20260825T125100Z/seed-11/pre-b-profiles.json
artifacts/pilot0-pair1-readout/readout.json
artifacts/pilot0-pair1-offline-analysis/uncertainty.json
artifacts/pilot0-pair1-offline-analysis/f2_diagnostics.json
artifacts/pilot0-pair1-offline-analysis/geometry.json
```
