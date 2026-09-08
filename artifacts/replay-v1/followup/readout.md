# Replay-v1 descriptive readout

Run: `replay-v1-continuation-20260907T075721Z`.

Contrasts are R-P minus R-S. Intervals are 50,000-resample paired-item trajectory 95% percentile intervals, conditional on each fixed source block. Raw rates and Jeffreys posterior means are reported separately.

## Source block 11: COMPLETED

| Quantity | R-S | R-P |
|---|---:|---:|
| selected_update | 220 | 20 |
| targeted_raw_retention_auc_0_20 | 0.11494141 | 0.17620443 |
| maps_raw_absolute_auc_0_480 | 0.32414767 | 0.29767329 |
| maps_raw_endpoint | 0.45788574 | 0.45703125 |
| maps_raw_baseline | 0 | 0.051879883 |
| maps_raw_absolute_auc_0_40 | 0.11023102 | 0.070675659 |
| maps_raw_gain_auc_0_40 | 0.11023102 | 0.018795776 |

| Contrast | Difference | 95% interval |
|---|---:|---|
| targeted/raw-absolute-AUC-0-20 | 0.061263021 | [0.042903646, 0.079394531] |
| stage-b/raw-absolute-AUC-0-40 | -0.039555359 | [-0.047825699, -0.031600952] |
| stage-b/raw-gain-AUC-0-40 | -0.091435242 | [-0.10292068, -0.080183296] |
| stage-b/raw-absolute-AUC-0-480 | -0.02647438 | [-0.03422815, -0.018756336] |
| stage-b/raw-baseline | 0.051879883 | [0.045166016, 0.059082031] |
| stage-b/raw-endpoint480 | -0.00085449219 | [-0.019165039, 0.017578125] |

Early gain-AUC difference -0.091435242 = absolute-AUC difference -0.039555359 − baseline difference 0.051879883.

### targeted trajectories

| Update | R-S raw | R-P raw | R-S posterior | R-P posterior |
|---:|---:|---:|---:|---:|
| 0 | 0.30729167 | 0.36979167 | 0.34583333 | 0.39583333 |
| 1 | 0.30729167 | 0.25130208 | 0.34583333 | 0.30104167 |
| 2 | 0.24348958 | 0.15494792 | 0.29479167 | 0.22395833 |
| 5 | 0.12369792 | 0.076822917 | 0.19895833 | 0.16145833 |
| 10 | 0.06640625 | 0.31380208 | 0.153125 | 0.35104167 |
| 20 | 0.071614583 | 0.0234375 | 0.15729167 | 0.11875 |
| 40 | 0.0091145833 | 0.010416667 | 0.10729167 | 0.10833333 |
| 80 | 0.0013020833 | 0.015625 | 0.10104167 | 0.1125 |
| 160 | 0.0065104167 | 0.0026041667 | 0.10520833 | 0.10208333 |
| 320 | 0 | 0.0013020833 | 0.1 | 0.10104167 |
| 480 | 0 | 0 | 0.1 | 0.1 |

### sentinel trajectories

| Update | R-S raw | R-P raw | R-S posterior | R-P posterior |
|---:|---:|---:|---:|---:|
| 0 | 0.041666667 | 0.35546875 | 0.13333333 | 0.384375 |
| 1 | 0.02734375 | 0.24088542 | 0.121875 | 0.29270833 |
| 2 | 0.037760417 | 0.12109375 | 0.13020833 | 0.196875 |
| 5 | 0.03515625 | 0.075520833 | 0.128125 | 0.16041667 |
| 10 | 0.032552083 | 0.36588542 | 0.12604167 | 0.39270833 |
| 20 | 0.036458333 | 0.013020833 | 0.12916667 | 0.11041667 |
| 40 | 0 | 0.0078125 | 0.1 | 0.10625 |
| 80 | 0 | 0.0065104167 | 0.1 | 0.10520833 |
| 160 | 0.0013020833 | 0.0013020833 | 0.10104167 | 0.10104167 |
| 320 | 0 | 0 | 0.1 | 0.1 |
| 480 | 0 | 0 | 0.1 | 0.1 |

### stage-b trajectories

| Update | R-S raw | R-P raw | R-S posterior | R-P posterior |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.051879883 | 0.029411765 | 0.07823989 |
| 1 | 0.013916016 | 0.064086914 | 0.042509191 | 0.08972886 |
| 2 | 0.039550781 | 0.061279297 | 0.066636029 | 0.087086397 |
| 5 | 0.065795898 | 0.062866211 | 0.091337316 | 0.088579963 |
| 10 | 0.07019043 | 0.068725586 | 0.095473346 | 0.094094669 |
| 20 | 0.1328125 | 0.073364258 | 0.15441176 | 0.098460478 |
| 40 | 0.15344238 | 0.074707031 | 0.17382813 | 0.099724265 |
| 80 | 0.18835449 | 0.17419434 | 0.20668658 | 0.19335938 |
| 160 | 0.25036621 | 0.25036621 | 0.26505055 | 0.26505055 |
| 320 | 0.43835449 | 0.38439941 | 0.4419807 | 0.39119945 |
| 480 | 0.45788574 | 0.45703125 | 0.46036305 | 0.45955882 |

R-S F3: [profiles/seed-11-R-S.json](profiles/seed-11-R-S.json).

R-P F3: [profiles/seed-11-R-P.json](profiles/seed-11-R-P.json).

The JSON companion contains per-item counts, own-baseline curves, half-lives, fixed first-attainment rows, early authoritative failure summaries, and final validation counts.

## Source block 29: NO_MATCH

Matching unavailable; no Stage-B evidence collected.
