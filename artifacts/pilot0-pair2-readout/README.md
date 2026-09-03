# Pair-2 scientific readout

Run: `pilot0-pair2-seed29-20260830T204211Z`. Seed: 29. Local artifacts only.

Matching selected B-S Stage-A update 40 and B-G update 20; each had 17/96 targeted exact successes on the cadence matching panel. Both Stage-B schedules ran 480 updates.

Sources: [terminal result](/Users/elyb/Documents/DuraSeed-v1/runs/pilot0/pilot0-pair2-seed29-20260830T204211Z/result.json), [matching record](/Users/elyb/Documents/DuraSeed-v1/runs/pilot0/pilot0-pair2-seed29-20260830T204211Z/seed-29/matching.json), [exact readout data](readout.json), [both F3 profiles](F3.md).

All sequences below follow this Stage-B update grid:

`[0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480]`

Rates are proportions, not percentages. Printed sequences use six decimal places; readout.json retains the stored precision, exact success/trial counts, and every source path. Pass@k uses the stored unbiased item-averaged estimator. Posterior means use the item-level Jeffreys estimator `(successes + 0.5)/(draws + 1)`, averaged equally over items.

The zero-success posterior floor is 0.1 with four draws and 0.029411764705882353 with 16 draws; it is not observed Pass@1. a_monitor has 192 targeted and 192 sentinel items × 4 draws; a_validation has 256 targeted and 256 sentinel items × 16 draws; b_validation has 512 MAPS items × 16 draws.

## F1 · retention trajectories

![F1 retention](F1-retention.svg)

### B-S

Targeted a_monitor:

- Exact successes/trials: `[135/768, 75/768, 20/768, 4/768, 3/768, 0/768, 0/768, 0/768, 0/768, 0/768, 0/768]`.
- Pass@1: `[0.175781, 0.097656, 0.026042, 0.005208, 0.003906, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`.
- Pass@4: `[0.473958, 0.312500, 0.104167, 0.020833, 0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`.
- Posterior mean: `[0.240625, 0.178125, 0.120833, 0.104167, 0.103125, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000]`.
- Posterior absolute AUC: `0.10047960069444446`.

Targeted a_validation, selected checkpoint → Stage-B update 480:

- Exact successes/trials: 727/4096 → 0/4096.
- Posterior mean: `0.19646139705882354` → `0.029411764705882353`; change `-0.16704963235294118`.
- Pass@1: `0.177490234375` → `0.0`.
- Pass@4: `0.4400218921703297` → `0.0`.
- Pass@16: `0.796875` → `0.0`.

Sentinel a_monitor:

- Exact successes/trials: `[46/768, 36/768, 17/768, 12/768, 3/768, 0/768, 0/768, 0/768, 0/768, 0/768, 0/768]`.
- Pass@1: `[0.059896, 0.046875, 0.022135, 0.015625, 0.003906, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`.
- Pass@4: `[0.192708, 0.161458, 0.088542, 0.062500, 0.015625, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`.
- Posterior mean: `[0.147917, 0.137500, 0.117708, 0.112500, 0.103125, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000, 0.100000]`.
- Posterior absolute AUC: `0.10035481770833335`.

Sentinel a_validation, selected checkpoint → Stage-B update 480:

- Exact successes/trials: 214/4096 → 0/4096.
- Posterior mean: `0.07858455882352941` → `0.029411764705882353`; change `-0.04917279411764706`.
- Pass@1: `0.05224609375` → `0.0`.
- Pass@4: `0.16644702953296706` → `0.0`.
- Pass@16: `0.3984375` → `0.0`.

### B-G

Targeted a_monitor:

- Exact successes/trials: `[152/768, 154/768, 106/768, 39/768, 19/768, 0/768, 0/768, 3/768, 0/768, 0/768, 0/768]`.
- Pass@1: `[0.197917, 0.200521, 0.138021, 0.050781, 0.024740, 0.000000, 0.000000, 0.003906, 0.000000, 0.000000, 0.000000]`.
- Pass@4: `[0.546875, 0.578125, 0.458333, 0.197917, 0.098958, 0.000000, 0.000000, 0.015625, 0.000000, 0.000000, 0.000000]`.
- Posterior mean: `[0.258333, 0.260417, 0.210417, 0.140625, 0.119792, 0.100000, 0.100000, 0.103125, 0.100000, 0.100000, 0.100000]`.
- Posterior absolute AUC: `0.10199761284722224`.

Targeted a_validation, selected checkpoint → Stage-B update 480:

- Exact successes/trials: 827/4096 → 0/4096.
- Posterior mean: `0.21943933823529413` → `0.029411764705882353`; change `-0.19002757352941177`.
- Pass@1: `0.201904296875` → `0.0`.
- Pass@4: `0.566313959478022` → `0.0`.
- Pass@16: `0.90234375` → `0.0`.

Sentinel a_monitor:

- Exact successes/trials: `[152/768, 155/768, 122/768, 30/768, 19/768, 1/768, 0/768, 5/768, 0/768, 0/768, 0/768]`.
- Pass@1: `[0.197917, 0.201823, 0.158854, 0.039062, 0.024740, 0.001302, 0.000000, 0.006510, 0.000000, 0.000000, 0.000000]`.
- Pass@4: `[0.604167, 0.515625, 0.468750, 0.140625, 0.088542, 0.005208, 0.000000, 0.026042, 0.000000, 0.000000, 0.000000]`.
- Posterior mean: `[0.258333, 0.261458, 0.227083, 0.131250, 0.119792, 0.101042, 0.100000, 0.105208, 0.100000, 0.100000, 0.100000]`.
- Posterior absolute AUC: `0.10228407118055556`.

Sentinel a_validation, selected checkpoint → Stage-B update 480:

- Exact successes/trials: 866/4096 → 0/4096.
- Posterior mean: `0.22840073529411764` → `0.029411764705882353`; change `-0.19898897058823528`.
- Pass@1: `0.21142578125` → `0.0`.
- Pass@4: `0.5723944024725275` → `0.0`.
- Pass@16: `0.90625` → `0.0`.

## F2 · learning curves

![F2 learning](F2-learning.svg)

Baseline-relative means subtract the same arm's update-0 score. The headroom-normalized series divides that difference by `1 - baseline + 1e-12`. AUC uses trapezoidal integration on the linear update grid, divided by 480; raw-gain AUC is absolute AUC minus the update-0 posterior mean. The figures use a transformed x-axis only for display.

### B-S

- Exact successes/trials: `[0/8192, 26/8192, 453/8192, 569/8192, 598/8192, 587/8192, 634/8192, 595/8192, 604/8192, 1125/8192, 2200/8192]`.
- Absolute Pass@1: `[0.000000, 0.003174, 0.055298, 0.069458, 0.072998, 0.071655, 0.077393, 0.072632, 0.073730, 0.137329, 0.268555]`.
- Absolute Pass@4: `[0.000000, 0.012416, 0.188619, 0.233958, 0.246702, 0.248124, 0.262281, 0.251431, 0.252212, 0.351411, 0.445970]`.
- Absolute Pass@16: `[0.000000, 0.046875, 0.466797, 0.564453, 0.605469, 0.613281, 0.640625, 0.626953, 0.636719, 0.691406, 0.697266]`.
- Pass@1 minus own baseline: `[0.000000, 0.003174, 0.055298, 0.069458, 0.072998, 0.071655, 0.077393, 0.072632, 0.073730, 0.137329, 0.268555]`.
- Absolute posterior mean: `[0.029412, 0.032399, 0.081457, 0.094784, 0.098116, 0.096852, 0.102252, 0.097771, 0.098805, 0.158663, 0.282169]`.
- Posterior mean minus own baseline: `[0.000000, 0.002987, 0.052045, 0.065372, 0.068704, 0.067440, 0.072840, 0.068359, 0.069393, 0.129251, 0.252757]`.
- Headroom-normalized posterior relative series: `[0.000000, 0.003078, 0.053622, 0.067353, 0.070786, 0.069484, 0.075047, 0.070431, 0.071496, 0.133168, 0.260417]`.

- `absolute_auc`: `0.14901625689338233`.
- `final_score`: `0.2821691176470588`.
- `headroom_normalized_gain_auc`: `0.12322887073850937`.
- `initial_score`: `0.029411764705882353`.
- `raw_gain_auc`: `0.11960449218749997`.

### B-G

- Exact successes/trials: `[438/8192, 630/8192, 409/8192, 561/8192, 575/8192, 614/8192, 624/8192, 1135/8192, 1404/8192, 2708/8192, 3020/8192]`.
- Absolute Pass@1: `[0.053467, 0.076904, 0.049927, 0.068481, 0.070190, 0.074951, 0.076172, 0.138550, 0.171387, 0.330566, 0.368652]`.
- Absolute Pass@4: `[0.186267, 0.246686, 0.168534, 0.235287, 0.242898, 0.257673, 0.255565, 0.365275, 0.388201, 0.544554, 0.588303]`.
- Absolute Pass@16: `[0.486328, 0.535156, 0.421875, 0.583984, 0.619141, 0.644531, 0.611328, 0.685547, 0.681641, 0.806641, 0.798828]`.
- Pass@1 minus own baseline: `[0.000000, 0.023438, -0.003540, 0.015015, 0.016724, 0.021484, 0.022705, 0.085083, 0.117920, 0.277100, 0.315186]`.
- Absolute posterior mean: `[0.079733, 0.101792, 0.076402, 0.093865, 0.095473, 0.099954, 0.101103, 0.159812, 0.190717, 0.340533, 0.376379]`.
- Posterior mean minus own baseline: `[0.000000, 0.022059, -0.003332, 0.014131, 0.015740, 0.020221, 0.021369, 0.080078, 0.110983, 0.260800, 0.296645]`.
- Headroom-normalized posterior relative series: `[0.000000, 0.023970, -0.003620, 0.015356, 0.017104, 0.021973, 0.023221, 0.087016, 0.120599, 0.283396, 0.322347]`.

- `absolute_auc`: `0.25622642367493875`.
- `final_score`: `0.37637867647058826`.
- `headroom_normalized_gain_auc`: `0.19178461818539294`.
- `initial_score`: `0.07973345588235295`.
- `raw_gain_auc`: `0.1764929677925858`.

## Stored paired contrast · B-S minus B-G

- F2 raw-gain Stage-B AUC difference: `-0.056888475605085836`.
- F1 update-480 targeted posterior retention difference: `0.0`.

## F3 · pre-Stage-B profiles

[Both profiles, including panel metrics and full-source links](F3.md). The MAPS update-0 values above are the baseline Stage-B task performance before Stage-B training.

## Local checks

Both F1/F2 cells reproduced exactly from the saved evaluation item counts using the existing reducer. Both stored F2 AUCs reproduced from the 11-point linear grid. No new sampling, training, or remote calls were used for this readout.
