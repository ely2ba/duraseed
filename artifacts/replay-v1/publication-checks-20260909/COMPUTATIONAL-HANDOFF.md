# DuraSeed computational handoff — 9 September 2026

Completed computations only. The paper, frozen protocol, selection records, and original observations have not been edited. No new training was performed.

R-S and R-P are the shared-prompt SFT replay arms using solver traces and archived correct policy traces, respectively. B-S and B-G are the original Pilot SFT and RL procedures.

## Half-life uncertainty

50,000 NumPy PCG64 percentile replicates; identical sampled item indices for both arms and all checkpoints; each item's four realized draws kept together

Treats evaluation items as independent exchangeable clusters. Conditional on selected checkpoints, training seeds, families, realized draws, and grid interpolation. Does not measure training-seed or selection uncertainty, or uncertainty from unobserved times between checkpoints. No family-cluster adjustment or multiplicity correction.

Half-life uses each resampled arm's own update-0 raw Pass@1 and the existing first-downward-crossing definition. Interpolation is unchanged. All 50,000 replicates were finite in every reported comparison; none were discarded.

### Targeted

| Comparison | First arm: updates [95% CI] | Second arm: updates [95% CI] | Second − first [95% CI] |
|---|---:|---:|---:|
| replay-seed11-targeted (R-S, R-P) | 4.250000 [3.556250, 5.245902] | 1.689189 [1.445312, 2.000000] | -2.560811 [-3.546606, -1.819799] |
| pilot-pair1-targeted (B-S, B-G) | 2.664286 [1.940787, 3.437500] | 4.104651 [3.669811, 4.535398] | 1.440365 [0.534588, 2.306267] |
| pilot-pair2-targeted (B-S, B-G) | 1.136364 [0.893333, 1.377053] | 3.343284 [2.507692, 3.964789] | 2.206920 [1.327799, 2.897482] |

### Sentinel

| Comparison | First arm: updates [95% CI] | Second arm: updates [95% CI] | Second − first [95% CI] |
|---|---:|---:|---:|
| replay-seed11-sentinel (R-S, R-P) | 28.571429 [0.761879, 34.285714] | 1.527174 [1.297468, 1.740260] | -27.044255 [-32.742489, 0.813946] |
| pilot-pair1-sentinel (B-S, B-G) | 6.315789 [1.875000, 8.571429] | 4.110714 [3.702128, 4.539286] | -2.205075 [-4.525641, 2.189942] |
| pilot-pair2-sentinel (B-S, B-G) | 1.684211 [1.055556, 3.875000] | 3.500000 [2.976744, 3.934579] | 1.815789 [-0.389046, 2.543531] |

## Update-10 re-evaluation

Same saved sampler path, 384 monitor items, four draws per item, recorded seeds, role-colon rendering, temperature 1, top-p 0.95, and 4,096-token cap. A new client session was used. Both roles contain 768 completions.

| Role | Original correct / 768 | Recheck correct / 768 | Recheck − original Pass@1 [paired item-bootstrap 95% CI] |
|---|---:|---:|---:|
| targeted | 241 | 229 | -0.015625 [-0.049479, 0.018229] |
| sentinel | 281 | 282 | 0.001302 [-0.037760, 0.040365] |

| Role / observation | Mean tokens | Median tokens | Cap reached / 768 | Valid output / 768 |
|---|---:|---:|---:|---:|
| targeted / original | 1339.522 | 570.5 | 144 | 288 |
| targeted / recheck | 1376.099 | 528.0 | 145 | 276 |
| sentinel / original | 1344.812 | 738.5 | 118 | 329 |
| sentinel / recheck | 1433.423 | 783.0 | 145 | 330 |

Valid output means valid answer tag, syntax, and lexing together. Invalid outputs remain failures. Full failure-code counts are in `recheck-comparison.json`.

Matched prompts/settings/seeds: 1,536/1,536. Identical completion token sequences: 455/1,536; identical completion text: 455/1,536; identical correctness labels: 1135/1,536.

### Single-point AUC sensitivity (not a replacement primary result)

| Role | Original R-S AUC | Original R-P AUC | R-P AUC using recheck at u10 only | Difference from R-S |
|---|---:|---:|---:|---:|
| targeted | 0.114941 | 0.176204 | 0.170345 | 0.055404 |
| sentinel | 0.034538 | 0.188607 | 0.189095 | 0.154557 |

The existing 0–20 grid and raw-Pass@1 normalization are unchanged; u10 has weight 0.375. This explicitly post-hoc calculation changes only the u10 observation. The original registered results remain available and unchanged.

## Replay adapter geometry

Existing Pilot geometry reducer, 249 adapted modules per checkpoint, rank 32 and alpha 32 (scale 1). These are the full saved LoRA factors, with no M0 subtraction. Frobenius norms aggregate squared module norms; BA spectra are block-diagonal unions of module spectra, not a full-network operator.

| Checkpoint | ‖A‖F | ‖B‖F | ‖BA‖F | σ1 | Stable rank | Entropy effective rank |
|---|---:|---:|---:|---:|---:|---:|
| seed-11-R-S-stage_a-u220 | 52.564260 | 15.570208 | 10.286665 | 6.441737 | 2.550018 | 4771.200195 |
| seed-11-R-P-stage_a-u20 | 51.660042 | 6.012484 | 3.576050 | 2.689161 | 1.768371 | 5636.273235 |
| seed-11-R-P-stage_b-u10 | 52.126001 | 12.594039 | 7.820221 | 5.470817 | 2.043308 | 5577.324642 |

Selected replay R-S/R-P B-factor norm ratio: 2.589646. Per-module/per-layer norms and spectra are in `geometry.json`. Downloaded adapters are retained locally.

The downloaded stored u10 adapter differs from the stored selected R-P adapter; parameter differences are in `recheck-comparison.json`. Download metadata and current tensor identity do not retrospectively prove which weights a historical request served.

## Usage and files

Recheck: 203,112 observed prefill tokens, 2,157,713 sampled tokens, zero training tokens. Local token-priced cost: $4.43869136. This is not a settled provider invoice; checkpoint download/analysis generated no new model tokens or retained remote checkpoints.

Machine-readable outputs: `half-life-uncertainty.json`, `geometry.json`, `recheck-comparison.json`. All confidence intervals here describe evaluation-item uncertainty, not replication across training seeds.

Reproduction from the repository root (local analysis only):

```sh
PYTHONPATH=src .venv/bin/python tools/replay_half_life_uncertainty.py --output artifacts/replay-v1/publication-checks-20260909
PYTHONPATH=src .venv/bin/python tools/replay_publication_summary.py --recheck runs/replay-v1/replay-u10-recheck-20260909T114500Z --output artifacts/replay-v1/publication-checks-20260909 --geometry-archive runs/replay-v1/publication-geometry-20260909
```
