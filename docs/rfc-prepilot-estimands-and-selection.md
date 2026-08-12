# RFC: pre-Pilot estimands and checkpoint selection

Status: accepted before Pilot 0
Date: 10 August 2026

## Problem

The pre-calibration plan made headroom-normalized MAPS GainAUC and retention at
matched MAPS progress primary. It also selected the earliest Stage-A checkpoint
inside a narrow validation band without fixing a sufficiently precise,
independent sampling procedure.

Three problems follow:

1. Dividing gain by remaining score headroom assumes that headroom is a linear
   opportunity to learn. Stage-A zero-shot MAPS transfer is itself meaningful.
2. Requiring an origin to reach a MAPS level conditions retention on a
   post-treatment outcome. Methods that learn MAPS poorly may be selectively
   absent from the comparison.
3. A noisy earliest-in-band rule can select favorable monitor noise. Imposing a
   monotonic curve would instead erase real peaks or regressions.

Recent adjacent work also studies excessive SFT followed by RL
([arXiv:2606.09932](https://arxiv.org/abs/2606.09932)), circuit preservation
under SFT versus RL
([arXiv:2605.28860](https://arxiv.org/abs/2605.28860)), and SFT/RL
non-decoupling ([arXiv:2601.07389](https://arxiv.org/abs/2601.07389)). DuraSeed
therefore needs a narrower claim: future learnability under its fixed supervised
Stage-B probe, not general or subsequent-RL plasticity.

## Decision

- Primary future-learning endpoint: raw MAPS GainAUC.
- Primary stability endpoint: retained Stage-A capability after the common
  fixed Stage-B budget.
- Always report zero-shot MAPS, headroom-normalized and absolute AUC, final
  score, and time-to-threshold.
- Matched-progress retention is a sensitivity. Freeze only levels reached by at
  least 95% of pilot origins; report reach rate and retention conditional on
  reaching.
- Locate the earliest apparent Stage-A crossing with the cheap monitor, then
  independently re-evaluate its predecessor, the crossing checkpoint, and its
  successor on all 512 validation items with fresh fixed seeds. Use 16
  samples/item, extending every candidate to 32 if panel-mean sampling SE
  exceeds 0.0075. Select the in-band real checkpoint closest to target, tie to
  earlier. Do not smooth or interpolate checkpoints.
- Freeze the M0-derived Stage-A prompt allocation; monitor boundary movement but
  never rescan into a method-specific curriculum.
- Require concrete seeds for scientific generations, record B-O coverage at
  each refresh, and keep completion-only forward KL and greedy decoding as
  secondary diagnostics.
- Keep at least 20% of the then-uncommitted grant balance outside any
  confirmatory launch reservation.

## Alternatives rejected

- Monotonic smoothing: Stage-A acquisition may genuinely regress.
- Matched-progress stability as primary: it conditions on reaching.
- Rank sweeps, RL Stage B, a second model, or extra method cells before the core
  B-S/B-G result: these spend independent-seed budget without repairing the
  primary inference.

## Effect on existing work

No paid result informed this change. Pilot 0 and Stage A have not started. The
completed engineering smoke, M0 calibration, TCES cap calibration, broad
boundary scan, and the boundary refinement running when this RFC was accepted
do not estimate the changed endpoints and remain valid. No rerun is required.

The public protocol and strict pre-Pilot config are updated
together. Exact TCES/MAPS semantics, family eligibility, panels, acquisition
methods, splits, and budgets through Phase 4 are unchanged.
