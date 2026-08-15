# RFC: pre-Pilot estimands and checkpoint selection

Status: accepted before Pilot 0; claim and reporting scope clarified before
Stage A
Date: 10 August 2026; clarified 15 August 2026

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

Two further ambiguities require explicit treatment. First, equality on one
aggregate Stage-A score does not imply identical family coverage, reliability,
output validity, diversity, or sampling behavior. Second, `B-S/B-G` changes the
data source and exploration distribution as well as the update objective, so it
cannot identify an objective-only “SFT versus RL” effect.

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
- Report monitor RetentionAUC over the already-sampled `a_monitor` Stage-B
  trajectory as a secondary, monitor-population summary. Do not describe it as
  a full-validation AUC; the fixed-budget `a_validation` endpoint remains
  primary.
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
- Describe the matched follow-up exactly as matching targeted-panel
  exact-success within the frozen band. Do not imply globally matched behavior.
  At every selected pre-Stage-B origin, report target and sentinel means,
  per-family accuracies, the full Cover curve, invalid-output rate, completion
  length, Pass@k for supported `k`, verified strategy-family diversity, and
  token surprisal. These are descriptive profiles, not extra matching gates.
- Report Stage-A capability curves against cumulative acquisition cost and the
  relevant cumulative token axes. Preserve a separate resource vector: teacher
  examples, cached and uncached prefill tokens where available, sampled tokens,
  training tokens, optimizer updates, and fixed checkpoint/storage charges. Do
  not collapse unlike resources into a claim of equal compute.
- Report Pass@k only when `k` is no greater than the number of independent draws
  collected for that item; do not extrapolate unsupported larger-k values.
- Freeze the M0-derived Stage-A prompt allocation; monitor boundary movement but
  never rescan into a method-specific curriculum.
- Require concrete seeds for scientific generations, record B-O coverage at
  each refresh, and keep completion-only forward KL and greedy decoding as
  secondary diagnostics.
- Keep at least 20% of the then-uncommitted grant balance outside any
  confirmatory launch reservation.
- State the first `B-S/B-G` result as a comparison of two complete acquisition
  procedures, not as an intrinsic loss-only or representation-level effect.
- Keep equal crossed panels, strict split separation, adequate breadth and
  precision, and no use of method outcomes in task construction as the panel
  invariants. The current 12/12 design remains preferred, but any different
  equal size must be justified prospectively by feasibility and power before
  Stage A; it cannot be chosen to rescue a method result.
- If a reproducible `B-S/B-G` effect appears, prioritize supervised replay of a
  prospectively frozen, verifier-correct subset of `B-G`-generated rollouts
  before broad method expansion. Freeze that follow-up separately; it is not a
  Pilot-0 arm and receives no method identifier here.
- If the effect survives independent seeds, prioritize a same-model
  higher-rank or full-weight replication before a broad model-size sweep to
  test whether the result is specific to rank-32 LoRA competition.

## Alternatives rejected

- Monotonic smoothing: Stage-A acquisition may genuinely regress.
- Matched-progress stability as primary: it conditions on reaching.
- Matching on every observed pre-B feature: coverage, diversity, entropy, and
  related differences may be consequences or mechanisms of the acquisition
  procedure. Report them rather than automatically matching them away.
- Rank sweeps, RL Stage B, a second model, or extra method cells before the core
  B-S/B-G result: these spend independent-seed budget without repairing the
  primary inference.
- Broad method expansion immediately after an effect: the frozen-rollout
  supervised-replay control attacks the leading data-versus-objective ambiguity
  more directly.

## Effect on existing work

No paid result informed this change. Pilot 0 and Stage A have not started. The
completed engineering smoke, M0 calibration, TCES cap calibration, broad
boundary scan, and the boundary refinement running when this RFC was accepted
do not estimate the changed endpoints and remain valid. No rerun is required.

The public protocol and strict pre-Pilot config are updated
together. Exact TCES/MAPS semantics, family eligibility, panels, acquisition
methods, splits, and budgets through Phase 4 are unchanged. The clarification
adds reporting from already-declared checkpoints and generations; it does not
change Pilot-0 execution, create a new method identifier, or resolve the still-
unpublished panel assignment. Any panel-construction amendment remains a
separate prospective decision before Stage A.
