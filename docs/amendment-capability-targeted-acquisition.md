# Capability-targeted acquisition amendment

Status: **FROZEN**. This is an explicitly outcome-aware successor
to terminal run `acquisition-calibration-stage-a-correction-20260818T134223Z`,
written before Pilot 0 or sealed-test access. Its only purpose is to make the
capability-matched B-S/B-G origins needed for F1 retention, F2 Stage-B learning,
and F3 pre-B behavior.

## Gates

- **Tier 1 — infrastructure, hard:** finite optimization metrics, clean leakage,
  and durable, authenticated evidence/checkpoints.
- **Tier 2 — acquisition viability, hard:** paired targeted capability-gain
  approximate 95% lower bound `> 0`; family reachability `>= 0.60`;
  verifier-valid answer-tag rate `>=` paired M0 minus `0.10`; absolute targeted
  length-stop rate `<= 0.50`; and detected-loop fraction among capped targeted
  outputs `<= 0.15`. The existing detector is unchanged: a loop has at least
  `0.80` exact agreement at a lag from 4 through 256 in the final 1,024 token
  IDs. The `0.15` threshold is the rounded midpoint between the calibration
  anchors B-G@50 `0/56` and B-S@50 `17/58 = 0.293`.
- **Tier 3 — descriptive, never eligibility:** paired length-stop delta,
  completion length, token surprisal, verified diversity, sentinel movement,
  supported Pass@k, and family spread. Paired length is moved here because caps
  were strongly stratum-dependent (broad-random `47.8%` versus boundary
  `26.9%`), so the delta mixes behavior with the fixed prompt mixture. The
  frozen loop detector directly distinguishes nondegenerate capped work from
  looping, while every cap and length remains visible in F3. Sentinel movement
  was a hard calibration gate but is descriptive here because held-out-family
  movement is part of the phenomenon under study; gating on it would match away
  that effect.

B-G is frozen without further calibration at LR `1e-5`, 50 updates, group size
8, the existing 4,096-token cap, and every other existing optimizer/sampling
setting. Its existing no-reroll execution rule for zero-mixed or nonfinite
completed updates remains in force.

### When each gate is evaluated

All Tier-2 hard decisions occur only on a 192-target-sample evaluation: a
threshold confirmation or the epoch-cap evaluation, which is always 192
targeted samples. Capability-gain approximate-95% CI floor, family
reachability, verifier-valid-tag retention, absolute length-stop rate, and loop
fraction are decided only there. A loop fraction with denominator zero (`0/0`)
passes.

The 96-target plus 96-sentinel cadence evaluations are monitoring only, except
for two cost/format tripwires: absolute length-stop rate `<= 0.50` and
verifier-valid-tag retention `>=` paired M0 minus `0.10`. A cadence fails when
either guard fails, and branch (ii) fires only after two consecutive failed
cadences. Loop fraction is recorded but never decides at cadence: its typical
15--30 capped-output denominator is too small, and the calibration's update-10
replay would read `3/15 = 0.20` and spuriously stop. Capability CI and family
reachability are never evaluated at cadence.

## One B-S dose run

B-S stays at LR `1e-4`, starts fresh weights-only from M0 with a fresh optimizer,
and repeats the correction run's canonical schedule: 1,568 presentations of
1,552 distinct verified solver traces per epoch. Freeze `E = 6` epochs (`294`
updates), `C = 10`, and
`theta = 19/96`. Evaluate one paired seeded completion on each of 96 targeted
and 96 sentinel items at updates `10,20,...,290`. At the epoch cap `294`, run
the final 192-target-sample evaluation instead.
Whenever a cadence evaluation has at least `19/96` targeted exact successes and
fewer than three confirmations have run, pause training and run a
192-target-sample confirmation at that checkpoint; confirmation must reach the
equivalent `38/192` and pass Tiers 1–2. If confirmation is below
`38/192`, record it without overwriting any earlier confirmation, continue
training, and leave the threshold stop armed. At most three confirmations may
run. After a third failed confirmation, launch no more confirmations and run to
the epoch cap, subject only to the frozen cadence tripwires. The epoch-cap
192-target evaluation is the final Tier-2 decision. Ten-update cadence
preserves the observed B-G@10 anchor, supports later matching, and bounds
overshoot to less than one fifth of an epoch.

Every branch is terminal: (i) confirmed threshold plus clean Tiers 1–2 proceeds
to Pilot 0; (ii) two consecutive cadence-tripwire failures, or a hard Tier-2
failure at a qualifying confirmation or the cap, ends the solver-SFT route;
(iii) a clean epoch-cap miss ends it as dose-limited. A confirmation below
`38/192` is the explicitly nonterminal exception described above. No branch
authorizes another amendment.

## Pilot 0 delta

For each paired seed, run both methods fresh from M0. The frozen Pilot B-S
recipe is LR `1e-4`, cadence evaluations and checkpoints every 10 updates, and
training through the full 294-update cap regardless of where the dose run
stopped. The dose run establishes that the overlap zone is reachable and
validates the gate set live; it does not select Pilot duration. B-G uses its
already-frozen recipe. Matching is post hoc: select the nearest targeted
exact-success pair whose paired 95% intervals overlap; ties go to the earlier
checkpoints, and no overlap is reported as unavailable without seed
replacement. Compute the complete Tier-3 profile at every selected checkpoint,
then apply the unchanged Stage-B recipe.

Tinker sampler-only paths are sufficient for evaluation but **not** for
trainable Stage-B initialization. Each cadence point must therefore retain the
sampler path plus its paired trainable state path; Stage B restores the latter
weights-only (`full_state=False`) with a fresh optimizer. No Stage-A optimizer
state enters Stage B.

The Stage-B recipe and outcome roles are fixed in
[PROTOCOL.md, “Stage-B probe calibration”](../PROTOCOL.md#stage-b-probe-calibration)
and [“Outcomes”](../PROTOCOL.md#outcomes). The measurement schedule assumed by
the `338,821,120`-sample-token Pilot ceiling is pinned here: Stage B uses
`shortest2_cap2`, LR `3e-4`, batch 32, and updates
`[0,1,2,5,10,20,40,80,160,320,480]`; at every point both methods evaluate all
512 `b_validation` items with 16 independent samples/item and the 128-token
MAPS cap. At each of the ten post-origin points, both methods evaluate the
frozen crossed 384-item `a_monitor` manifest with four independent samples/item
and the 4,096-token TCES cap; the origin monitor evidence is reused from the
pre-B evaluation. Both methods evaluate all 512 `a_validation` items with 16
independent samples/item immediately before Stage B and at update 480. Report
Pass@k at `k in {1,4}` for four-draw monitor evidence and at `k in {1,4,16}`
for 16-draw evidence. Pilot Stage-B reporting omits `k=32`; no 32-draw Stage-B
evaluation is included or authorized by this amendment. These details pin the
previously implicit components; no Stage-B measurement component remains to be
defined after freeze.

## Implementation scope

Phase 2 implements only the dose-run path: the tiered reducer with the timing
rules above, the six-epoch schedule, cadence evaluations, theta stop and
confirmation continuation, and exactly one trainable-state save at the
triggered stop checkpoint or epoch cap. Restore that state weights-only with a
fresh optimizer and authenticate the restored checkpoint without further
training; this is the live validation of the save-to-weights-only-restore path
required by Pilot. All ten `pilot0_*` modules and cadence state-checkpointing
move to Phase 2b, authorized only by branch (i). The matching rule is already
frozen above, so deferring its implementation creates no later discretion.
Phase 2 adds only two unit tests: theta-stop behavior including continuation
after a failed confirmation, and gate timing/tier classification.

## Cost and authority

Pinned rates per million tokens are `$0.66` prefill, `$1.995` sampled, and
`$1.463` trained; cached prefill is `$0.132`. The ceilings below assume every
completion uses all 4,096 tokens and B-S reaches the latest possible successful
checkpoint.

| bounded path | prefill | sampled | trained | fixed storage | ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| B-S dose run | 824,685 | 25,559,040 | 1,898,772 | `$1.50` | **`$55.82`** |
| Pilot Stage A, one paired seed | 1,797,891 | 55,705,600 | 28,953,108 | `$7.80` | **`$162.48`** |
| Full Pilot, one paired seed, including unchanged Stage B and F1–F3 evaluations | <=29,171,075 | 338,821,120 | 33,850,128 | `$11.60` | **`<=$756.33`** |

Two paired Pilot seeds therefore require a planning ceiling of `$1,512.66`.
The human explicitly accepted that ceiling in the `FROZEN` reply, so it
supersedes the stale `$600` contract. These ceilings assume 100% cap utilization,
versus roughly 55% realized utilization in calibration. Before the dose launch,
reconcile the terminal calibration's session
`11c608b5-339f-5253-85be-d505ba751a36` with a lag-cleared raw export using the
existing `calibration-reconcile` command. The launch condition is exactly:
reconciled **actual** lifetime spend plus the exact Phase-2 preflight-computed
dose ceiling must be `<= $300`; the table's `$55.82` planning estimate is not
the launch cap. Conservative floors understate spend and cannot support this
cap check.

This amendment is single-use and exhausts design-edit authority until Pilot 0
data exists. It does not reopen LRs, warm starts, tasks, panels, prompts,
verifiers, decoder, caps, group size, sealed tests, or the Stage-B recipe.
