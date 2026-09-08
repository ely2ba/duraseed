# DuraSeed supervised trace replay — replay-v1

Prepared 2026-09-05. Scientific design fixed before new outcomes; **not a launch
authorization**. This is the bounded package requested in
`DURASEED_FOLLOWUP_SPEC.md`, with the user's later instruction to omit unnecessary
process and drop gauge work. Pilot 0, its protocol, original analyses and
prospective prediction remain unchanged. No gauge experiment, additional seed,
model, task, online rollout generation or outcome-dependent rescue is included.

## Question and scope

Does the source of successful acquisition traces change subsequent skill
retention when both acquisition arms use supervised cross-entropy on exactly
the same prompts, followed by identical MAPS training? R-S uses deterministic
solver traces; R-P uses previously generated, verifier-correct B-G traces.
This intervention bundles trace content, length, format and strategy. It is not
an isolated intervention on reinforcement versus supervised learning, nor a
general test of model plasticity. Matching one score does not establish global
behavioral equivalence or equal selected training dose.

The two source blocks are the completed Pilot-0 seeds 11 and 29. Their
calibration, software checks, retrospective analyses and prospective Pair-2
prediction are distinguished in the accompanying evidence audit. No Pilot
outcome is reclassified as confirmatory evidence for this new intervention.

## Existing-data analysis

First reproduce the four original F1/F2 cells and all 28 original uncertainty
intervals with their original estimands and random seeds. Retain those outputs.
Extend the analysis separately using unconditional raw Pass@1, item-averaged
Jeffreys scores explicitly labeled as posterior, absolute AUC, own-baseline
gain AUC, and endpoint. For grid t ending at T, AUC is trapezoidal integral/T;
gain AUC = absolute AUC − baseline. Report that decomposition for each contrast.

Use all registered checkpoints in training order, not sorted downstream scores,
for `(MAPS raw Pass@1, targeted TCES raw Pass@1)` paths. First-attainment
sensitivities are exactly MAPS thresholds 0.06, 0.07, 0.10 and 0.20. Interpolate
both update and TCES within the same first upward-crossing bracket. A baseline
already at/above a threshold is `baseline_exceeded`; never crossing is
`not_reached`. Compare post-training crossings only when both arms qualify.
Retain reversals, all thresholds, unavailable rows and observed brackets.

Half-life is the first downward crossing of half that arm's own raw baseline,
linearly interpolated. Zero baselines are undefined; non-crossings remain
right-censored. TCES is total exact performance, not survival restricted to
newly acquired items. Report unconditional correctness and conditional
correctness among output-valid completions with denominators. Disjoint original
verifier failure codes remain separate from overlapping cap, tag and syntax
indicators; never repair historical outputs or relax a verifier.

New uncertainty uses 50,000 paired item-cluster trajectory percentile resamples
with NumPy PCG64: unsigned big-endian first eight SHA256 bytes of
`duraseed-replay-v1|pilot-audit|<contrast>`. Each resampled item retains both arms,
all checkpoints and its realized draw records. New contrasts are B-G minus B-S;
original intervals keep their original B-S minus B-G orientation. Intervals are
pointwise and conditional on these runs/items/draws, not training-seed population
intervals. No optional family-block sensitivity is added. The reproducible
notebook and Markdown are `artifacts/replay-v1/pilot-audit/`.

## Frozen shared-prompt corpora

Authenticate all 6,400 archived acquisition draws per block: 50 completed B-G
updates × 16 groups × 8 draws. Include updates 1 through 50, inclusive. Verify
the generating sampler predates its optimizer update and that tasks belong to
the original block's `a_rl_train` manifest. Re-run the historical exact verifier
on the stored full text with the original stop normalization; any disagreement
is an integrity block. Keep only correct, non-length-stopped completions below
their recorded cap. Do not strip derivations, repair tags, shorten, or rank by
quality/length.

For every eligible task choose the smallest SHA256 of UTF-8
`duraseed-replay-v1|<pair_seed>|<task_id>|<sample_id>`, with the complete sample ID
as tie-breaker. Reconstruct the original deterministic solver trace for the
same task and intended family. Both arms contain exactly the same task IDs.
Order them by SHA256 of `duraseed-replay-order-v1|<seed>|<task_id>` (task-ID
tie-break). Retain the original sentinel/leakage contract; no rebalancing.

| Block | Archived eligible draws | Shared prompts | Targeted / intermediate / broad-random | Maximum full rendered tokens |
| --- | ---: | ---: | --- | ---: |
| 11 | 1,479 | 579 | 331 / 135 / 113 | 4,195 |
| 29 | 1,416 | 557 | 325 / 133 / 99 | 4,217 |

All 36 acquisition families occur in each resulting corpus. Each block sampled
800 unique prompts before success conditioning. Full exclusion/selection
records and corpus bytes are retained privately under
`runs/replay-v1/preparation/block-{11,29}/`. Fewer than 32 unique prompts would
be `DATA_BLOCKED`, with no replacement block or new samples.

Use the pinned role-colon renderer and local tokenizer files associated with
the public Qwen revision `68c46c4b3498877f3ef123c856ecfde50c39f404` for offline
preparation only. All 12,800 archived token arrays reproduce their stored
normalized text. This bounded compatibility does **not** identify Tinker's
original upstream weight/tokenizer revision. New training must restore the
authenticated Tinker M0 path, not substitute the public base model. Before
training, the provider tokenizer must reproduce every frozen full datum's IDs,
shifted targets, lengths and completion-only mask; otherwise stop.

The follow-up-only supervised `max_length` is 4,217 full rendered tokens.
Inputs/targets shift by one; no completion token is truncated. Global Pilot
defaults remain unchanged. Per-example completion weights sum to one.

## Stage A: exactly four supervised runs

| Block | Arms | Origin | LR | Batch | Updates | Prompt presentations per arm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 11 | R-S, R-P | original M0, weights only | 1e-4 | 32 | 294 | 9,408 |
| 29 | R-S, R-P | original M0, weights only | 1e-4 | 32 | 294 | 9,408 |

Both procedures use rank-32 LoRA and completion-only, per-example-mean supervised
cross-entropy. Fresh Adam on entry: beta1=0.9, beta2=0.95, epsilon=1e-12,
weight_decay=0, grad_clip_norm=0, constant learning rate. At update u and batch
offset j, use ordered corpus index `((u−1)*32+j) % N`. No shuffle or early
scientific gate. Nonfinite applied updates count as failures, never rerolls.

At updates 10,20,…,290,294 save sampler and trainable-state checkpoints and
evaluate the original 96 targeted + 96 sentinel cadence items once each. TCES
cap=4,096, temperature=1.0, top_p=0.95. Retain every cadence point.

## Nomination and matching

After **all four** Stage-A runs finish, nominate exactly three checkpoints per
arm nearest the original target score: 31/96 for block 11, 17/96 for block 29.
Use cadence raw targeted success; ties go to earlier updates. Freeze all twelve
nominees before any high-draw assessment. No replacements.

Each nominee receives exactly 16 draws on the original 256 targeted
`a_validation` items (4,096 completions), TCES cap 4,096. These are candidate
assessments, not an independent held-out estimate. Among at most nine pairs per
block require both raw means within absolute 0.03 of that block's target and
within 0.03 of one another. Select lexicographically by
`(absolute_between_arm_gap, summed_absolute_target_distance,
summed_updates, R-S_update, R-P_update)`. Use exact rational arithmetic on counts.

No eligible pair is `NO_MATCH` for that block. It receives no Stage-B spending;
the other block continues as frozen. Record both block decisions before any
new Stage-B outcome is collected. No threshold, nominee, duration or endpoint
changes after seeing results.

Selected checkpoints receive a separate descriptive F3 profile: all 512
`a_validation` items ×16 and all 384 `a_monitor` items ×4. These required samples
are the Stage-B update-0 TCES origins, reused exactly once. F3 is never a
rematching gate; report any mismatch. Supported Pass@k uses existing draws only
and k∈{1,4,16} not exceeding each panel's draw count. Report target/sentinel and
family accuracy, coverage at thresholds {0.01,0.02,0.05,0.10,0.25,0.50,0.75},
invalidity, length, supported Pass@k, verified strategy diversity and surprisal.
MAPS update-0 draws below supply pre-training new-task performance.

## Stage B: at most four matched follow-ups

Restore selected Stage-A weights with a **fresh optimizer** at entry; retain
optimizer state on subsequent resumes. Use original `b_train` 4,096 tasks in
the original datum order, shortest2_cap2 solver programs, LR 3e-4, batch 32,
480 updates, same Adam parameters as above. The datum index is
`((u−1)*32+j) % 4096`. No newly generated training corpus.

| Measurement | Items × draws | Updates | Cap |
| --- | --- | --- | ---: |
| MAPS `b_validation` | 512 ×16 | 0,1,2,5,10,20,40,80,160,320,480 | 128 |
| TCES `a_monitor` | 384 ×4 | same grid; update 0 reused from F3 | 4,096 |
| TCES `a_validation` | 512 ×16 | before training (F3 reuse), 480 | 4,096 |

Sampler/state checkpoint pairs accompany the ten nonzero Stage-B points. No
sealed test or `a_ood` item is opened. Evaluation randomness uses the existing
explicit seed derivation with block seed and namespace
`replay-v1.<stage>.<purpose>.<update>`, shared across arms at a common point;
stage∈{stage_a,stage_b}, purpose∈{cadence,candidate,a_validation,a_monitor,
b_validation}. This pairs the realization where the platform permits, not a
claim of bitwise cross-model equality. Existing immutable generation records
use neutral `method=null`; arm-bearing labels and `replay-identity.json` bind
R-S/R-P without changing or misusing the legacy B-S/B-G schema.

## Frozen estimands and reporting

Primary: per-block R-P minus R-S targeted unconditional raw-Pass@1 retention
AUC over [0,1,2,5,10,20], trapezoidal integral divided by 20. Directional
hypothesis: positive. Key F2 result: **absolute** raw MAPS AUC over 0–480,
alongside baseline, endpoint and full curves. Historical early gain AUC0–40 is
secondary and is always accompanied by absolute AUC and baseline decomposition.
Report both blocks separately; no training-seed population interval or post-hoc
combined score. All matching/unavailable outcomes are retained.

Follow-up item-trajectory bootstrap uses the same 50,000-resample algorithm,
namespace `duraseed-replay-v1|followup|<contrast>`, with contrast names including
block seed, population, metric and R-P-minus-R-S. No new endpoints or windows
are selected after outcomes. F3 and engineering diagnostics are not primary
success criteria. Negative, null, opposite-direction and unavailable results
are reported without inventing a favorable mechanism narrative.

## Spend, retention and execution boundary

The manifest-derived full-cap schedule has 64,930,768 prefill, 887,095,296 sample
and 41,778,640 training tokens. No cache discount is assumed. At the verified
current rates $0.66/$1.995/$1.463 per million respectively, token ceiling is
$1,873.73157272. Exactly 120 Stage-A +40 Stage-B checkpoint pairs are reserved.
Each has a 30-day TTL; proposed bounds are 2GB state +0.5GB sampler. At
$0.10/GB-month, that is $40 before any provider backup liability.

Thus the **conditional** main ceiling is $1,913.73157272; add up to $20 for one
disposable full-length engineering check and 10% of main ($191.373157272) for
documented recovery: $2,125.104729992, rounded up to $2,125.11. This is not yet
an unconditional invoice maximum: provider backup duration and size bounds,
and other committed account charges remain unverified. The 2026-09-05 console
balance is $3,743.52. After these uncertainties were disclosed, the owner
directed launch ("just start", reiterated subsequently). This later direction
authorizes execution with the disclosed storage allowances and unknown external
commitments; it does not turn them into verified facts. The operational package
cap remains $2,125.11, with per-call token/storage reservations unchanged.
No verified budget breach is waived. Recompute the preflight, never cut the schedule,
if service evidence changes financial assumptions. Final package ceiling must
be ≤$2,400 and preserve $1,343.74 of available credits after other commitments.

The only engineering smoke addresses the genuinely new full-length supervised
path: one frozen longest datum, one disposable M0 update, one state/sampler
save and both restore modes. No scientific evaluation, fifth acquisition arm,
or new sampling. Its artifact is explicitly engineering-only and outside all
scientific analyses. It runs only after the same package approval, ≤$20.

Journal requests and full-cap liabilities before dispatch. A pending request
blocks retry/relaunch; failed calls are not presumed free. Resume only from
authenticated durable checkpoints with correct optimizer continuity. Recovery
must preserve run identity, data order, masks, loss reduction, metrics and
sampling. No infrastructure change may reroll a scientific failure. Checkpoint
expiry, integrity loss, service limits or exhausted approved liability stop
the affected work; never substitute a base model or buy an outcome-dependent
rescue. No automatic TTL extension or unrelated experiment.

The approval string binds final protocol/config bytes and the exact ceiling:
`AUTHORIZE DURASEED REPLAY <protocol_sha256> <config_sha256> CEILING $<approved_total>`.
The owner's later plain-language launch direction substitutes for that literal
approval-string format for this launch only. The run records the actual words,
final protocol/config hashes, ceiling and unresolved assumptions. No fabricated
hash-form approval is recorded. No external GPU rental.

## Completion

After authorized execution, publishable local outputs include raw counts,
matched selections and all candidate assessments, failure/length diagnostics,
paired uncertainty, chronological plots, minimal reproduction instructions,
related-work verification and a complete manuscript draft. Keep public relative
paths and exclude keys, private billing identifiers, sealed tasks and large
adapter tensors. Reconcile settled and committed DuraSeed charges; account-wide
deltas are not DuraSeed attribution. End with one ICLR-main-track versus narrower
venue recommendation supported by actual evidence, not a new research program.
No public push, release, submission or further paid run is authorized here.
