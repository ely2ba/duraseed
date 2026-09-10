# Replay training-order replication

Fixed 2026-09-09 before this replication starts. The owner authorized a full
additional replay seed in parallel with the dense-retention run ("do it") and
accepted direct between-arm matching within three percentage points. This is a
new follow-up informed by the completed results; it does not replace them.

## What changes

Use source block 11's existing 579 shared prompts, exact solver traces and exact
archived policy traces, original M0, and original evaluation manifests. Acquisition
order seed **47** sorts the paired records by
`(order_key(47, task_id), task_id)` using the existing replay ordering function.
Both arms see the same new prompt order. No traces or tasks are regenerated.
Sampling uses experiment seed 47 with the existing purpose/checkpoint namespaces,
shared between arms. The source block remains 11 in the record.

This repeats training with a different fixed data order on the same archived
corpus. It is not a new policy-data collection or an independent family block.
Both arms restore the same M0 weights with fresh optimizers; the changed seed
does not create a different M0 initialization. Stage-B data order is unchanged.

## Acquisition and prospective selection

R-S and R-P each run all 294 acquisition updates: LR 1e-4, batch 32,
completion-only per-example-mean SFT, rank-32 LoRA. Preserve the replay optimizer,
renderer, masks and full-trace max_length. Save sampler/state pairs and measure
the original 96 targeted +96 sentinel cadence items once at updates
10,20,...,290,294. No early performance stop.

After **both** arms finish, let `q` be the smaller of the two arms' maximum
cadence targeted raw-Pass@1 scores. In each arm nominate the three checkpoints
closest to `q`, breaking ties by earlier update. Save all six nominees before
any high-draw candidate assessment. Each gets the original 256 targeted
validation items x16 draws. The anchor guides nomination only: it is not an
eligibility band, historical target, or additional matching gate.

A candidate pair qualifies exactly when its two high-draw raw success means
differ by at most **3/100**. Select lexicographically by
`(-min(R-S_score,R-P_score), abs(R-S_score-R-P_score),
R-S_update+R-P_update, R-S_update, R-P_update)`, using exact rational counts.
There is no historical-score requirement or new fixed acquisition floor.
The capability-first ranking avoids choosing a lower-scoring match when a
higher-scoring eligible pair is available. Cadence maxima are noisy and the six
assessments do not guarantee overlap. If none qualifies, record `NO_MATCH` and
do not start Stage B. No replacement nominees, widening, or extra acquisition.

## Full matched continuation and measurements

Use the unchanged replay Stage-B recipe: original 4,096 MAPS shortest2_cap2
records in original order, LR 3e-4, batch 32, 480 updates, fresh Adam on entry
and full optimizer continuity thereafter. Both stages use beta1=0.9, beta2=0.95,
epsilon=1e-12, weight_decay=0, grad_clip_norm=0. Evaluation temperature=1.0,
top_p=0.95; TCES cap 4,096 and MAPS cap 128.

| Measurement | Items x draws per arm | Stage-B updates |
| --- | --- | --- |
| TCES monitor | 384 x4 | 0,1,2,5,10,20,40,80,160,320,480 |
| MAPS validation | 512 x16 | same grid |
| TCES full validation / F3 | 512 x16 | 0 and 480 |

Update-0 observations supply the full descriptive F3 profile and are counted
once. Retain both complete trajectories and the existing F3 quantities. Save
sampler/state pairs at each nonzero Stage-B measurement; TTL remains 30 days.

Primary: R-P minus R-S targeted unconditional raw-Pass@1 trapezoidal AUC on
[0,1,2,5,10,20], divided by 20; retain the original positive directional
hypothesis. Key F2: absolute raw MAPS AUC0-480, with baseline, endpoint, curves,
and the existing secondary gain-AUC0-40 decomposition. First-crossing half-life
remains a descriptive companion. Use the existing 50,000 paired-item trajectory
bootstrap with seed-47 contrast names. No pooled seed-level interval or changed
endpoint/window. The separate dense-grid follow-up does not change these rules.

## Execution

Separate run root, Tinker session, launchd job and billing record; no changes to
the running dense experiment or historical evidence. Eight rolling item workers
keep the independently seeded sampling calls unchanged. All original
restore/update calls are retained. A process-local cache reuses
identical supervised datums across checkpoints, clearing on a fresh arm/stage
or a changed renderer/tokenizer. No batch, token, target or loss mask changes.
The existing ledger uses the exact finite schedule (at most 1,548 updates,
282,880 completions and
80 sampler/state checkpoint pairs); no new engineering smoke. Respect pending
requests without retries or cancellation. Missing original states, integrity
failures and nonfinite applied updates remain failures, not rerolls. There is
no authorized replacement run. Analysis runs locally after completion.
