# Can a behavioral clone inherit a checkpoint's learning future?

Approved 2026-09-10, before this experiment's training or sampling. Current
execution status is in [STATUS.md](../STATUS.md). This specification governs
one new experiment; the completed Pilot and replay evidence remain unchanged.

## Question and comparison

Can supervised training on one frozen RL teacher's outputs produce a student
that reproduces its arithmetic behavior on unseen problems and responds
similarly to the same subsequent training?

Compare checkpoint replacement with checkpoint repeatability: how much does
the continuation trajectory change when the student replaces the teacher,
relative to rerunning either checkpoint? This is one teacher acquisition and
one student acquisition, followed by repeated continuations from the two fixed
origins. It is not a training-seed population estimate.

## 1. Teacher

Use the existing format-capable M0, `Qwen/Qwen3.5-9B-Base`, rank-32 LoRA, and
block-11 acquisition procedure. The selected historical Pilot teacher states
are no longer retained on Tinker, so this is a fresh realization rather than
a restoration of B-G@30.

- Run exactly **30 RL updates**, using the original acquisition prompt order.
- The teacher is **update 30**, designated before training.
- Preserve the original verifier-rewarded group-relative RL recipe: learning
  rate `1e-5`, 16 prompt groups per update, eight draws per prompt group, and
  the existing reward, advantage, and optimizer rules.
- Restore M0 weights with a fresh optimizer. No alternate checkpoint is chosen
  from subsequent continuation behavior.
- Retain sampler and trainable-state checkpoints on the original ten-update
  cadence, with 30-day retention for this experiment.

## 2. Unfiltered endpoint corpus

Select **800 distinct prompts unused during this teacher's RL acquisition**
from the existing prepared pool: 400 boundary, 200 intermediate, and 200
broad. Keep them disjoint from evaluation items. Sentinel families remain
excluded from acquisition and distillation training.

Sample eight responses per prompt from the frozen update-30 teacher:
**6,400 completions**, temperature `1.0`, top-p `0.95`, response cap `4096`.
Keep every completion, including wrong, malformed, repetitive, and capped
responses. There is no correctness, likelihood, length, or quality filter.

The corpus contains the actual sampled token sequences and observed
termination information. A capped completion receives no synthetic ending.
This is distillation of the specified capped sampler, not full-logit
distillation or a claim to reproduce the teacher's unrestricted distribution.

## 3. Student acquisition

Start one student from the same M0 with fresh optimizer and unchanged adapter
configuration. Use learning rate `1e-4`, batch size 32, and exactly 294 updates
unless an execution failure prevents completion. Use one fixed corpus
permutation, cycling as necessary. The fixed configuration records the
permutation and sampling namespaces before launch.

Mask prompt tokens from the supervised loss. For each sampled response use
the **sum** of response-token negative log-likelihood, divided by one fixed
corpus-wide mean response length. This preserves sampled-sequence likelihood
weighting while choosing a common loss scale. Do not normalize each response
by its own length or weight it again by its teacher probability.

This loss differs from historical replay's per-example mean loss. The
experiment tests the specified endpoint-distillation procedure, not the
isolated causal effect of replacing evolving-policy samples with endpoint
samples.

Evaluate and retain student checkpoints at updates
`10,20,...,290,294`. Complete the fixed acquisition schedule before nomination.

## 4. Behavioral clone selection

### Nomination

On the existing cadence panel, rank retained student checkpoints by

\[
d_{\mathrm{nom}}=
\frac{|\Delta p_{\mathrm{targeted}}|}{0.03}+
\frac{|\Delta p_{\mathrm{sentinel}}|}{0.05},
\]

where each difference is from the frozen teacher and each score is raw
Pass@1, including invalid completions in the denominator. Nominate the three
lowest-distance checkpoints, breaking ties by earlier update.

### Selection profiles

Evaluate the teacher and all three nominees on the existing profile panel:
256 targeted and 256 sentinel problems, 16 draws per problem. Apply:

| Measurement | Allowed student–teacher difference |
|---|---:|
| Targeted raw Pass@1 | ±3 percentage points |
| Sentinel raw Pass@1 | ±5 percentage points |
| Fraction solved at least once in 16 draws | ±10 percentage points |
| Median response length | Student/teacher ratio in [2/3, 1.5] |
| Validity rate | ±10 percentage points |
| Distinct verified strategy-family count | Student/teacher ratio in [2/3, 1.5] |

Coverage, length, validity, and strategy gates apply separately to both roles.
Coverage is observed success at least once in 16 draws; it is not posterior
coverage at a threshold. Validity and strategy families use the existing
profile reducers. If the teacher's strategy count is zero, require the
student's count to be zero. No new strategy classification is introduced.

Among passing nominees, select the smallest mean absolute per-item Pass@1
discrepancy, weighting the two roles equally. Break ties by nomination
distance, then earlier update. MAPS outcomes are not consulted.

### Item-disjoint confirmation

Before billable work, prepare a separate panel of 256 targeted and 256 sentinel
problems using the same profile construction. Exclude all training, selection,
and monitoring items. Label it as newly constructed confirmation, not as the
historical sealed final test.

After selection is irrevocable, collect two independent teacher profiles
(`T-a`, `T-b`) and one selected-student profile (`S`) on that panel, each with
16 draws per problem and independent sampling streams. The student must pass
the same profile gates against `T-a`.

For each role define

\[
D(X,Y)=\operatorname{mean}_i|\widehat p_X(i)-\widehat p_Y(i)|.
\]

Also require `D(S,T-a) − D(T-b,T-a) ≤ 0.03` separately for each role. This is
an observed item-agreement gate relative to the teacher's own sampled
repeat discrepancy, not a confidence-interval equivalence test.

If no nominee passes selection, or the selected nominee fails confirmation,
stop before Stage B. Retain and report all measured candidate profiles and
failed criteria. No fallback candidate, wider gate, extra draws, extended
student training, or replacement teacher is allowed.

## 5. Common continuation

Only after confirmation passes:

| Run | Starting checkpoint | Final Stage-B update |
|---|---|---:|
| T1 | Frozen teacher | 480 |
| T2 | Frozen teacher | 20 |
| S1 | Selected student | 480 |
| S2 | Selected student | 20 |

T1 and S1 are designated as the long continuations before observing outcomes.
All four receive the unchanged MAPS `shortest2_cap2` data and order, batch 32,
learning rate `3e-4`, and a fresh Adam optimizer at entry. Preserve beta1 `0.9`,
beta2 `0.95`, epsilon `1e-12`, weight decay `0`, gradient clipping `0`, loss
construction, and adapter configuration. Preserve optimizer state throughout
each continuation. Repeat continuations do not create new acquisition seeds.

Use independent evaluation sampling streams with identical decoding settings
and draw counts. Every continuation receives a fresh update-0 evaluation;
baseline samples are not shared. Record the actual platform-supported random
seeds without claiming deterministic equality where the platform does not
provide it.

### Measurements

| Measurement | Items × draws per run | Updates |
|---|---:|---|
| TCES monitor | 384 × 4 (192 items per role) | All four: every integer 0–20; T1/S1 also 40,80,160,320,480 |
| MAPS validation | 512 × 16 | All four: 0,1,2,5,10,20; T1/S1 also 40,80,160,320,480 |
| TCES full endpoint validation | 512 × 16 | T1/S1 at 480 |

TCES uses temperature `1.0`, top-p `0.95`, and cap `4096`; MAPS uses the
unchanged settings with cap `128`. Report MAPS baseline only after clone
selection, as an outcome rather than a matching gate. Record both absolute
scores and own-baseline gains. Retain required trainable states and sampler
checkpoints for 30 days, and archive adapters locally.

## 6. Primary analysis: dense trajectory separation

The primary trajectories are targeted raw Pass@1 over updates 0–20.
For each pair of runs compute

\[
d(X,Y)=\frac{1}{20}\operatorname{Trapz}_{0:20}
|\widehat p_X(t)-\widehat p_Y(t)|.
\]

Apply the trapezoidal rule to absolute differences at the 21 registered
integer updates. Do not substitute a different continuous crossing formula.
Report all six distances: four teacher–student distances, teacher–teacher,
and student–student. The primary summary is

\[
E=\frac{d(T1,S1)+d(T1,S2)+d(T2,S1)+d(T2,S2)}{4}
-\frac{d(T1,T2)+d(S1,S2)}{2}.
\]

`E` measures observed cross-checkpoint separation beyond the average observed
within-checkpoint repeat separation. Report negative values without clipping.
Show full curves alongside the six distances and `E`; describe whether cross
distances consistently exceed the within-checkpoint repeat distances or overlap.

This analysis is descriptive and estimation-first. Do not attach a binary
inheritance/equivalence score or treat a bootstrap interval containing zero as
evidence of equivalence. Absolute sampled trajectory distances include
evaluation noise; two repeats do not supply a population repeatability bound.
The offline precision check motivated retaining the nonlinear repeat-adjusted
distance without a formal bootstrap equivalence decision.

## 7. Registered secondary results

- The same six distances and `E` for sentinel retention.
- Signed targeted and sentinel raw retention AUC over 0–20, divided by 20.
- First downward crossing of half each run's own update-0 raw Pass@1,
  linearly interpolated between observed checkpoints; report unobserved
  crossings as censored.
- MAPS baseline, absolute trajectories, and own-baseline gain trajectories.
- MAPS raw-gain AUC over 0–40 and 0–480, and update-480 endpoints, for T1/S1.
- Pre-continuation profiles, item coverage overlap, response-length
  distributions, validity, and verified strategy counts.
- Selected student update, acquisition exposure, and adapter factor norms.

Use the established paired whole-item bootstrap for signed-AUC and endpoint
contrasts, retaining all observations from a sampled item together. Intervals
describe item uncertainty conditional on the observed continuations; they do
not quantify acquisition-seed or teacher-population uncertainty. No secondary
outcome changes selection, duration, or primary analysis.

## 8. Execution and scope

Parallelize independent rollout groups within a fixed-policy RL update and
collect all groups before applying the update. Run independent continuations
concurrently where service capacity permits. Reuse prepared supervised datums
without changing token sequences, loss masks, data order, or optimizer steps.
Concurrency does not change draw counts or random-seed assignments.

The owner authorized the package on 2026-09-10 following the approximately
`$2,147` full-cap planning estimate and its stated use of the prior reserve.
The final finite-manifest launch ceiling is **`$2,148.44`**, including 123
sampler/state checkpoint-pair storage allowances at `$0.25` each and 30
ephemeral sampler snapshots. This is a `$1.44` correction to the approximate
estimate, with no change to the experimental matrix. Teacher acquisition
started on 2026-09-10. Observed token costs and checkpoint-storage allowances
remain distinguishable from settled charges.

Do not retry ambiguous in-flight updates or sampling calls automatically.
Missing required data/state, service failure, nonfinite applied updates,
unrecoverable integrity failures, and budget exhaustion are stopping conditions,
not permission to reroll. No additional teacher, student recipe, acquisition
seed, model, task, correct-only arm, gauge intervention, or outcome-dependent
rescue is included. There is no automatic fallback experiment.

The final report includes selection and confirmation results, four dense
trajectories, cross-versus-repeat distances, signed retention and learning
results, profiles, and actual spending. Publish a failed clone attempt if
confirmation is unavailable or unsuccessful. Keep all historical results
separate and unchanged.
