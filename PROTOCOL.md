# DuraSeed pre-calibration protocol

Status: pre-pilot; remote boundary collection, the deterministic three-cohort
replay, and the amended panel freeze are complete. The original match-first
rule failed structurally, after which the prospectively accepted
de-duplicate-before-match rule reduced 49 observation-eligible families to 37
exact algebraic representatives. All 37 passed the unchanged 22/22 test
capacity requirement; two disjoint 12-family panels and 12 separate
intermediate families are now frozen. The archived and current production
split builders produced byte-identical train and gate manifests. Acquisition
calibration then made three bounded attempts to construct a common task-
specific teacher warm start. None produced a checkpoint passing the frozen
gates in both panel orientations. The final dose-8 M1 attempt was stopped at a
no-pending-call boundary once seed-17 update 14 could no longer pass format;
seed-37 update 14 was not launched. Before Stage A or Pilot 0, the shared
teacher warm start was prospectively retired. The core B-S and B-G procedures
now branch directly from the same M0. Their six-arm update-10 screen completed
on 18 August 2026 and recorded `no_eligible_learning_rate`. That terminal
decision is invalid because its outer-wrapper reducer contradicted the TCES
prompt, solver-teacher targets, and verifier; the raw evidence remains valid and
preserved. A frozen correction selects B-S `1e-4` and B-G `1e-5` from those
existing screens and permits exactly one continuous two-arm M0-to-update-50
finalization. Pilot 0 has not started. An
archived engineering-only Tinker smoke and raw M0 audit were executed on 9
August 2026, and the v1 live smoke gate passed on 13 August 2026. The raw audit
rejected the zero-update candidate on the declared format gates, so conditional
task-agnostic format training was required. A common trainable M0 has now been
selected. The renderer and common TCES completion cap are now fixed at
`role_colon` with the DuraSeed answer/role stops and 4096 tokens. MAPS task
difficulty and its Stage-B recipe are fixed; TCES panel membership and the two
selected Stage-A learning rates are frozen, while duration and common RL
settings remain unresolved. Pilot 0 has not started, and
no result for the stability–future-learnability hypothesis has been produced.

This file is the public authoritative scientific contract, with
`duraseed_pilot_config.yaml` as its machine-readable counterpart. They become
a frozen pilot protocol only after empirical calibration of renderer, M0, task
difficulty, learning rates, durations, generation limits, and spend controls.
Exact task semantics, data separation, and already-declared scientific
invariants remain fixed now.

The pre-Pilot endpoint, checkpoint-selection, and claim-scope correction is
recorded in
[`docs/rfc-prepilot-estimands-and-selection.md`](docs/rfc-prepilot-estimands-and-selection.md).
The accepted teacher-exposure repair is recorded in
[`docs/rfc-teacher-exposure-progressive-repair.md`](docs/rfc-teacher-exposure-progressive-repair.md).
The subsequent dose-8 low-repetition amendment is recorded in
[`docs/rfc-teacher-seed-dose8-low-repetition.md`](docs/rfc-teacher-seed-dose8-low-repetition.md).
Its superseding direct-M0 decision is recorded in
[`docs/rfc-direct-m0-stage-a-origin.md`](docs/rfc-direct-m0-stage-a-origin.md).
The post-screen answer-tag correction and hard stop are recorded in
[`docs/rfc-stage-a-answer-tag-contract.md`](docs/rfc-stage-a-answer-tag-contract.md).

## Question

> When LLM post-training procedures reach comparable immediate capability, do
> they leave different stability–future-learnability profiles—different
> retention of the newly acquired skill and different efficiency in learning
> the next skill under a fixed supervised adaptation probe?

DuraSeed measures the joint profile using one controlled checkpoint lineage:
Stage-A capability acquisition, retention of that capability during Stage B,
and efficiency in learning Stage B.

“Plasticity” is used only as shorthand for supervised adaptation plasticity
under this fixed Stage-B probe. The study does not claim general or subsequent-
RL trainability.

## Methods and scope

| Method | Acquisition procedure | Scientific role | Execution phase |
|---|---|---|---|
| `G-U` | uniform-prompt RL | context/acquisition control | exploratory pilot |
| `B-S` | fixed offline solver-teacher SFT directly from M0 | core stability–future-learnability | first end-to-end result |
| `B-O` | refreshed verified current-policy SFT directly from M0 | conditional mechanism pilot | after a reproducible core result |
| `B-G` | boundary-weighted group-relative RL directly from M0 | core stability–future-learnability | first end-to-end result |

The first complete result is `B-S` versus `B-G` with two paired pilot seeds.
`G-U` remains an optional prompt-allocation control and `B-O` a conditional
mechanism pilot. The old `G-B` identifier is retired because direct-M0 `B-G`
is the same procedure; `R-G` and its targeted-versus-random teacher-allocation
question are retired with the task-specific seed. The final confirmatory
matrix and fresh seed count are chosen only after pilot variance, effect, and
cost are measured. Method breadth is cut before seed replication.

The first `B-S/B-G` estimand compares two complete acquisition procedures. It
does not isolate their update objectives: the procedures also differ in data
source, trajectory distribution, exploration, solution exposure, token and
compute use, and gradient statistics. Any first result is stated at that level
and is not generalized to an intrinsic “SFT versus RL representation” effect.

After that two-seed Pilot 0, variance reconnaissance extends B-S/B-G to a
total of 3--4 paired pilot seeds. Those seed-level variance and cost estimates
inform the mechanism/context pilot decision and any justified B-O or G-U runs.
Only after those gates determine the confirmatory matrix is the
design powered and frozen, preregistered, and run on fresh confirmatory seeds.

If `B-S/B-G` separates reproducibly, the first mechanism follow-up is
supervised replay of a prospectively frozen, verifier-correct subset of
`B-G`-generated rollouts. This directly probes the data-source-versus-objective
ambiguity and takes priority over broad method expansion. Its exact selection,
budget, and comparison must be frozen after the core effect is established and
before that follow-up runs; it is not part of Pilot 0 and has no method
identifier yet.

The ordered core comparison is `B-S/B-G`: fixed solver-teacher SFT versus
boundary-weighted group-relative RL as complete acquisition procedures. If
later justified, `B-G/G-U` tests prompt targeting, `B-O/B-S` tests refreshed
current-policy data, and `B-G/B-O` is a secondary objective comparison. These
optional comparisons are not a frozen confirmatory set.

## Common initialization

```text
Qwen/Qwen3.5-9B-Base
  -> zero-update rank-32 LoRA audit
  -> optional task-agnostic format training when the audit fails
  -> one M0
  -> every Stage-A branch
```

The raw audit checks answer-tag compliance, parser-invalid output, truncation,
renderer artifacts, and task success after excluding pure format failures. It
also checks zero-update LoRA sampling/log-probability agreement with the base
client within the service's measured numerical/stochastic contract.

If the audit passes, the verified zero-update LoRA state is M0. Otherwise a
format checkpoint is evaluated after whole-response outer whitespace removal.
It must achieve at least 97% canonical wrapper compliance, at least 95% exact
canonical response success, nonempty successful payloads, and authenticated
format-only training data on all 256 format items. Generated TCES and MAPS
results are later task-calibration diagnostics, not format-selection gates.
The earliest passing step is selected, with the smallest learning rate as the
tie-break. The post-audit, pre-full-confirmation correction is recorded in
[`docs/rfc-format-selection-estimand.md`](docs/rfc-format-selection-estimand.md).
The first complete fixed confirmation at that coordinate is the selection
draw. A later implementation or service failure may trigger materialization or
recovery, but it does not permit resampling the selection coordinate and
replacing the observed result.

Calibration outcome (9 August 2026): step 2 / `1e-4` was the first eligible
coordinate. Its frozen selection draw achieved 250/256 canonical format
successes (97.66%). Because the service rejected training restoration from the
sampler-only checkpoint, the same two updates were reconstructed from the raw
zero-state weights with a fresh optimizer. The reconstructed sampler achieved
251/256 canonical successes (98.05%) and was saved with its full state as the
non-expiring common M0. This is calibration evidence; Pilot 0 has not started.

The first selected-M0 TCES cap diagnostic sampled 64 items independently at
`1024`, `2048`, and `4096` tokens. It selected no cap: observed length-stop
rates were 53.1%, 50.0%, and 54.7%. Inspection showed that `role_colon`
advertised `"\n\nUser:"` while this checkpoint often emitted `"\nUser:"`, and
same-seed requests at different caps were not trajectory-identical. Those rows
remain valid diagnostics, but they cannot establish a nested cap comparison.

The corrected common sampling contract stops at the first `</answer>` or user
role boundary. Cap calibration uses one 4096-token reference trajectory on 128
fixed TCES validation items, then asks which smaller candidate cap would have
censored an answer-terminated trajectory. Persistent failure to emit a closing
answer tag is scored as task failure, not mislabeled as evidence for a larger
cap. Selection requires at least 64 answer-terminated trajectories, at most 5%
observed censoring across all 128 items, and no censored exact success. The
smallest passing cap is selected. Uncertainty and nontermination are reported;
this is an operational cap choice, not an equivalence claim.

Corrected calibration outcome (9 August 2026): 73/128 reference trajectories
emitted a closing answer tag and 8/128 were exactly correct. Candidate 2048
would have censored 4/128 answer completions, including one exact success;
4096 censored none (Wilson 95% interval 0–2.91%) and was selected. The run used
16,920 prompt and 164,216 sampled tokens, with an immediate token-cost estimate
of $0.33878 and conservative accounting of $1.50 pending billing. This remains
calibration evidence, not a Pilot-0 result.

Every active method declares the same M0 state. `B-S`, `B-G`, and any later
`G-U` or `B-O` arm branch weights-only directly from M0 with a fresh optimizer.
There is no task-specific teacher checkpoint between M0 and Stage A.
Interruption recovery uses full state; Stage B again restores only the selected
Stage-A weights so its optimizer is fresh.

Stage B continues the same rank-32 LoRA adapter. The resulting estimand is
conditional on continued shared-adapter low-rank adaptation; it is not presented
as full-weight or adapter-independent plasticity. If the core effect is
reproducible, a same-model higher-rank or full-weight replication is the first
generality test of this low-rank-space explanation. It does not precede the
core result.

## Crossed targeted/sentinel panels

After the M0 scan, deterministic matching with a frozen allocation seed divides
approximately 24 usable boundary families into Panel A and Panel B. Matching
uses M0 posterior mean, informative-group probability, tree depth, operator
multiset, noncommutative operation count, fractional-intermediate profile,
target magnitude, valid-family multiplicity, teacher-trace length, and
available disjoint instances.

Before matching, each family must materialize leakage-free prefixes of 16
`a_seed_train`, 8 `a_seed_gate`, 64 `a_rl_train`, 16 `a_monitor`, and 22
`a_validation` items. After the provisional 24-family match, every selected
family must additionally materialize 22 `a_test_single` items before membership
is frozen. Here “single” means the complete exact solution set contains exactly
one selected intervention family, not that the arithmetic task has only one
valid family globally. Candidate and protected-split numeric tasks are globally
disjoint. Ordinary splits probe a two-times prefix; the panel-aware
single-family filter may scan up to eight candidate indices per required item.
Generic `a_test_multi` strategy diversity and `a_ood` operand-count evaluation
are constructed separately. The Stage-A prompt-pool demand is checked again
after duration selection.

Panel A is targeted and Panel B sentinel for half the seed blocks; the assignment
is reversed for the other half. Targeted families may enter Stage-A teacher and
optimization data. Sentinel families enter none of it and are evaluated at M0,
post-A, and throughout Stage B. Existing family-disjointness leakage checks
fail closed on a sentinel/training overlap.

Panel membership is frozen by the completed amended three-cohort construction.
The allocation seed is fixed at
`6448342238137851489`. The current pilot design prefers two matched 12-family
panels so the target/sentinel cross-over is exactly symmetric. The scientific
invariants are equal crossed target/sentinel panels, adequate family breadth
and precision, strict split separation, and no use of method outcomes in task
construction. The number 12 is not itself an invariant. A smaller equal panel
size cannot silently replace the current design: it would require a prospective
power- and feasibility-based amendment before Stage A and without inspecting
`B-S/B-G` outcomes. Transition under the current design also requires 12
distinct non-panel intermediate families for the frozen Stage-A prompt pool.

Three-cohort freeze outcome (15 August 2026): all 115 finalists passed the
ordinary split-capacity audit and 49 were observation-eligible. Deterministic
matching selected 24 panel candidates, but the final selected-panel audit
passed only 12/24. Each failure was isolated to `a_test_single`: 0 of 22
required items were available after the full 176-index scan, while the other
five protected splits passed. The other 12 candidates supplied all 22/22 test
items. The old and current reducers and teacher-token maps agreed exactly. No
panel artifact was published, and the frozen outcome is
`confirmation_complete_panel_selection_unresolved`. This is structural
feasibility evidence, not a B-S/B-G outcome; no acquisition or Pilot execution
is authorized by it.

An outcome-blind follow-up checked every observation-eligible family against
all 49 candidates under the unchanged 22/22 `a_test_single` requirement.
Thirty-one passed, establishing that a valid 24-family capacity-qualified pool
exists. The failure pattern is consistent with exact algebraic duplicates
being protected together after matching. The accepted prospective cure
collapses the 49 fixed families into 37 exact rational-function classes, keeps
the frozen-rank representative of each class, and applies the unchanged 22/22
filter before matching. Matching receives only test-qualified representatives.
The intermediate pool is separate: it takes the top 12 ordinary five-split-
qualified representatives outside both panels and does not require
`a_test_single`. The amended freeze subsequently passed: all 37
representatives met the unchanged requirements, the matcher froze two
disjoint 12-family panels, and 12 separate ordinary-capacity-qualified
intermediates were retained. The production prompt and split build also passed
exact archived/current equivalence. The complete rule is recorded in
[`docs/rfc-boundary-panel-algebraic-deduplication.md`](docs/rfc-boundary-panel-algebraic-deduplication.md).

The first completed Stage-3 confirmation evaluated 38 finalists. Fifteen
passed every frozen observation and split-capacity gate, so it did not freeze
panels. Those 15 eligibility decisions are locked; the 23 Stage-3 failures are
not retested. Before any further family output, one final 128-family extension
was frozen. It takes exact-family ordinals 65 through 192 in deterministic
first-appearance order from the same authenticated `a_candidate` stream,
excluding the original 64-family cohort. It runs as two fixed operational
64-family subcohorts, ordinals 65–128 and 129–192, each using the unchanged
Stage-1, Stage-2, and Stage-3 evidence budgets and gates. Both subcohorts run
regardless of the first subcohort's yield, and the combined eligible pool is
formed only after both finish. The deterministic split-capacity audit occurs
after the Stage-2 gate and before paid Stage-3 sampling; each subcohort confirms
exactly its Stage-2 gate passers that also pass that audit. Known capacity
failures receive no paid confirmation samples.

Continuation requires at least 36 eligible families: the top 24 under the
existing confirmed-`I8`, capacity, and allocation-seed ranking enter unchanged
ten-covariate panel matching, while ranks 25–36 form the distinct non-panel
intermediate family pool required by Stage A. If the union remains below 36,
Phase 3 stops unresolved: there is no further block, failed-family retest,
threshold relaxation, panel or intermediate-pool shrinkage, or automatic
teacher-dose launch under that amendment. Its preferred design remains
symmetric 12/12; any later size change requires the separate prospective rule
above. The complete
pre-result amendment is recorded in
[`docs/rfc-boundary-family-extension.md`](docs/rfc-boundary-family-extension.md).

Before Stage-A outcomes, freeze an equal-weight Gower-style distance from each
sentinel family to its nearest targeted family using the ten matching
covariates. Sentinel gain versus this distance is descriptive only and cannot
change panel membership.

The frozen allocation seed is `6448342238137851489`, derived from root seed 11
under `family_selection.panel_allocation`. Once at least 36 families qualify,
the matcher retains the 24 with greatest confirmed `I8`, then verified split
capacity, and uses the seed only for exact ties. Closest Gower pairs are
oriented to minimize maximum and then mean panel imbalance across the ten
covariates; ranked eligible families 25–36 remain outside both panels as the
intermediate pool.

## Retired teacher warm-start calibration

The original design placed a common task-specific solver-teacher SFT checkpoint
between M0 and every core Stage-A branch. Three bounded pre-Pilot attempts found
no checkpoint that passed the frozen health gates in both crossed panel
orientations.

The first run selected dose 2 / `1e-4` at update 16 in seed 17, but the same
coordinate learned repetitive nontermination in seed 37 and failed format. A
progressive dose-2 repair found no joint 4/8/12-update pass: update 8 passed
format in both orientations but remained below the mixed-group gate, and the
complete seed-17 update-12 target assessment missed format by three responses
(`742/768` versus `745/768`). It was stopped before completing an assessment
that could not pass.

The final M1 amendment tested 96 distinct traces per orientation at dose 8 /
`1e-4`, with checkpoints 6/10/14. At update 6, seed 17 and seed 37 respectively
had `21/96` and `11/96` mixed groups and `663/768` and `711/768` valid-format
outputs. At update 10 they had `25/96` and `17/96` mixed groups; seed 17 missed
format at `741/768`, while seed 37 passed format but reached only 7/12 families.
At seed-17 update 14, 586 of the first 640 target samples had valid answer tags;
even 128 perfect remaining samples could reach only 714/768, below the required
745. M1 was stopped safely and seed-37 update 14 was not launched.

These observations are failed nuisance-recipe feasibility evidence, not a
B-S/B-G comparison and not evidence that SFT warm starts are generally
unstable. Before any Stage-A or Pilot outcome, the direct-M0 amendment retired
the task-specific seed rather than lowering a gate or opening another recipe
search. It also retired `R-G` and its teacher-allocation question. The old
teacher-dose, repair, and allocation records remain preserved as historical
artifacts but no longer gate the experiment.

The M0 boundary evidence supports direct RL without claiming certainty about a
live rollout. Across all 24 frozen panel families, predicted informative group-
size-8 probability ranges from 0.581 to 0.883, with median 0.739 and mean
0.7485; none is below 0.50. A separate held-out one-draw M0 check observed
17/192 verified successes across 10/24 family blocks. The scan predictions are
therefore treated as feasibility evidence, while the Stage-A B-G screen below
provides the realized rollout-health gate.

## Stage-A learning-rate and duration calibration

Stage-A calibration branches every B-S and B-G coordinate weights-only from the
same non-expiring M0 with a fresh optimizer. Both use the frozen 0.50/0.25/0.25
boundary/intermediate/broad-random prompt schedule. B-S trains on a fixed,
verified solver-teacher corpus; B-G samples its own attempts and applies the
common verifier-scored group-relative objective. There is no shared task-
specific training before this split.

B-S screens `[1e-4, 3e-4, 1e-3]`; B-G screens `[1e-5, 3e-5, 1e-4]`. All six
arms train to update 10 and are evaluated on the same 96 targeted and 96
sentinel monitor items with one paired seeded completion each. Sentinel
evidence is recorded but cannot select the learning rate. The smaller LR is
selected among eligible candidates within two paired sampling standard errors
of the best targeted gain.

The completed screen initially terminated with no eligible LR because the old
format reducer required the entire stripped completion to start with
`<answer>` and end with `</answer>`. That is not the TCES task contract: the
prompt requests a concise derivation before one final tag, solver-teacher
targets use that form, and the exact verifier permits free-form derivation
outside one well-ordered tag pair. Paired M0 verifier-valid tag rate was 69/96,
despite only 27/96 passing the erroneous outer-wrapper rule. The resulting
terminal decision is invalid calibration output; its immutable evidence is not
discarded or relabeled.

The corrected operational gate uses the verifier's `valid_answer_tag` field.
On the targeted panel, each arm's current valid-tag rate must be at least its
paired M0 rate minus 0.10. Invalid-tag and length-stopped samples continue to
receive zero reward
and remain reported outcomes. The unchanged health gates require at most 0.50
absolute length-stop rate, at most a 0.10 increase from paired M0, finite
metrics, and clean leakage. Every B-G update must contain at least one mixed
reward group, and an eligible B-G arm must average at least 0.20 mixed groups
across updates 1--10. No prompt, teacher, verifier, decoder, task, panel,
completion cap, or LR grid changed.

Applying that rule once to the sealed screens selects B-S `1e-4` under the
existing paired two-SE/smaller-LR rule and B-G `1e-5` as the only B-G arm that
also passes the unchanged length-stability gate. Because the screen saved only
sampler checkpoints, the selected trajectories cannot be resumed for training.
Exactly one finalization therefore recreates only these two branches
continuously from M0 through update 50; it does not rerun the grid. Update 25 is
a training-metric waypoint only and creates no evaluation checkpoint.

At update 50, the targeted panel uses two paired samples per item and the
sentinel panel one. B-G uses official importance sampling, group-mean
advantages without standard-deviation normalization, 16 prompt groups of eight
rollouts per actual update, and no KL penalty. Constant-reward groups are
recorded and skipped without resampling; an update with no mixed group fails the
bounded run. The final 50-update duration freezes only if the final reducer,
including the corrected paired valid-tag rule, passes both methods. If either
arm fails, the TCES acquisition route ends: there is no step 100, new LR,
warm-start, prompt, decoder, task, panel, completion-cap, or group-size rescue.
Any later study is a separately motivated redirect. The finalization is capped
at `$107.84`; conservatively charging the completed screen at its full local
committed ceiling gives a `$268.846932025` lifetime maximum, leaving
`$31.153067975` under the unchanged `$300` acquisition-calibration limit. This
is Phase-4 calibration, not Pilot 0.

The same acquisition-calibration freeze also selects one common RL
objective/configuration before any multi-method run. If the entropy-collapse
gate requires a fallback, that fallback applies to every RL cell. The initial
common configuration is the exact group-relative objective above. Its existing
pre-outcome duration gate requires the final-ten mixed-group rate to be at
least 0.20 and all optimization metrics to be finite. Token surprisal,
completion diversity, valid-family diversity, success diversity, and loss
health must also be present and structurally valid, but have no additional
outcome-tuned threshold. Failure hard-stops for a separately approved
protocol-wide fallback rather than choosing one after outcomes.

The `$25` engineering live smoke requires real, non-fabricated data, exact
online/offline reward parity, a verified stop contract, full-state resume, and
weights-only branching. Its 4096-token probe is a transport diagnostic, not a
new acquisition-cap selector. The accepted
[`common max-token ratification`](docs/rfc-acquisition-max-token-ratification.md)
independently fixes 4096 as the shared Stage-A completion cap for all methods.
A method-specific completion cap is forbidden.

## Stage-B probe calibration

The first fixed 512-item MAPS curves ran two preselected profile/LR recipes
through 320 supervised updates. `current_provisional` at `3e-4` rose from
18/512 (3.52%) at M0 to 149/512 (29.10%), with raw gain AUC 0.1151.
`moderate_candidate` at `1e-4` rose from 27/512 (5.27%) to 66/512 (12.89%),
with raw gain AUC 0.0397. These are calibration curves, not comparisons between
Stage-A methods. Neither recipe reached the provisional 40–80% terminal range,
so profile, learning rate, and duration remain unresolved.

Before collecting further evidence, the duration extension is narrowed to the
only recipe with strong late learning: `current_provisional` at `3e-4`. It
branches weights-only from the same M0 with a fresh optimizer, canonical cyclic
batch-32 data order, the same 512 validation items, and the fixed evaluation
grid `[0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480, 640]`. This calibration
selects a complete Stage-B recipe rather than claiming an isolated profile or
learning-rate effect. The selected duration is the earliest real checkpoint
with observed exact success in the inclusive 40–80% terminal band, at least
95% valid program syntax, and at most 5% length stops. If no checkpoint through
640 qualifies, the recipe remains unresolved and the grid is not expanded
automatically.

The completed extension remained below that frozen selection band. Exact
success was 160/512 (31.25%) at 480 updates and 171/512 (33.40%) at 640;
program syntax was 99.8% and 100%, respectively, with no length stops. Across
the predeclared 0–640 grid, raw GainAUC was 0.2181 and headroom-normalized
GainAUC was 0.2242. The curve demonstrates continued supervised learnability,
but no Stage-B duration was selected. These are calibration measurements, not
Pilot 0 results, and the duration grid is not extended automatically.

The prospective `current_provisional` `1e-3` arm also failed. Its best observed
exact success was 30/512 (5.86%) at 80 updates and it ended at 12/512 (2.34%)
at 640; no checkpoint passed the frozen gate. The authenticated two-arm
decision therefore selected no MAPS recipe and forbids an automatic extension.
This rules out another learning-rate arm on the same profile.

Before collecting any further MAPS result, one easier profile and one arm are
frozen. `shortest2_cap2` changes only `max_shortest_length` from five to two;
all moduli, constants, instruction-set width, latent-length controls,
ambiguity limits, and other generator settings remain those of
`current_provisional`. In the existing generator this also makes the explicit
task maximum program length two, so the name records both calibrated axes.
This is still the same MAPS DSL, parser, solver, verifier, and split contract.
The choice follows the completed `3e-4` terminal stratum: shortest-length-two
items reached 169/424 (39.86%), whereas shortest-length-three items reached
2/88 (2.27%); late syntax was 100% and length stops were zero.

The single arm uses fresh calibration seed 37, with a newly generated disjoint
4,096-item `b_train` and 512-item `b_validation`; `b_test` remains sealed. It
branches weights-only from the same M0 with a fresh optimizer, uses `3e-4`,
canonical cyclic batch-32 training, the unchanged 128-token cap and
`[0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 480, 640]` evaluation grid, and
sampler-only candidate checkpoints with seven-day TTL. The unchanged selection
rule is the earliest real checkpoint with 40–80% exact success, at least 95%
valid program syntax, and at most 5% length stops. The token-cost upper bound is
$33.5702656 under an exact $40 authorization and reservation.

The authenticated completed arm first passed at 480 updates: exact success was
208/512 (40.625%), valid program syntax was 510/512 (99.609375%), and there
were no length stops. The 640-update checkpoint also passed, with 241/512
(47.0703125%) exact success, 511/512 (99.8046875%) valid program syntax, and no
length stops. The frozen earliest-passing rule therefore selects
`shortest2_cap2`, `3e-4`, and 480 updates. Across the complete predeclared
0–640 calibration grid, absolute AUC was 0.30164642333984376, raw GainAUC was
0.25281829833984376, and headroom-normalized GainAUC was 0.2657966504103982.
These full-grid AUCs are calibration diagnostics, not Pilot outcomes, and the
profile change supports no cross-profile equivalence claim. Candidate
checkpoints retain their seven-day TTL; the selected object is the Stage-B
recipe. Pilot 0 has not started and `b_test` remains sealed.

## Outcomes

For item `i`, with `s_i` successes in `n_i` samples, the threshold-free estimate
is:

```text
p_i = (s_i + 1/2) / (n_i + 1)
```

Items are equally weighted within each panel. Primary acquisition summaries are
targeted-panel mean gain, sentinel-panel mean gain, and targeted-minus-sentinel
specificity. Posterior-soft `Cover@0.25` and the full Cover curve are key
secondary reliability measurements, always compared at identical sample
budgets.

At every declared Stage-A checkpoint, report target and sentinel capability
against update progress, cumulative acquisition cost, and the relevant
cumulative token axes. Cost never stands in for the underlying resource vector,
which is reported separately as teacher examples, prefill tokens (cached and
uncached where available), sampled tokens, training tokens, optimizer updates,
and any fixed checkpoint/storage charge. This makes compute efficiency visible
without pretending that an SFT training token and an RL rollout token are
interchangeable.

Stage B is MAPS only and continues the selected Stage-A LoRA weights with a
fresh optimizer. Let `u` be normalized Stage-B progress, `q(u)` the MAPS score,
and `q0=q(0)`. The primary future-learning endpoint is:

```text
GainAUC_raw
  = integral [q(u) - q0] du
  = absolute_auc - q0
```

Primary stability is retained targeted Stage-A capability on the full
`a_validation` population at the common fixed Stage-B budget. Let
`a_monitor(u)` be targeted-panel Stage-A capability on the smaller `a_monitor`
population along that normalized Stage-B trajectory. The already-sampled
monitor curve also reports:

```text
MonitorRetentionAUC = integral a_monitor(u) du
```

This secondary trajectory summary is explicitly monitor-population evidence;
it is not presented as a 512-item validation AUC. The full-validation
fixed-budget endpoint remains primary. Headroom-normalized gain and absolute
AUC are sensitivities; zero-shot `q0` is reported separately. Retention at
matched MAPS progress is a reach-aware sensitivity that reports reach rate plus
retention conditional on reaching.
The main presentation is the full stability–future-learnability frontier.
Sentinel change, A→B Cover change (`Cover_B - Cover_A`), transitions, Pass@k,
invalid output, diversity, reward-group health, token use, cost, and the future
auxiliary exact-task control are secondary reports. Pass@k is reported only
when `k` is no greater than the number of independent draws actually collected;
unsupported larger-k values are omitted rather than extrapolated.
One greedy pass on selected final checkpoints is a post-selection robustness
sensitivity and never participates in selection.

Checkpoint matching uses validation only. After fixed-budget Pilot 0, the common
target freezes as the minimum of the four seed-by-method step-50 targeted-panel
exact-success posterior means, before candidate draws. The cheap monitor
identifies the first apparent crossing; the predecessor, crossing checkpoint,
and successor are re-evaluated on all 512 validation items with fresh
independent seeds and 16 samples/item. All candidates extend to 32 samples/item
if panel-mean sampling SE exceeds 0.0075. Select the real in-band checkpoint
closest to the target, tie to the earlier checkpoint. Never smooth or
interpolate a checkpoint, and mark a method unavailable when it does not reach
the band. Final tests are never used for calibration, panel selection,
checkpoint selection, or method choice.

“Capability matched” in this follow-up means matched on targeted-panel
exact-success within that frozen band; it is not a claim of globally equivalent
behavior. Before Stage B, publish the target mean, sentinel mean, per-family
accuracies, full Cover curve, invalid-output rate, completion-length
distribution, supported Pass@k values, verified strategy-family diversity, and
token surprisal for every selected origin. These are descriptive diagnostics,
not additional matching gates: residual differences may be consequences of the
acquisition procedure rather than nuisance variables to erase.

## Comparability

Comparability is contrast-specific, not a claim of globally equal compute:

| Contrast | Required match or qualification |
|---|---|
| `B-S/B-G` | same M0, panels, prompt allocation, adapter rank, renderer, and cap; compare fixed-budget and targeted-panel-exact-success-matched checkpoints; complete procedures differ in data, objective, exploration, and resource use |
| `B-G/G-U` | same M0 and RL configuration; differs in prompt allocation; optional only |
| `B-O/B-S` | same M0 and matched Stage-A performance; current-policy generation is the mechanism difference |
| `B-G/B-O` | same M0 and matched Stage-A performance; objective comparison is secondary |

All methods use common model, M0, rank, renderer, completion cap, task manifests,
and declared seed namespaces. Exact teacher and rollout accounting is preserved.
The M0-derived Stage-A prompt allocation remains fixed throughout training; no
method-specific boundary rescan is allowed. B-O additionally records accepted-
item/family coverage, effective family count, duplication, zero-acceptance, and
generated/accepted/optimized token totals at every refresh.

## Calibration namespaces and provisional values

Engineering (`5`), pilot (`11,29,47,71`), and confirmatory
(`97,131,197,251,307,401`) seed namespaces are disjoint. Confirmatory seeds are
fresh and the final count is power- and cost-based.

Supervised LR candidates are `1e-4,3e-4,1e-3`; group-relative RL candidates are
`1e-5,3e-5,1e-4`. No LR is selected. Stage-A's provisional checkpoint grid is
`0,10,25,50,100,200,400`; Stage B uses an early-dense provisional grid. Both
durations may change during calibration and then freeze before paired runs.

The boundary scan is adaptive: a four-sample broad screen, targeted refinement
to roughly 16–32 samples, and finalist confirmation on multiple held-out items.
Family eligibility uses equal-item estimates and useful `I8` signal, with a
provisional floor of `0.25`; thresholds freeze only after M0 calibration.

## Tinker execution contract

- Model: `Qwen/Qwen3.5-9B-Base`; adaptation: rank-32 LoRA.
- The archived smoke pinned SDK/Cookbook versions and `role_colon`; the v1 live
  smoke independently verifies the rebuilt runtime before scientific access.
- Project identity comes from `TINKER_PROJECT_ID`.
- Every future remote command is phase-scoped, preflighted, explicitly
  confirmed, bounded by pessimistic client-side token/spend guardrails, and
  reconciled against raw Tinker billing usage and a pinned price snapshot.
- A client-side guardrail is not described as a server-enforced hard dollar
  cap; current public Tinker APIs do not document one.
- Selected checkpoints explicitly disable accidental short TTL and record M0,
  immediate parent, sampler path, full-state path, and optimizer inheritance.
- Every Pilot/confirmatory generation has a concrete request seed. Cross-client
  bitwise identity is tested but not assumed.
- Before behavioral forward-KL is used, identical origin-sampled completion
  tokens are scored under both checkpoints and completion-position alignment is
  checked through a small service acceptance test.
- Confirmatory launch reservations may consume at most 80% of the then-
  uncommitted grant balance.

**Live smoke gate** is an engineering gate, not a numbered research phase.
Its artifact phase label is `live-smoke-gate`, and its total authorized spend
is capped at `$25`. It passes only with real, non-fabricated data, exact
equality between online rewards and offline verifier rewards, a verified stop
contract, and successful full-state resume plus weights-only branching. Any
failure stops before a scientific run.

The grant is funding, not authorization. `duraseed live-smoke` is preflight
only unless `--execute` and an explicit authorized cost are supplied.

## Tasks, data, and reproducibility

TCES and MAPS retain their exact parsers, solvers, generators, and verifiers.
Generation is deterministic from declared seeds. Content-addressed manifests,
training/validation/sealed-test/OOD separation, instance/content/family leakage
checks, exact solver teachers, raw generations and rewards, item-level
M0→Stage-A→Stage-B links, and reproducible analysis remain mandatory.

No PAC-Bayes analysis, representation geometry, Hessian work, natural or RL
Stage B, rank sweep, second model, hierarchical confirmatory model,
supervised-adaptation-plasticity intervention, or broad exact-task refactor
precedes the core B-S/B-G result.
