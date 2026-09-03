# Confirmatory design notes — non-binding

_2026-08-30. Post-hoc notes from Pilot pair 1, seed 11._

**No design authority is exercised here.** These are candidates and questions,
not decisions, amendments, preregistration, launch authorization, or new gates.
Nothing in this document changes the frozen protocol, charter, matching rule,
measurement schedule, or running pair-2 job. The analysis used pair-1 local
artifacts only. Any future design would require a separate prospective decision
before its confirmatory outcomes are observed.

Source: [offline analysis package](../artifacts/pilot0-pair1-offline-analysis/README.md),
with paired item-bootstrap intervals, early-window statistics, recorded failure
codes, and archived adapter geometry. Intervals are conditional on one fitted
seed pair and its selected checkpoints; they are not seed-to-seed uncertainty.

## Early-window retention estimand candidates

Candidate summaries include absolute targeted Pass@1 AUC over a prospectively
fixed early window, own-baseline-relative AUC over that window, and time to a
fixed fraction of initial retention. The package reports 0–20-update AUC and
interpolated half-life because they were requested after pair 1 completed;
those windows and summaries are therefore exploratory, not confirmatory.

For context, targeted raw-rate half-lives in pair 1 are 2.664286 updates for
B-S and 4.104651 for B-G, interpolated within the observed [2,5] bracket.
Relative raw-rate AUC(0–20) is 0.183079 and 0.240612, respectively. An eventual
estimand would need its window, interpolation, non-crossing rule, denominator,
and seed-level aggregation fixed in advance. It should distinguish retention
of total pre-B performance from survival of specifically newly acquired ability.

Absolute and relative summaries answer different questions when starting
scores differ. Relative scores become unstable near zero. The four-draw
Jeffreys floor also matters: B-S sentinel posterior half-life cannot cross half
its initial score because that threshold lies below 0.1. Raw rates and posterior
scores must remain explicitly labelled, with any future floor treatment fixed
prospectively rather than chosen from these outcomes.

## Equivalence-margin matching on a high-draw statistic

A candidate future rule could require a confidence interval for a paired,
high-draw targeted capability difference to lie inside a scientifically chosen
equivalence margin. The margin would need external justification before new
outcomes; overlapping confidence intervals or a non-significant difference are
not an equivalence test.

Pair 1 matched at 31/96 for both checkpoints on the cadence panel. On the
separate 16-draw targeted validation panel, B-S minus B-G Pass@1 is 0.026367,
with an item-bootstrap 95% CI of [-0.003906, 0.057129]. This is a conditional
description of residual mismatch, not evidence for any particular margin.
Any future matching plan would need independent confirmation or explicit
selection-aware uncertainty, candidate-search and tie rules, draw budgets,
and a no-match outcome frozen in advance. These notes choose none of them.

## F2 estimand sign-dependence

For pair 1, B-S minus B-G update-480 Pass@1 is -0.025513; the corresponding
posterior-score difference is -0.024012. The stored posterior raw-gain AUC
difference is +0.056936, and the absolute posterior AUC difference is +0.009947.
Endpoint, integrated absolute performance, and improvement from an arm's own
baseline are distinct estimands; their signs need not agree.

A future confirmatory plan should choose its primary question and estimand
before observing its outcomes, report the other frozen summaries alongside it,
and avoid choosing a preferred headline from the observed signs. Baseline
concentration can be reported using a predeclared item-set definition. Selecting
items by observed baseline success also induces selection/regression-to-mean
effects; same-item trajectories alone do not remove that issue.

## Replay-experiment motivation

A candidate supervised-replay comparison could use a prospectively fixed,
verifier-correct subset of B-G rollouts as SFT data from the same origin. This
could help separate consequences of the training objective from consequences
of the trajectory distribution (length, strategy use, content, and formatting).
Replay is not an intervention on just one factor unless the data-selection,
exposure, token-budget, optimizer, and checkpoint-selection rules are specified.
No replay run, recipe, selection rule, or spend is authorized here.

The F2 failure-code check does not support cap/tag repair as a description of
the B-G 40–80 transition in this run: those counts are zero at both endpoints;
successes rise from 592 to 1,407 out of 8,192 while wrong-target counts change
from 7,600 to 6,785. This does not identify a causal learning mechanism.
Descriptive LoRA geometry likewise cannot establish that an effective-rank or
norm difference causes retention or transfer. Factor norms depend on LoRA
parameterization; blockwise BA spectra describe the saved adapters, not a
network-level causal quantity.

## prospective mechanical prediction recorded before pair-2 outcome inspection.

_Added 2026-08-30 at 21:11:34 UTC, before reading any pair-2 outcome; non-binding._

prospective mechanical prediction recorded before pair-2 outcome inspection.
Inherited LoRA factor
scale may act as an effective step-size multiplier under the fresh
Stage-B Adam optimizer (function-space movement via the A-pathway
scales with ‖B‖, ~7x larger for B-S@140 than B-G@30). If so, it
predicts jointly: faster F1 decay AND faster early F2 gain for the
larger-scale arm, in any pair. It predicts nothing about pre-B
transfer, strategy diversity, generalization gap, or F2 endpoint.
A merge-to-base + identically-initialized fresh Stage-B adapter
comparison would test it. Descriptive observation for the record:
both arms' dominant adapter singular direction lies in the unembed
module; B-S's is near-rank-1 with ~11x B-G's magnitude. No design
authority exercised.

## Candidate gauge-rescaling follow-up — non-binding

_Added 2026-08-30 at 21:34:34 UTC; no authorization._

A candidate follow-up would compare functionally identical LoRA
parameterizations `(A, B)`, `(2A, B/2)`, and `(A/2, 2B)`, plus a version
whose B norm is matched to the other arm with compensating inverse scaling
of A so that BA, and hence the starting function, remains fixed. Each version
would receive identical Stage-B data and the same recipe, with a fresh optimizer
and the same seed. The question is whether F1 and F2 vary with parameterization
at an exactly fixed starting function.

Adam epsilon and any weight decay break exact gauge invariance and are part
of the mechanism being tested. A refinement would distinguish unembed-only
equalization from equalization across all adapted modules. The B-scale asymmetry
may partly be driven by the 10× Stage-A learning-rate difference between the
two frozen procedures; the gauge test does not depend on the cause of that
asymmetry.

This is a non-binding candidate only: no design or launch authority is exercised,
no experiment or spend is authorized, and no running job, gate, measurement,
or computation is changed.

## Prospective Pair-2 geometry-prediction scoring rule

_Added 2026-09-03 at 09:22:40 UTC, after the geometry-first prediction was
recorded and before opening any Pair-2 F1/F2/F3 outcome._

Pair 2: seed 29, selected checkpoints B-S@40 and B-G@20, with ‖B‖F ratio
5.46.

The geometry-based prediction is:

1. Faster forgetting under B-S: B-S has the shorter targeted raw-Pass@1
   retention half-life, defined as the first downward crossing of 50% of that
   arm's own update-0 Pass@1, linearly interpolated between registered
   checkpoints. If one arm never crosses, it is treated as longer than an arm
   that does cross. If neither crosses, this leg is inconclusive.
2. Faster early Stage-B learning under B-S: B-S has the larger
   baseline-relative raw-Pass@1 gain AUC over the fixed grid
   `[0, 1, 2, 5, 10, 20, 40]`, trapezoidally integrated and divided by 40,
   where `gain(t) = Pass@1(t) − Pass@1(0)`.

Scoring:

- both predictions hold = CONFIRMED
- exactly one holds = PARTIAL MISS
- neither holds = FAILED
- an inconclusive F1 leg does not count as holding

Report but do not use for scoring:

- targeted relative retention at update 2;
- raw Pass@1 gain at update 40;
- stored 0–480 raw-gain AUC;
- F2 endpoint;
- F3 profiles;
- geometry and other diagnostics.

Also report the descriptive observation that Pair 2's ‖B‖F ratio, 5.46×, is
smaller than Pair 1's 7.36×, but do not score or interpret a
magnitude/dose-response relationship from only two pairs.

## Pair-2 prospective prediction outcome

_Recorded 2026-09-03 at 10:07:08 UTC, after the human confirmed
`PREDICTION RECORDED` and released the outcome read order._

**PREDICTION: CONFIRMED**

| Registered leg | B-S | B-G | Result |
|---|---:|---:|:---:|
| Targeted raw-Pass@1 retention half-life, updates | 1.136364 | 3.343284 | held |
| Baseline-relative raw-Pass@1 gain AUC, updates 0–40 | 0.069696045 | 0.018778992 | held |

The complete frozen readout is in
[`artifacts/pilot0-pair2-readout/outcome-report.md`](../artifacts/pilot0-pair2-readout/outcome-report.md).
No post-outcome metric, window, normalization, or scoring rule was added.
