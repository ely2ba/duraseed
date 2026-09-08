# DuraSeed

**How a model learns a skill can change what happens when it learns the next one.**

Made possible by **$5,000 in compute credits from Thinking Machines Lab (TML)**
through its [Tinker Research Grant](https://thinkingmachines.ai/news/tinker-research-and-teaching-grants/).

**Experiments complete · September 2026.** Two Pilot pairs and a supervised
trace-replay follow-up are finished. Results and analysis are below; the paper
is being revised and will be shared separately.

[The study](#the-study) · [Results](#results) · [Related work](#where-this-fits) ·
[Data and reproduction](#data-and-reproduction)

Most evaluations ask what a model can do now. DuraSeed asks what its current
score leaves out: how much of a newly learned skill survives the next training
stage, and how readily the model learns the next task.

We first compared supervised learning with reinforcement learning. We then
kept supervised learning fixed and changed the source of its worked solutions.
In both comparisons, similar arithmetic scores concealed large differences in
which problems the models could solve and how they responded to later training.
The follow-up also complicated the retention story: broader problem coverage
did not mean slower initial forgetting.

## The study

![DuraSeed: two acquisition comparisons, score-based checkpoint selection, and the same later training](docs/assets/duraseed-pipeline.svg)

The first task is arithmetic expression synthesis: use a given set of numbers
to construct an expression that reaches an exact target. The second is short
program synthesis: write a sequence of operations that implements a modular
arithmetic transformation. Both have deterministic answer checkers; no judge
model scores the outputs.

All branches start from a common format-trained **Qwen3.5-9B-Base** origin and
use rank-32 LoRA adapters. After acquiring arithmetic, selected checkpoints
receive the **same 480 updates of supervised program-synthesis training**, with
the same data order and a fresh optimizer. Arithmetic is not rehearsed during
this later stage.

| Comparison | What changes during arithmetic learning? | Completed scope |
|---|---|---|
| **Pilot 0** | **B-S:** supervised solver solutions. **B-G:** on-policy RL with verifier rewards. | Two matched seed pairs, each followed through all 480 later updates. |
| **Trace replay** | **R-S:** solver solutions. **R-P:** correct solutions archived from Pilot RL. Both use supervised learning on shared prompts and the same prompt order. | Two source blocks, four acquisition runs. One block matched and completed later training; the other had no match. |

We record three views of each selected checkpoint: **F3**, its starting
behavior and transfer to the new task; **F1**, retention of the arithmetic
skill during later training; and **F2**, learning of the new task, both in
absolute terms and relative to its own starting score.
F1 measures total post-acquisition accuracy, including capability already
present at the common origin; it does not isolate only newly acquired ability.

## Results

### Similar scores, different sets of solvable problems

The replay checkpoints were selected at **1,504/4,096** and **1,503/4,096**
correct arithmetic attempts. On fresh draws over those same 256 problems,
their average accuracy was again close: **35.72% for R-S, 34.59% for R-P**.
But R-S solved **170** problems at least once in 16 tries; R-P solved **247**.

![Recorded successes on each of 256 arithmetic problems: R-S covers 170, R-P covers 247 despite similar average accuracy](docs/assets/results-coverage.svg)

R-S answered 35 problems correctly on every attempt and never solved 86.
R-P had no perfect-16 problems, but only nine it never solved. The two averages
therefore describe different distributions of success: repeated reliability on
a narrower set versus occasional success across almost the whole panel.

This difference extended beyond the families used for acquisition. On held-out
arithmetic families, R-P's success rate was **35.84%**, against **2.76%** for
R-S. During acquisition, R-P led on this held-out-family panel at all **30
recorded checkpoints in each source block**. These are validation results,
not an evaluation of the unopened test sets.

### The starting profiles were different too (F3)

The selected replay checkpoints, before any program-synthesis training:

| Starting profile | Solver traces · R-S @220 | Archived policy traces · R-P @20 |
|---|---:|---:|
| Targeted arithmetic success per attempt | 35.72% | 34.59% |
| Targeted problems solved at least once / 256 | 170 | 247 |
| Held-out-family success per attempt | 2.76% | 35.84% |
| Median targeted response length, tokens | 84 | 628 |
| Distinct verified strategy signatures, targeted | 52 | 512 |
| Mean targeted token surprisal, nats/token | 0.061 | 0.178 |
| New-task success before training | 0.00% | 5.19% |

The arithmetic profiles use 16 attempts per item. Strategy signatures are
verifier-derived structural labels, not a count of distinct reasoning methods.
The full [F3 profiles](artifacts/replay-v1/followup/profiles/) also report output
validity, length stops, repetition, and per-family results.

### What survived, and what was learned next? (F1 / F2)

Retention uses a separate monitor panel: 192 targeted items with four attempts
per checkpoint. Its starting scores therefore differ from the 16-draw profile
above.

![Replay arithmetic retention over updates 0–20 and program-synthesis learning over updates 0–480](docs/assets/results-trajectories.svg)

R-P lost half its starting arithmetic score sooner than R-S: **1.69 versus
4.25 updates**. It then rebounded at update 10, from **59 to 241 correct
attempts out of 768** between updates 5 and 10. Its average arithmetic score
over the first 20 updates was consequently higher: **17.62% versus 11.49%**.
First decline and performance over a window give different answers here.

On the new task, R-S had the larger early improvement and the larger average
absolute score over the full training run. At the endpoint, the two branches
were close: **45.79% versus 45.70%**. Both finished with zero correct arithmetic
attempts on the final targeted and held-out-family panels.

| Replay summary · source block 11 | R-S | R-P |
|---|---:|---:|
| Targeted arithmetic half-life, updates | 4.250 | 1.689 |
| Mean targeted arithmetic score, updates 0–20 | 11.49% | 17.62% |
| Mean new-task score, updates 0–40 | 11.02% | 7.07% |
| Mean new-task gain over own baseline, updates 0–40 | 11.02 pp | 1.88 pp |
| Mean new-task score, updates 0–480 | 32.41% | 29.77% |
| New-task endpoint, update 480 | 45.79% | 45.70% |

Here, a window average is the trapezoidal area under the recorded curve divided
by the window's width. Half-life is the first downward crossing of half the
branch's own starting score, interpolated between checkpoints. All scores use
**raw Pass@1: correct attempts divided by all attempts**, including invalid
outputs. “pp” means percentage points.

Paired item-bootstrap 95% intervals put R-P's arithmetic window-average
advantage at **6.13 pp [4.29, 7.94]**, R-S's full-window new-task advantage at
**2.65 pp [1.88, 3.42]**, and the endpoint difference, R-P minus R-S, at
**−0.09 pp [−1.92, 1.76]**. These intervals resample the same items with their
paired trajectories; they do not measure uncertainty across training seeds or
checkpoint selection. [Full counts, curves, and intervals](artifacts/replay-v1/followup/readout.md).

### The two Pilot pairs

Before trace replay, the comparison changed the whole arithmetic-acquisition
procedure: supervised learning (**B-S**) versus RL (**B-G**). The RL branch had
a longer targeted arithmetic half-life in both pairs, a higher new-task
baseline, and a higher new-task endpoint. B-S made the larger early gain from
its own baseline in both pairs.

| Pilot result · entries are B-S / B-G | Pair 1 · seed 11 | Pair 2 · seed 29 |
|---|---:|---:|
| Selected arithmetic checkpoints | 140 / 30 | 40 / 20 |
| Matching score | 31/96 / 31/96 | 17/96 / 17/96 |
| Targeted half-life, updates | 2.664 / 4.105 | 1.136 / 3.343 |
| New-task baseline | 0.00% / 4.99% | 0.00% / 5.35% |
| Mean new-task gain, updates 0–40 | 7.69 / 1.77 pp | 6.97 / 1.88 pp |
| Mean absolute new-task score, updates 0–40 | 7.69% / 6.77% | 6.97% / 7.22% |
| Mean new-task gain, updates 0–480 | 26.23 / 20.18 pp | 12.71 / 18.75 pp |
| New-task endpoint | 37.82% / 40.37% | 26.86% / 36.87% |

The absolute early-performance ordering changes between pairs, as does the
full-window gain ordering. “Larger improvement” and “higher performance” should
not be used interchangeably. The [existing-data audit](artifacts/replay-v1/pilot-audit/README.md)
puts both definitions alongside the original trajectories and failure counts.

Adapter measurements supplied a separate prospective prediction. The LoRA
**B-factor norm** was larger in B-S by **7.36× in Pair 1** and **5.46× in Pair
2**. Before inspecting Pair-2 outcomes, the recorded prediction identified B-S
as having the shorter targeted half-life and the larger early baseline-relative
gain. Both legs held. The [prediction and scoring record](docs/results/pilot0-pair2-prediction.md)
preserves the timing: this was before outcome inspection, not before execution.
It is an association with saved parameters, not evidence that their scale
caused the trajectories.

## What the study establishes—and what it does not

In these experiments, a matched arithmetic score did not make two checkpoints
interchangeable as starting points for further training. Trace replay retained
large differences in problem coverage, held-out-family performance, and
new-task transfer even when both acquisition branches used supervised
learning. Its later trajectories do **not** support a blanket claim that
policy-derived traces slow forgetting: R-P crossed its half-score threshold
first, then rebounded.

The scope is one model, one adapter rank, two Pilot pairs, and one matched replay
continuation. Pilot compares complete procedures, not just loss functions.
Replay changes a bundle of trace properties—content, length, format, and
strategy—and training-token doses are not matched. Matching a scalar score is
not an equivalence test for generalization or internal state.

The original replay matching design unnecessarily required agreement with
historical Pilot-0 scores in addition to agreement between the new arms. We
removed that historical-score requirement after seeing the candidate scores,
before any replay Stage-B outcomes. Block 11 then matched; block 29 still did
not and received no Stage-B training. Both decisions are retained in the
[matching record](artifacts/replay-v1/followup/selection.json).

## Where this fits

[SFT Memorizes, RL Generalizes](https://proceedings.mlr.press/v267/chu25c.html)
studies generalization after acquisition.
[RL's Razor](https://arxiv.org/abs/2509.04259) and
[Retaining by Doing](https://arxiv.org/abs/2510.18874) study preservation of
pre-existing abilities while learning through SFT or RL.
[Good SFT Optimizes for SFT, Better SFT Prepares for Reinforcement Learning](https://arxiv.org/abs/2602.01058)
shows that a checkpoint's present performance need not predict its performance
after a common later training recipe.

DuraSeed follows a newly acquired skill into that later stage. We measure its
retention, learning of the next task, and the starting behavioral profile in
the same comparison. Later training is supervised for every branch; the
contrast concerns what was inherited from acquisition. The replay study then
asks how much of the pattern remains when both branches acquire the skill
through SFT. This is a controlled case study of training history, not a claim
to have discovered sequential forgetting or training on model-generated data.

## Data and reproduction

| Material | What is available |
|---|---|
| [Pilot-0 raw data](https://github.com/ely2ba/duraseed/releases/tag/pilot0-data-v1) | 519,424 recorded completions, verifier rewards, prompts, token records, evaluations, and matching selections. [Portable data guide](docs/pilot0-data.md). |
| [Pilot audit and notebook](artifacts/replay-v1/pilot-audit/) | Raw and baseline-relative trajectories, paired uncertainty, item counts, and failure breakdowns for both pairs. |
| [Pair 1](artifacts/pilot0-pair1-readout/README.md) / [Pair 2](artifacts/pilot0-pair2-readout/README.md) | Original F1/F2 readouts; starting profiles and adapter geometry in the [Pair-1](artifacts/pilot0-pair1-offline-analysis/README.md) and [Pair-2](artifacts/pilot0-pair2-offline-analysis/README.md) analysis packages. |
| [Completed replay package](artifacts/replay-v1/followup/README.md) | Both acquisition histories, matching decisions, per-item outcome counts, F3 profiles, retention and learning curves, corpus lineage, and descriptive breakdowns. |
| [Methods and technical detail](docs/TECHNICAL.md) | Protocols, metric definitions, implementation, and the short procedural history. |

The compact replay release supports count-based analyses; it does not include
raw generation text or adapter tensors. Private account and billing metadata,
personal filesystem paths, and unopened test sets are excluded. The paper is
still being revised and is **not part of this release**. No further experiments
are running or planned for this study.

Regenerate the two result graphics from the checked-in counts, with Python's
standard library only; this makes no service calls:

```sh
python tools/make_readme_figures.py
```

## Acknowledgements

This project was made possible by a
[Tinker Research Grant from Thinking Machines Lab](https://thinkingmachines.ai/news/tinker-research-and-teaching-grants/),
which provided the compute credits used to run the experiments.

## AI-assisted development

I developed DuraSeed as an independent research project, from framing the core
question and hypotheses through experimental design, implementation decisions,
debugging, analysis, and interpretation. I made the scientific and engineering
decisions throughout, including how to structure the comparisons, what to
measure, how to respond to failures, and what conclusions the evidence could
support.

I used OpenAI Codex as a coding partner, particularly for
implementation, boilerplate, scaffolding, tests, repetitive analysis, and
debugging. That leverage made it possible to execute a project of this scope
independently in weeks rather than months.

Coding agents also tend to produce more machinery than an experiment needs, so
part of the work was actively reviewing, simplifying, and redirecting the
implementation to keep the software serving the science rather than the other
way around.

Licensed under the [Apache License 2.0](LICENSE).
