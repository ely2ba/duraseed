# DuraSeed

**Can models that score similarly today learn — and forget — differently tomorrow?**

DuraSeed teaches the same starting model an arithmetic skill in two ways,
selects checkpoints with similar measured scores, then gives both identical
training on a new task. We follow what happens to the old skill and how quickly
the new one develops.

**30 August 2026:** the first paired pilot is complete; the second is running.
The results below are from **one paired seed**, not a general verdict on
supervised learning versus reinforcement learning.

## The idea

A benchmark score is a snapshot. Models are trained repeatedly, so we also
care about what a checkpoint is ready for: will its skills survive later
training, and can it learn something else?

Nearby work asks how much pre-existing ability is lost *while* a skill is
learned by supervised fine-tuning (SFT) or reinforcement learning (RL).
DuraSeed follows the acquired skill into the *next* training stage, holding
that later training identical.

![DuraSeed: common origin → two learning procedures → capability match → identical later training → retention and new learning](docs/assets/duraseed-pipeline.svg)

## How the experiment works

**Learn an arithmetic skill — Stage A.** Both branches start from the same
format-capable Qwen3.5-9B-Base checkpoint, called M0, with fresh optimizers.
Training changes a rank-32 LoRA adapter: a small set of trainable weights
attached to the model. M0 has prior output-format training, but no task-specific
arithmetic warm start.

The task is to reach a target value using each supplied number once.
An exact verifier checks the answer; there is no judge model.

- **B-S (SFT)** learns from fixed, verified solver derivations.
- **B-G (RL)** generates attempts and learns from verifier rewards through
  group-relative reinforcement learning.

These are two complete training procedures. Their examples, exploration,
learning rates, and training budgets differ — this does not isolate the loss
function alone.

**Match checkpoints.** Every ten updates, we save and evaluate each branch.
After Stage A, a frozen rule selects real checkpoints with the closest
targeted scores whose uncertainty intervals overlap. Matching is unavailable
if no pair qualifies; there is no seed replacement. It matches one measured
capability, not the models' entire behavior.

**Train on something new — Stage B.** The selected checkpoints keep their
learned adapters, reset their optimizers, and receive the same 480-update
supervised program-synthesis training. We measure:

- **F1 — retention:** how arithmetic performance changes during later training.
- **F2 — future learning:** performance on the new task, both as an absolute
  score and as improvement from each model's own starting point.
- **F3 — starting profile:** before Stage B, a fuller picture of accuracy,
  held-out families, output length and validity, solution diversity, and
  initial performance on the new task.

## First pilot results

Pair 1 uses seed `11`. The matching rule selected **B-S at update 140** and
**B-G at update 30**: both scored **31/96** on the matching panel.
On a separate, higher-draw targeted evaluation, their scores were **33.62%**
and **30.98%**. Similar matching scores are not proof of equal capability.

The figures below show observed single-attempt success rates (*Pass@1*) and
changes from baseline, using the saved evaluations.
The [detailed results](docs/results/pilot0-pair1.md) include counts, uncertainty
intervals, both starting profiles, and the exploratory follow-up analyses.
Monitoring and higher-draw validation use separate item panels, so their
starting percentages need not match.

### The old skill: the difference is early, not at the endpoint

![Arithmetic success during the first 20 later-training updates, for targeted and held-out families](docs/assets/pilot-pair1-retention.svg)

On the targeted monitor, B-S starts at **29.82%** and B-G at **30.86%**.
After two updates on the new task, they score **16.93%** and **27.21%**.
By update ten, both are below 1%: **0.52%** and **0.78%**.

At the final update-480 evaluation, each branch records **0/4,096 successes**
on targeted arithmetic items and another **0/4,096** on held-out families.
That is zero observed success under this evaluation, not proof that every
trace of the ability has disappeared.

### The new skill: final score and improvement tell different stories

![New-task learning over 480 updates, shown as absolute success and improvement from each branch's own baseline](docs/assets/pilot-pair1-learning.svg)

B-G can already solve **4.99%** of new-task attempts before Stage B; B-S
records **0%**. After 480 identical training updates, they reach **40.37%**
and **37.82%**, respectively. B-G's endpoint is **2.55 percentage points
higher**; the paired item-bootstrap 95% interval is **0.77 to 4.35 points**.

The ordering reverses for the frozen summary of **improvement over the whole
training run**. This averages the learning curve above each branch's own
baseline, rather than comparing just its endpoint. On the study's smoothed
score, that average gain is **0.2469 for B-S** and **0.1899 for B-G**.
The exact definition and interval are in the [results note](docs/results/pilot0-pair1.md#f2--subsequent-maps-learning).

These intervals resample shared evaluation items. They do **not** tell us how
much the result will vary across independently trained seed pairs.

### A similar target score leaves large differences elsewhere

Before Stage B, the higher-draw profiles show:

- **Held-out arithmetic success:** 4.17% for B-S, 31.64% for B-G.
- **Targeted response length:** a median of 78 tokens for B-S, 1,464 for B-G.
  The 4,096-token cap is reached in 1.98% and 30.32% of attempts.
- **Verified solution variety:** 40 versus 542 distinct strategy families
  among successful targeted completions, from 4,096 attempts per branch.
  These count verifier-defined expression structures, not inferred reasoning
  processes.

The adapter weights also differ: the aggregate size of LoRA's **B factor**
is **7.36× larger in B-S**. This is a description of the saved parameters,
not evidence that parameter scale caused the learning or retention curves.

A separate check of B-G's new-task curve found **no cap or format failures
at updates 40 and 80**, while success rose from **7.23% to 17.18%**.
That rise was accompanied by fewer wrong-target answers, not the resolution
of recorded cap or format errors.

## What comes next

**Pair 2 (seed `29`) is running under the same frozen recipe.** The two
12-family panels exchange trained-target and held-out roles across the pairs.
Its authorization did not depend on the direction or size of pair-1 results;
the commitment is recorded in [STATUS.md](docs/STATUS.md).

The two pairs will assess feasibility and effect direction, with a first,
crude look at seed-to-seed variability. More fresh paired seeds are needed
before a powered confirmatory study; neither these item-level intervals nor
two pairs replace that step.

One candidate follow-up asks whether **adapter scale changes the effective
speed of later learning**, even with the same nominal learning rate.
We can test that by rescaling LoRA's two factors in opposite directions:
the model's starting function stays fixed, while its parameterization changes.
A prediction linking larger B-factor scale to faster early forgetting and
learning was recorded before inspecting pair-2 outcomes. This remains a
non-binding hypothesis; no such intervention has been run or authorized.

## Scope and background

This is one 9B model, rank-32 adapters, synthetic tasks with exact verifiers,
and one later-training recipe. The first result is one paired seed, with a
narrow, noisy capability match and substantial differences elsewhere. It
cannot establish that SFT or RL is generally better, or identify a single
cause of the observed trajectories. Final test sets remain sealed during Pilot.

Getting here involved simplifying the task families, retiring an unstable
shared task-specific warm start, and aligning the response-format measurement
with the prompt and verifier. Those pre-Pilot corrections led to the current
common-origin comparison. The calibration history and implementation details
live in the [technical appendix](docs/TECHNICAL.md), not in the result claims.

## Reproduce and inspect

Local setup and validation are credential-free:

```bash
uv sync --extra dev
uv run pytest
uv run duraseed validate
```

- [Pair-1 results](docs/results/pilot0-pair1.md) — measured curves, profiles,
  uncertainty, and descriptive diagnostics
- [`PROTOCOL.md`](PROTOCOL.md) — scientific design and outcome definitions
- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) — implementation, calibration, and
  process detail
- [`docs/STATUS.md`](docs/STATUS.md) — current experimental state

## Development and support

DuraSeed is an independent, agent-assisted research project and a learning
experience. Codex handles much of the coding; the scope and implementation
have been deliberately narrowed to keep the repository compact and the
science legible. Constructive input on the design, analysis, implementation,
or presentation is welcome through
[GitHub Issues](https://github.com/ely2ba/duraseed/issues).

This work was made possible by a generous **$5,000 research grant from
[Thinking Machines Lab](https://thinkingmachines.ai/)**.

Licensed under the [Apache License 2.0](LICENSE).
