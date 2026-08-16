# DuraSeed

A model can reach the same score in more than one way. That does not necessarily
mean the different training methods leave it in the same condition.

One method might produce a skill that survives later training better. Another
might leave the model more ready to learn what comes next. If we only look at
the score immediately after training, we cannot see either difference.

DuraSeed asks a simple question:

> When two post-training methods teach the same model the same new skill to a
> comparable level, do they differ in how well that skill lasts and how easily
> the model learns the next skill?

This matters because useful models are rarely trained once and left alone.
They are fine-tuned, updated, and adapted again. A method that looks best on
today's benchmark may be a worse foundation for tomorrow's training if it
causes more forgetting or makes the next task harder to learn.

The study is deliberately narrow: one base model, rank-32 LoRA adapters, and
two synthetic task families with exact answers. This lets us measure the
trade-off without relying on a judge model or debating whether an answer was
mostly right. The goal is not to declare one training method universally
better. It is to find out whether the path used to acquire a skill can change
what happens when the model is trained again.

## The first comparison

Every run begins from the same format-capable `Qwen/Qwen3.5-9B-Base`
checkpoint and receives the same initial boundary-teacher training. The first
comparison then splits into two paths:

- **B-S** continues with supervised fine-tuning on a fixed set of correct,
  solver-generated answers.
- **B-G** learns from its own sampled attempts, scored by an exact verifier,
  using group-relative reinforcement learning.

In plain terms, this compares learning from a fixed book of worked solutions
with learning by trying solutions and receiving exact feedback. It is a
comparison between two complete acquisition procedures, not a claim that only
their loss functions differ. The procedures also differ in where their answers
come from, which answers the model sees, and how much generation and training
work they require. If they produce different later behavior, the first result
will therefore be about those complete procedures—not a universal claim that
"RL representations" are better or worse than "SFT representations."

Stage A uses arithmetic-expression problems. The model must combine a given
set of numbers, use each exactly once, and reach a target value. We look for
strategy families near the model's current ability boundary: hard enough to
leave room for learning, but not so hard that every answer is wrong.

Those families are divided into two closely matched 12-family panels. One is
trained while the other is held out as a sentinel, and their roles reverse
across paired seeds. This helps distinguish retention of the taught skill from
broader changes that affect everything.

After Stage A, every checkpoint receives the same Stage-B supervised training
on a separate modular program-synthesis task. We measure two things together:

1. how much of the Stage-A skill remains as Stage B progresses; and
2. how quickly the model learns Stage B under the same training recipe.

Recent work asks several nearby questions. [SFT Memorizes, RL
Generalizes](https://arxiv.org/abs/2501.17161) compares how behavior acquired
through SFT and RL generalizes to unseen variants. [RL's
Razor](https://arxiv.org/abs/2509.04259) and [Retaining by
Doing](https://arxiv.org/abs/2510.18874) study how well a model's pre-existing
abilities survive while it learns a new task, while [continual post-training
work](https://arxiv.org/abs/2507.05386) follows learned tasks through a longer
sequence. Other results show that a checkpoint's current score need not predict
how well it responds to [the training that comes
next](https://arxiv.org/abs/2602.01058). To our knowledge, the less-studied
combination is the one DuraSeed targets: checkpoints produced by two complete
acquisition procedures, compared at similar Stage-A capability, then exposed
to exactly the same, separate Stage-B training while we measure both retention
of Stage A and learning of Stage B. This is the distinction—not a claim that
sequential forgetting or model plasticity has gone unstudied.

Pilot 0 first compares B-S and B-G after the same fixed Stage-A duration. A
required follow-up then selects real checkpoints at comparable Stage-A
capability and repeats the common Stage-B probe. More precisely, those
checkpoints are matched on targeted-panel exact-success. We do not pretend that
one number makes their behavior identical. Before Stage B, we will also show
their target, sentinel, and family-by-family accuracy; how broadly success is
spread rather than concentrated in a few families; invalid-output rate;
completion length; Pass@k at supported sample counts; verified strategy
diversity; and sampled-token surprisal. These measurements reveal what remains
different without matching away differences that may be part of the phenomenon.

We will also show how Stage-A capability changes with cumulative acquisition
tokens and cost. Prompt/prefill, sampled, and training tokens remain separate
in the resource accounting: one dollar total is useful, but it should not hide
the very different work performed by the two procedures.

## Why the design looks this way

- **Exact tasks keep the measurement honest.** The generators, solvers, and
  verifiers are deterministic, so correctness does not depend on subjective
  grading.
- **One common origin keeps the comparison interpretable.** The model, adapter
  rank, initial checkpoint, panels, and Stage-B recipe are shared across paths.
- **Crossed panels control for family quirks.** Target and sentinel roles swap
  across paired seeds instead of relying on one lucky task split.
- **The separation rule matters more than the number 12.** Two matched
  12-family panels remain the preferred design. But 12 is not worth preserving
  by weakening data separation or looking at method results. If that size is
  genuinely infeasible, any different equal panel size must be chosen
  prospectively from feasibility and power alone, before Stage A begins.
- **Final tests stay sealed.** Training, calibration, checkpoint selection, and
  method decisions use separate data.
- **Independent runs come before a scale sweep.** For the first study, learning
  the seed-to-seed variance is more valuable than spreading the budget across
  several model sizes or adapter ranks. If the core effect holds up, a
  same-model higher-rank or full-weight replication is the most direct way to
  test whether it depends on the constrained LoRA update space, if the platform
  and budget allow it. A larger-model replication comes later.
- **Method breadth is earned by the result.** B-S versus B-G comes first. B-O
  and the broader controls remain available, but they are added only if the
  first results show that they answer a useful mechanism question. If B-S and
  B-G separate reproducibly, the first mechanism test will be a supervised
  replay of a frozen, verifier-correct subset of B-G-generated answers. That
  attacks the data-versus-objective ambiguity more directly than immediately
  adding a wide collection of methods.

Here, "future learnability" has a specific meaning: learning under this fixed
supervised Stage-B probe. It is not presented as a universal measure of model
plasticity.

## Where things stand

Most of the groundwork is finished. The project has been rebuilt in a clean
public repository, the original tasks, verifiers, and analysis replay with zero
differences, and the code paths for the remaining pre-pilot gates pass their
local tests. Earlier calibration established the common M0 starting point,
selected the canonical training format, froze the MAPS recipe at
`shortest2_cap2` with learning rate `3e-4` through step 480, and identified 15
eligible families in the first boundary cohort.

The live engineering smoke passed on real remote execution. It exercised
training and sampling end to end, reproduced verifier rewards exactly online
and offline, confirmed the stop contract, and tested both full-state resume and
weights-only branching. This is an engineering result, not evidence for the
research hypothesis.

Remote data collection for the boundary search is complete. Cohorts 2 and 3
both finished broad sampling, refinement, and confirmation. The final
three-cohort replay also completed and reproduced the old and new reducers
exactly.

That replay found 49 observation-eligible families, comfortably above the 36
needed to form two panels plus an intermediate pool. The matching step selected
24 families for the two panels. But the final, stricter data-separation check
found that only 12 of those 24 could supply all 22 required single-family test
items; the other 12 could supply none. Because the core design requires two
complete 12-family panels, no panel assignment was published.

This is a genuine feasibility result for the current panel-construction rule,
not a software failure and not a result about B-S versus B-G. Acquisition
calibration and Pilot 0 remain stopped. A subsequent outcome-blind diagnostic
found that 31 of the 49 families can each supply all 22 test items even when all
49 candidates are treated as reserved. This shows that at least 24
capacity-qualified families exist and supports a simple prospective cure:
collapse exact algebraic duplicates and apply the unchanged test-capacity rule
before matching. That narrow amendment is now accepted: the 49 families form
37 exact algebraic classes. Panel families must pass the unchanged 22/22 gate;
the 12 intermediate families retain their original, separate five-split
eligibility rule and are selected outside both panels. The amended freeze has
now completed: all 37 representatives passed the unchanged capacity audit, two
disjoint 12-family panels and 12 separate intermediate families were frozen,
and the archived and current production split builders produced byte-identical
train and gate manifests. Acquisition calibration is the next experimental
stage; it has not started yet.

If a valid panel construction is frozen, the remaining path is:

1. calibrate the shared teacher dose, Stage-A learning rates and duration, and
   common RL setup;
2. run the two-seed, fixed-budget B-S/B-G Pilot 0;
3. run the required targeted-exact-success-matched checkpoint follow-up and
   publish the complete pre-Stage-B profile;
4. extend to three or four paired pilot seeds to estimate variance;
5. if a reproducible effect appears, test supervised replay of frozen,
   verifier-correct B-G rollouts before expanding broadly across methods;
6. decide whether B-O or any other control adds enough scientific value;
7. freeze and preregister the confirmatory design; and
8. run it on fresh seeds, reserving higher-rank or full-weight replication for
   an effect worth following up.

Pilot 0 has not started, no sealed test data has been opened, and no result is
claimed for the stability–future-learnability question. Everything completed so
far is calibration, feasibility evidence, and experiment preparation.

## Repository layout

Start with [`docs/MAP.md`](docs/MAP.md) for the component map, scientific data flow, and provenance locations. Current gate status is tracked in [`docs/STATUS.md`](docs/STATUS.md).

- `src/duraseed/tasks/`: TCES and MAPS generators, parsers, solvers, and exact verifiers.
- `src/duraseed/data/`: split, manifest, leakage, boundary, panel, and matching logic.
- `src/duraseed/training/`: verified-data builders and calibration decision reducers.
- `src/duraseed/evaluation/`: item-level transition and frontier analysis.
- `src/duraseed/runtime/`: bounded Tinker client, sampling, checkpoint, ledger, and billing code.
- `src/duraseed/runners/`: orchestration for the next authorized experimental gates.
- `frozen/v0/` and `provenance/`: preserved calibration evidence, hashes, and replay records.
- `PROTOCOL.md`: public scientific contract and calibration record.

## Local checks

Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
python tools/check_size.py
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv lock --check
uv run duraseed validate
```

These checks are local and credential-free. Remote Tinker access requires a separate, explicit launch authorization and spend cap.

## Research scope

The design compares acquisition paths under one base model, one adapter rank, synthetic exact tasks, and one frozen supervised Stage-B probe. Any eventual inference is conditional on those choices. Calibration outcomes and engineering checks are not results for the stability–plasticity hypothesis.

## License

[Apache License 2.0](LICENSE)
