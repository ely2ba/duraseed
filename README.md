# DuraSeed

**Does the way a model learns a skill change how well that skill survives later
training—and how well the model learns what comes next?**

DuraSeed is a controlled study of that question. We teach the same model the
same new capability through two different post-training procedures, align the
resulting checkpoints on acquired capability, and then give both exactly the
same downstream training. We measure both what they retain and how they learn.

The project is supported by a generous **$5,000 research grant from [Thinking
Machines Lab](https://thinkingmachines.ai/)**.

## The idea

Two checkpoints can earn the same score while being different in ways that
only become visible later. One may hold onto the new skill more reliably.
Another may be a better starting point for the next task. Immediate benchmark
accuracy cannot distinguish those possibilities.

DuraSeed follows the checkpoints forward:

`common origin → two acquisition paths → capability matching → identical downstream training → retention + learning`

Nearby studies compare how much pre-existing ability is lost while a skill is
being learned by SFT versus RL; DuraSeed holds the subsequent training
identical and asks whether the newly acquired skill's durability—and the
model's readiness for the next task—depends on how it was acquired.

The aim is deliberately modest. This is not a universal verdict on supervised
fine-tuning or reinforcement learning. It asks whether two concrete acquisition
procedures leave measurably different downstream consequences under one tightly
controlled setup.

## The experiment

### A literal common starting point

Both methods begin from the same `Qwen/Qwen3.5-9B-Base` rank-32 LoRA checkpoint,
called **M0**, with fresh optimizers. M0 has received a small task-agnostic
format intervention, but no demonstrations from the Stage-A task. It is
therefore task-cold rather than an untouched pretrained model.

### Stage A: acquire the same capability in two ways

Stage A uses exact arithmetic-expression problems. A model must combine a
given set of numbers, use each exactly once, and reach a target value. A
deterministic verifier scores every attempt, so there is no judge model and no
subjective partial credit.

- **B-S** learns from a fixed corpus of correct solver-generated derivations
  using supervised fine-tuning.
- **B-G** samples its own attempts and learns from exact verifier rewards using
  group-relative reinforcement learning.

These are complete procedures. They differ not only in objective, but also in
where their training answers come from, how exploration happens, and what
resources they consume.

The task families are split into two matched 12-family panels. Across paired
seeds, the trained and held-out sentinel roles are crossed. This makes it
harder for one convenient family split to drive the result.

### Match real checkpoints, then inspect what remains different

Each method follows its frozen Stage-A schedule, with evaluation and
checkpoints every ten updates. Afterward, we select the nearest real B-S/B-G
checkpoint pair whose targeted-capability intervals overlap. We do not invent
interpolated checkpoints or replace a seed when no overlap exists; that outcome
is reported as unavailable and Stage B does not run for that pair.

Matching one score does not make two models behaviorally identical. Before
Stage B, we therefore record a broader profile: target and sentinel accuracy,
family coverage, invalid and length-stopped outputs, completion length,
Pass@k, verified strategy diversity, token surprisal, baseline performance on
the Stage-B task before any Stage-B training, and separate token and cost
accounting.

### Stage B: the same downstream learning probe

Both selected Stage-A checkpoints then receive the same supervised training on
a separate modular program-synthesis task. Stage B continues the learned LoRA
weights with a fresh optimizer.

The core outcomes are:

- **F1 — retention:** how much of the Stage-A capability remains during and
  after Stage B;
- **F2 — future learning:** how quickly and how well each checkpoint learns
  Stage B; and
- **F3 — pre-Stage-B profile:** what behavioral differences remained even
  after capability matching.

## Where the project is now

The tasks, exact verifiers, common origin, crossed panels, Stage-B recipe, and
analysis rules are frozen. A bounded capability-dose run confirmed that the
overlap region needed for the paired comparison is reachable and validated the
checkpoint save-and-restore path.

**Pilot 0 has now begun.** The first paired seed (`11`) is running. It trains
B-S and B-G from M0, performs the frozen post-hoc match, and—only if matching is
available—runs the common Stage-B probe. The second paired seed remains
unlaunched until the first pair produces durable evidence and receives a
separate authorization.

No B-S/B-G scientific result is claimed yet, and final test data remain sealed.

The current plan is:

1. complete and inspect the first paired Pilot run;
2. run the separately authorized second pair once the first yields durable,
   complete artifacts—an authorization that does not depend on the direction
   or size of any scientific result ([`docs/STATUS.md`](docs/STATUS.md) records
   this commitment);
3. use the two pairs to assess feasibility and effect direction, with a first,
   crude look at seed-to-seed variability;
4. add fresh paired seeds for variance reconnaissance before fixing a powered
   confirmatory design; and
5. preregister and run confirmation only if the effect is worth pursuing.

## How the design got here

Pre-Pilot work made the study simpler and more literal.

An outcome-blind algebraic deduplication removed equivalent task families
before panel matching. A proposed task-specific teacher warm start was then
retired after bounded calibration showed it was not stable enough to serve as a
common origin. Both methods now branch directly from M0. A later audit found
that one format diagnostic demanded a stricter response shape than the task
prompt and verifier; the contract was corrected before Pilot outcomes existed,
while the original evidence was retained. Finally, a bounded dose run verified
that the supervised path could reach the planned matching region without
opening another tuning ladder.

These were feasibility and measurement corrections, not B-S/B-G results. They
were made before Pilot comparisons or sealed-test access, and the detailed
records remain public in the repository.

For the full protocol, calibration history, frozen gates, data construction,
budget accounting, failure rules, and provenance map, see the **[technical
appendix](docs/TECHNICAL.md)**.

## Scope

The first study uses one 9B model, rank-32 LoRA adapters, synthetic exact tasks,
and one fixed downstream probe. Any conclusion is conditional on that setup.
The design can show a difference between the complete B-S and B-G procedures;
it cannot by itself attribute that difference solely to the loss function or
claim that one family of methods is universally superior.

## Repository

- [`PROTOCOL.md`](PROTOCOL.md) is the public scientific contract.
- [`docs/STATUS.md`](docs/STATUS.md) tracks the live experimental state.
- [`docs/TECHNICAL.md`](docs/TECHNICAL.md) is the technical and process
  appendix.
- [`docs/MAP.md`](docs/MAP.md) maps code, data flow, and provenance.
- [`duraseed_pilot_config.yaml`](duraseed_pilot_config.yaml) contains the
  machine-readable frozen configuration.

Local validation is credential-free:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run duraseed validate
```

Remote execution is separately authorized and cost-capped.

## Development note

This is an independent research project and a learning experience. I use Codex
for much of the implementation. Agent-assisted development makes work at this
scale possible for one person, but it can also encourage unnecessary
scaffolding. I have tried to keep that tendency under control by narrowing the
study, removing obsolete paths, and keeping the public story centered on the
science.

Constructive criticism is welcome—on the scientific design, statistics,
implementation, or unnecessary complexity. Please [open an
issue](https://github.com/ely2ba/duraseed/issues) with questions or suggestions.

## License

[Apache License 2.0](LICENSE)
