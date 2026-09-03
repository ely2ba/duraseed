# DuraSeed technical appendix

This page is the detailed companion to the [project overview](../README.md).
It explains how the study is made executable and auditable without putting the
entire calibration history on the front page.

## Which document governs what

- [`PROTOCOL.md`](../PROTOCOL.md) is the public scientific contract.
- The frozen [capability-targeted acquisition
  amendment](amendment-capability-targeted-acquisition.md) governs the current
  Pilot where it is more specific than the earlier protocol.
- [`duraseed_pilot_config.yaml`](../duraseed_pilot_config.yaml) is the
  machine-readable configuration.
- [`STATUS.md`](STATUS.md) reports what has actually run.
- [`MAP.md`](MAP.md) maps components, evidence flow, and provenance locations.

Decision records preserve why a rule changed. They do not silently rewrite
older runs or turn calibration outcomes into method results.

## Frozen experimental specification

### Common origin

- Model: `Qwen/Qwen3.5-9B-Base`.
- Adaptation: rank-32 LoRA.
- Origin: the same non-expiring, format-capable M0 weights for B-S and B-G.
- Branching: weights-only restore with a fresh optimizer for each method and
  seed.
- M0 is task-cold for TCES, not raw-pretrained: its prior intervention is
  task-agnostic output-format training.

### Stage A: TCES

TCES is an exact arithmetic-expression task. The generator fixes the operands,
target, and allowed operations; the verifier checks syntax, one-use-per-input,
and exact arithmetic equivalence. Invalid and length-stopped outputs receive
zero reward.

The frozen family allocation contains two disjoint matched panels of 12
families each plus a separate 12-family intermediate pool. Target and sentinel
panel roles cross across paired seeds. Training, cadence monitoring,
validation, and final tests use distinct authenticated manifests.

The two core procedures are:

- B-S: fixed verified solver-teacher SFT, LR `1e-4`, batch 32, trained through
  294 updates in Pilot.
- B-G: on-policy verifier-rewarded group-relative RL, LR `1e-5`, group size 8,
  trained through 50 updates.

Both save a sampler checkpoint and paired trainable state every ten updates.
Sampler state supports evaluation; trainable state is required to initialize
Stage B. A completed B-G update with zero mixed-reward groups or nonfinite
training metrics is a counted procedure failure, never a lucky-seed reroll.

### Post-hoc capability matching

Cadence evaluations use one draw on each targeted item. After both frozen
Stage-A schedules finish, the matcher considers real cadence checkpoints only.
It selects the pair with the smallest targeted exact-success difference whose
approximate 95% intervals overlap; ties prefer the earlier total training and
then earlier B-S/B-G checkpoints.

If no pair overlaps, `matching.json` records `unavailable`, the seed is not
replaced, and the run stops before selected-profile evaluation or Stage-B
spend. Matching is deliberately narrow: it aligns targeted exact success, not
the models' complete behavior.

### F3 pre-Stage-B profile

At the selected checkpoints the study records:

- target, sentinel, and per-family exact success;
- family coverage curves;
- output-validity and length-stop rates;
- completion-length distributions;
- supported Pass@k values;
- verifier-confirmed strategy diversity;
- sampled-token surprisal;
- baseline MAPS performance before any Stage-B training; and
- prompt/prefill, sampled, and training tokens plus storage and dollar cost.

These are descriptive outcomes, not extra matching gates. In particular,
sentinel movement may be part of the phenomenon and is not matched away.
The post-hoc analysis of archived LoRA tensors is supplementary, outside F3
and every decision path; it changes no profile requirement or gate.

### Stage B: MAPS

Stage B is the same for both selected checkpoints:

- task/profile: MAPS `shortest2_cap2`;
- LR `3e-4`, batch 32, 480 updates;
- fresh Stage-B optimizer continuing the selected Stage-A LoRA weights;
- evaluation grid `[0,1,2,5,10,20,40,80,160,320,480]`;
- 512 `b_validation` items with 16 draws per point;
- 384 `a_monitor` items with four draws at each post-origin point; and
- 512 `a_validation` items with 16 draws before Stage B and at update 480.

Supported Pass@k is computed from already-required draws only: `k={1,4}` for
four-draw monitor evidence and `k={1,4,16}` for sixteen-draw evidence.

## Outcomes

F1 measures retention of TCES as MAPS training proceeds. F2 measures MAPS
learning, including raw-gain curves and AUC under the frozen schedule. F3 is the
full matched pre-Stage-B profile above. Resource curves keep prefill, sampled,
and trained tokens separate because B-S and B-G perform different kinds of
work.

The first evidence unit is a paired seed, not a collection of independent
samples. Pilot 0 is complete with two independently matched pairs: seed 11 and
seed 29. Their [public overview](../README.md), full F1/F2/F3 readouts, paired
offline analyses, and geometry reports are banked together. Pair-2 matching was
mechanically blind to pair-1 contrasts, and its geometry-based F1/F2 ordering
was [recorded prospectively](results/pilot0-pair2-prediction.md) before outcomes
were opened. Later fresh pairs are variance reconnaissance, not replacements
for a failed or unavailable seed.

The protocol conditionally prioritizes supervised replay of a frozen,
verifier-correct subset of B-G rollouts if a reproducible difference appears.
The newer gauge-rescaling candidate recorded with the
[prospective prediction](results/pilot0-pair2-prediction.md) is a separate
hypothesis, not a replacement for that priority. This Pilot-0 record does not
authorize an experiment or spend.

## How the pre-Pilot design evolved

### Format-capable M0 and exact tasks

M0 selection, the TCES token cap, and the MAPS Stage-B recipe were calibrated
before the acquisition comparison. The relevant public records are the
[format-selection RFC](rfc-format-selection-estimand.md), the [max-token
ratification](rfc-acquisition-max-token-ratification.md), and the Stage-B
section of [`PROTOCOL.md`](../PROTOCOL.md#stage-b-probe-calibration).

### Panels and separation

The first panel construction exposed exact algebraic duplicates that consumed
family-specific evaluation capacity. The outcome-blind [algebraic
deduplication amendment](rfc-boundary-panel-algebraic-deduplication.md)
collapsed equivalent families before matching, preserving the original
eligibility gate and producing disjoint authenticated panels.

### Retired teacher warm start

A task-specific shared teacher seed was explored as a possible common origin.
Three bounded attempts found no checkpoint that was both capable and stable
across the crossed orientations, so the warm start was retired rather than
expanded into an open-ended tuning ladder. The raw attempts remain calibration
evidence. See the [diagnostics](warm-start-diagnostics.md), [progressive
repair](rfc-teacher-exposure-progressive-repair.md), [dose-8
amendment](rfc-teacher-seed-dose8-low-repetition.md), and [direct-M0
decision](rfc-direct-m0-stage-a-origin.md).

### Answer-tag contract correction

The first direct-M0 Stage-A screen used an outer-wrapper diagnostic that was
stricter than the TCES prompt, solver targets, and verifier. The [answer-tag
contract correction](rfc-stage-a-answer-tag-contract.md) preserved the raw
screen, replaced only the incompatible diagnostic, and froze the selected B-S
and B-G learning rates before Pilot outcomes existed.

### Capability-dose validation

The frozen [capability-targeted amendment](amendment-capability-targeted-acquisition.md)
then ran one bounded B-S dose trajectory. It confirmed that the planned overlap
zone was reachable and validated trainable-state save followed by weights-only
restore. That branch authorized Pilot implementation; it did not select a
shorter Pilot B-S duration.

## Execution and failure handling

- Every remote launch has an explicit human-approved ceiling and its own
  session/run identity.
- Remote calls are journaled before issue and settled after completion. An
  ambiguous in-flight request is never silently retried.
- Scientific failures—such as B-G update-health failure or unavailable
  matching—are retained and count. Infrastructure interruption is handled
  separately and resumes the same seed only from durable state.
- Cadence checkpoints retain both sampler and trainable-state paths.
- Stage B restores selected Stage-A weights without inheriting the Stage-A
  optimizer.
- Final tests remain sealed through Pilot and calibration.
- Further pairs require separate launch authorization. Pair 2's dated
  authorization and outcome-independent criteria are recorded in
  [`STATUS.md`](STATUS.md).

## Budget and accounting

The accepted planning ceiling is `$774.04` per paired Pilot seed and
`$1,548.08` for two pairs. Each pair is separately authorized; the exact
manifest-derived preflight is `$772.40` for each of pairs 1 and 2. These are
worst-case ceilings assuming full token-cap use, not expected spend.

Accounting records prefill, sampled, and training tokens separately, adds
fixed checkpoint storage, and preserves hash-bound console evidence. As
authorized on 2026-08-30, account-wide lifetime formulas are retired because
the account also has non-DuraSeed usage: the launch condition is DuraSeed
run-ledger actuals plus pair 2's preflight within `$1,548.08`; the saved grant
balance is a separate sanity check, and final CSV attribution is deferred to
end-of-project accounting. [`STATUS.md`](STATUS.md) records that authorization;
the [frozen amendment](amendment-capability-targeted-acquisition.md#cost-and-authority)
preserves the earlier planning calculation and transcription corrections.

## Code and evidence map

- `src/duraseed/tasks/`: TCES and MAPS generators, solvers, renderers, and exact
  verifiers.
- `src/duraseed/data/`: manifests, splits, leakage checks, panels, and matching.
- `src/duraseed/training/`: datum/reward construction and pure decision
  reducers.
- `src/duraseed/evaluation/`: item-level acquisition, retention, learning, and
  frontier analysis.
- `src/duraseed/runtime/`: bounded service clients, token ledgers, sampling,
  and checkpoints.
- `src/duraseed/runners/`: Pilot Stage-A training and checkpointing, matching,
  selected profiles, Stage-B training/evaluation, and retained calibration
  orchestration around the frozen reducers.
- `frozen/v0/`: immutable carried calibration evidence.
- `provenance/`: hashes, replay equivalence, billing lineage, and public
  projections.
- `runs/`: generated local/live evidence, intentionally excluded from Git.

For exact paths and hashes, continue to [`MAP.md`](MAP.md) and
[`provenance/v0-map.md`](../provenance/v0-map.md).

## Reproduction and local validation

Python 3.11 or 3.12 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv lock --check
uv run duraseed validate
```

These commands are local and credential-free. Remote execution additionally
requires authenticated source artifacts, explicit project identity, a
DuraSeed-scoped billing record, a clean committed tree, and human cost
authorization.
