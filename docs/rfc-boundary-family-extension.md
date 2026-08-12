# Fixed 128-family boundary extension

Status: frozen on 10 August 2026 after the first Stage-3 confirmation was
complete and before any extension-family output was generated or inspected.

The completed boundary confirmation evaluated 38 finalists under the frozen
Stage-3 rule. Fifteen families passed every observation and split-capacity
gate, fewer than the 24 required for the symmetric crossed panels. This result
is valid calibration evidence: it is not rerun, reinterpreted, or replaced.

This amendment permits one fresh deterministic 128-family extension to
complete both the panel candidate pool and the required non-panel intermediate
pool. The extension is executed as two operational 64-family subcohorts, both
fixed before sampling. It changes the breadth of boundary discovery, not the
family estimand, per-family evidence budget, inclusion gates, panel size, or
matching rule.

## Locked original evidence

The 15 passing families from the completed confirmation are locked as eligible
families. Locking preserves their eligibility decisions; it does not guarantee
panel or intermediate-pool membership after the combined frozen ranking.

The 23 families that failed the completed Stage-3 observation gate remain
failed and receive no retest or additional samples. No other family from the
first 64-family broad cohort is reconsidered. Existing generations, rewards,
summaries, and capacity audits remain unchanged.

## Fresh extension

The extension contains exactly the next 128 distinct exact TCES families in
deterministic first-appearance order from the same authenticated
`a_candidate` generator stream. Concretely, enumerate that stream under the
same root seed and generator configuration, retain first occurrences of exact
family IDs, exclude the first 64 distinct families used by the completed broad
screen, and freeze distinct-family ordinals 65 through 192. No model output may
affect membership or ordering.

For bounded execution, the extension is partitioned before sampling into:

- subcohort 1: distinct-family ordinals 65 through 128; and
- subcohort 2: distinct-family ordinals 129 through 192.

Both subcohorts must run regardless of the first subcohort's eligible-family
yield. The first subcohort cannot trigger early selection or cancellation of
the second; this prevents an outcome-dependent stopping rule.

All numeric variants must be deterministic and task-disjoint from the
completed broad, refinement, and confirmation manifests. The preflight records
the exact family list and manifest identities before remote sampling.

## Unchanged three-stage funnel

Each operational 64-family subcohort uses the same evidence allocation and
decisions as the completed scan:

1. **Stage 1:** four numeric items per family and four completions per item.
2. **Stage 2:** refine every success-positive family plus a deterministically
   seeded 12-family audit selected from that subcohort's zero-success families
   under the unchanged audit rule. Bring each of the four existing items to 16
   total completions. Apply the frozen finalist gate to every refined family;
   do not rank a convenient subset.
3. **Stage 3:** first apply the unchanged deterministic split-capacity audit to
   every Stage-2 gate passer. Paid confirmation samples exactly the families
   that pass both the Stage-2 gate and that pre-confirmation audit; a known
   capacity failure is not sampled. For each admitted family, evaluate four
   new held-out numeric items with 16 completions per item, combine those with
   the four refined items, and apply the frozen eight-item observation gate.

All renderer, checkpoint, temperature, sampling, token-cap, exact-verification,
format, truncation, posterior, multi-item-success, single-item-domination, and
disjoint-capacity requirements remain those in
[`rfc-boundary-finalist-gate.md`](rfc-boundary-finalist-gate.md). In particular,
no threshold is relaxed because the first cohort yielded only 15 eligible
families.

## Combined decision and stopping rule

Only after both fresh subcohorts complete, form the eligible union from:

- the 15 locked passers from the completed confirmation; and
- families from either fresh subcohort that pass the pre-confirmation
  split-capacity audit and every unchanged Stage-3 observation gate.

Continuation requires at least 36 eligible families: 24 for the two core
panels and 12 distinct non-panel intermediate families required by the frozen
Stage-A prompt pools. Rank the combined eligible union once by greatest
confirmed `I8`, then greatest verified disjoint-instance capacity, with
allocation seed `6448342238137851489` resolving exact ties. The top 24 enter
the unchanged ten-covariate Gower matching and orientation rule to construct
two 12-family panels. Eligible ranks 25 through 36 become the frozen
intermediate family pool and cannot overlap either panel.

If the union contains fewer than 36 families, Phase 3 remains unresolved. Do
not add another subcohort, retest failed families, weaken a gate, shrink the
core panels or intermediate pool, or begin teacher-dose calibration
automatically.

This is a pre-result protocol amendment for one bounded 128-family extension.
Pilot 0 has not started, and it authorizes no Stage-A training or sealed-test
access.
