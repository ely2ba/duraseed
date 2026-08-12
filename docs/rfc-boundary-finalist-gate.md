# Boundary finalist and confirmation rule

Status: frozen before Stage-2 refinement outcomes were inspected (10 August
2026).

The public protocol fixes the adaptive three-stage scan and the scientific
requirements for a usable family, but deliberately leaves calibration
thresholds open. This note closes the thresholds needed to move from targeted
refinement to held-out-item confirmation. It does not select a family or panel.

## Stage-2 finalist gate

Every success-positive broad-screen family and every seeded zero-success audit
family receives 16 total observations on each of its four existing numeric
items. A family advances to Stage 3 only when all of the following hold:

- exact success occurs on at least two distinct items;
- equal-item posterior `I8` is at least `0.25`;
- equal-item posterior success is at most `0.50`;
- valid answer-tag compliance is at least `0.50`;
- length-stop frequency is at most `0.50`.

Audit families are assessed identically to success-positive families. All
families passing this gate advance; Stage 2 is not used to rank a convenient
top 24. The format and length thresholds are broad bottleneck safeguards, not
equivalence claims.

## Stage-3 confirmation

For each finalist, generate four new exact-family numeric items with task IDs
disjoint from the broad/refinement items. Draw 16 selected-M0 completions per
new item using the already fixed renderer, sampling distribution, and
4096-token cap. Combine these observations with the four refined items, giving
eight items and 128 observations per family.

A family is eligible for panel matching only when:

- equal-item posterior `I8` is at least `0.25`;
- equal-item posterior success is at most `0.50`;
- exact success occurs on at least four distinct items;
- no single item contributes more than `25%` of observed successes;
- answer-tag compliance is at least `0.50`;
- length-stop frequency is at most `0.50`;
- exact enumeration is complete and the required disjoint split instances can
  be materialized without leakage.

The four-item success requirement makes the protocol's 25% domination rule
meaningful rather than asking four sparse original items to exhibit perfectly
balanced counts. No threshold may be relaxed after Stage-3 outcomes. If fewer
than 24 families qualify, core panel construction stops unresolved rather than
filling the design with weak families or breaking the symmetric cross-over.

The core panel size is frozen at 12/12 before the Stage-3 run. The deterministic
matching rule is frozen after the Stage-2 finalist count and structural
distribution are known but before any held-out-item outcomes are observed.

The pre-match split audit requires per family: 16 `a_seed_train`, 8
`a_seed_gate`, 64 `a_rl_train`, 16 `a_monitor`, and 22 `a_validation` numeric
instances. After the provisional 24-family match, each selected family must
also supply 22 `a_test_single` instances before the match becomes final. The
single-family requirement means exactly one selected intervention family is
present in the task's complete exact family set; it does not require only one
globally valid family. The audit excludes every candidate and already assigned
protected-split numeric key. Ordinary capacity is probed over twice the
required prefix; the panel-aware single-family filter may scan eight candidate
indices per required item. Generic `a_test_multi` diversity is built separately.
The actual Stage-A prompt-pool demand is rechecked after duration calibration
rather than inferred from this bounded audit.

## Frozen panel matching

The allocation seed is derived from root seed `11` under
`family_selection.panel_allocation` and fixed as `6448342238137851489` before
Stage-3 outcomes. If more than 24 families pass confirmation, retain the 24
with highest confirmed `I8`, then greatest verified disjoint-instance capacity;
the allocation seed breaks exact ties. This ranking is predeclared and cannot
be changed after confirmation.

Compute equal-weight Gower distances on the ten public matching covariates,
using observed ranges for numeric variables and exact mismatch for categorical
profiles. Greedily form the closest remaining disjoint family pairs, then
enumerate all pair orientations and choose the A/B assignment minimizing first
the maximum and then the mean covariate imbalance. The allocation seed fixes
the otherwise arbitrary panel label and exact ties. Save directional
nearest-other-panel distances for the descriptive sentinel-transfer analysis.
