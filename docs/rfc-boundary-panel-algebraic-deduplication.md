# Algebraic de-duplication before panel matching

Status: accepted prospectively by Ely on 15 August 2026, after the completed
boundary evidence and before any Stage-A or B-S/B-G outcome. The amended local
freeze has not yet been executed.

## Why this amendment exists

The original three-cohort reducer matched 24 of 49 observation-eligible TCES
families and only afterward applied the 22-item `a_test_single` capacity gate.
Twelve selected families then had zero qualifying test items. The failure was
not model performance: several nominal family IDs describe the same exact
rational function, so protecting them together makes none of them an isolated
single-family test target.

An outcome-blind follow-up established that 31 singleton algebraic families
already pass 22/22 under protection by all 49 candidates. The 49 fixed family
IDs form 37 exact algebraic classes, with non-singleton class sizes
`6, 3, 3, 2, 2, 2`. This is enough to retain the intended 24 panel families
and 12 separate intermediate families without collecting new data or weakening
the split-capacity requirement.

## Fixed source and class rule

The source universe is exactly the 49 observation-eligible families in
`boundary-panel-freeze-20260815T032009Z`. It cannot be expanded, rescored, or
filtered using method outcomes.

Two family IDs are in the same class exactly when their expressions are equal
as expanded rational functions of operand ranks `r1` through `r5`. Equality is
checked with exact integer polynomial arithmetic by cross multiplication; it
is not a numerical probe. A class may contain different tree shapes or operator
multisets. Those differences remain reported, and the chosen representative's
ten matching covariates are carried unchanged. They do not make algebraically
identical families independent `a_test_single` targets.

Within each class, retain the highest family under the already-frozen order:
greatest confirmed `I8`, then greatest ordinary five-split capacity, with
allocation seed `6448342238137851489` used only for exact ties. Reapply that
same order to the resulting 37 representatives before either pool is selected.

## Two explicit eligibility pools

The panel and intermediate pools have different, pre-existing purposes and
therefore different gates.

For the **panel pool**, audit every algebraic representative while protecting
all representatives. Apply the unchanged requirements of 16 `a_seed_train`, 8
`a_seed_gate`, 64 `a_rl_train`, 16 `a_monitor`, 22 `a_validation`, and 22 of 22
`a_test_single`, including the unchanged probe and scan multipliers. Feed only
passing representatives to the unchanged 24-family matcher and unchanged
12/12 orientation rule.

For the **intermediate pool**, start from the full ordinary-capacity-qualified
representative universe, exclude every selected panel family, preserve the
same frozen ranking, and take the first 12. Intermediate families are required
to pass the original five ordinary splits but are not evaluation panels and do
not require `a_test_single` capacity.

The emitted selected-panel capacity rows contain exactly the 24 matched panel
families in matcher order. The amendment artifact separately records all
classes, representatives, panel-qualified IDs, the explicit intermediate IDs,
both eligibility rules, and hashes of its fixed source.

## Stopping rule and scope

If fewer than 24 representatives pass the panel gate, or fewer than 12
ordinary-eligible representatives remain outside the selected panels, the
freeze remains unresolved. Do not shrink panels, shrink the intermediate pool,
relax 22/22, add another boundary cohort, or begin Stage A automatically.

This amendment authorizes a local, zero-cost replay only. It changes no model
evidence, matching covariate, allocation seed, panel size, Stage-A recipe,
remote budget, or scientific claim. Pilot 0 remains unstarted until a complete
amended artifact is published and authenticated downstream.
