# Direct-M0 Stage-A origin

Status: accepted on 17 August 2026, after the bounded M1 warm-start run was
stopped and before Stage A, Pilot 0, or any B-S/B-G outcome.

## Problem and evidence

The original protocol required a common task-specific solver-teacher SFT seed
between M0 and the core methods. Three bounded attempts found no checkpoint
passing the frozen health gates in both crossed panel orientations. The first
dose-2 run collapsed into learned nontermination in seed 37. The progressive
dose-2 repair found no joint 4/8/12-update pass. Dose-8 M1 failed jointly at
updates 6 and 10; by 640/768 seed-17 target samples at update 14, only 586 were
valid answer tags, so even perfect remaining samples could reach only 714/768
rather than the required 745. The run was stopped safely before the unnecessary
seed-37 update-14 assessment.

This is a failed nuisance-recipe search, not a B-S/B-G result. It does not
justify weakening a gate, adopting a failed checkpoint, or opening another
seed, dose, learning-rate, duration, group-size, KL, or rehearsal ladder.

The frozen panels were selected at M0 to provide on-policy signal. Across all
24 panel families, predicted informative group-size-8 probability ranges from
0.581 to 0.883, with median 0.739 and mean 0.7485; none is below 0.50. A
separate held-out one-draw M0 check observed 17/192 successes across 10/24
family blocks. The predictions are feasibility evidence, not a guarantee about
realized Stage-A rollouts.

## Amendment

- Retire the task-specific teacher warm start. B-S and B-G branch weights-only
  from the same non-expiring M0 with fresh optimizers.
- B-S is fixed verified solver-teacher SFT from M0. B-G is boundary-weighted,
  verifier-scored group-relative RL from M0. They remain two complete
  acquisition procedures, not an isolated loss-function comparison.
- Retire G-B because it is now identical to B-G. Retire R-G and the targeted-
  versus-random teacher-allocation question. Keep G-U as an optional prompt-
  allocation control and B-O as a conditional mechanism pilot.
- Use the existing 10-update B-G LR screen as the realized rollout-health
  check. Every update must contain a mixed group, as already required, and an
  eligible arm must average at least 0.20 mixed-group rate across updates 1--10.
  If no LR passes the complete screen, stop once for a prospective decision; do
  not revive the warm-start ladder.
- Keep the existing B-S step-10 and step-50 wrapper-compliance and length-stop
  gates. They directly monitor the observed nontermination risk without a new
  checkpoint, sample request, or monitoring system.

The common M0, rank-32 LoRA, renderer, completion cap, frozen panels and prompt
allocation, manifests and split separation, capability-matched follow-up,
Stage-B recipe, outcomes, resource-vector accounting, and claim discipline are
unchanged. Comparability now rests on the literal common M0 and shared task
contract; teacher examples, rollouts, tokens, updates, and cost are reported
separately because the procedures do not consume identical information or
compute.

Historical warm-start RFCs and raw artifacts remain unchanged as pre-Pilot
feasibility evidence. This RFC supersedes them only as an execution gate.
