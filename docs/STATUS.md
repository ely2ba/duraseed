# Project status

| Phase gate | Status | Public evidence |
| --- | --- | --- |
| M0 + format training | **DONE** | Common M0 was selected after the [accepted canonical format-selection RFC](rfc-format-selection-estimand.md). |
| MAPS calibration | **DONE** | Ratified `shortest2_cap2` / `3e-4` / step-480 recipe; see the [public provenance record](../provenance/v0-map.md#ratified-stage-b-decision). |
| Boundary scan cohort 1 | **DONE** | The [frozen finalist gate](rfc-boundary-finalist-gate.md) produced 15 of 38 passing families, locked by the [extension protocol](rfc-boundary-family-extension.md). |
| Replay equivalence | **DONE** | The [v0 → v1 replay](../provenance/replay-equivalence.md) completed with zero divergences. |
| v1 rebuild, runtime, and live runners | **DONE** | Runtime and local execution paths for the smoke, boundary extension, acquisition calibration, and Pilot 0 pass locally; the [boundary extraction](rfc-boundary-protocol-extraction.md) and [calibration closure](rfc-calibration-instrument-closure.md) were equivalence-verified. |
| Live smoke gate | **DONE** | The real remote path passed exact online/offline reward parity, stop-contract verification, full-state resume, and weights-only branching under its `$25` cap; its billing closeout was completed before the boundary launch. |
| Boundary extension cohorts 2–3 | **REMOTE COLLECTION DONE** | Cohorts 2 and 3 completed broad sampling, refinement, and confirmation. |
| Three-cohort panel freeze | **DONE** | The [accepted amendment](rfc-boundary-panel-algebraic-deduplication.md) reduced the fixed 49 to 37 exact algebraic representatives. All 37 passed the unchanged capacity audit; disjoint 12/12 panels and 12 separate intermediates were frozen. Archived and current production split builders then produced byte-identical 384-row train and 192-row gate manifests. |
| Teacher-exposure calibration | **RETIRED AFTER THREE BOUNDED FAILURES** | The original dose-2 run, [progressive repair](rfc-teacher-exposure-progressive-repair.md), and [dose-8 M1](rfc-teacher-seed-dose8-low-repetition.md) found no checkpoint passing every gate in both orientations. M1 failed jointly at updates 6 and 10; seed-17 update 14 had only 586 valid answer tags after 640 target samples, so even perfect remaining samples could not reach 745/768. It was stopped safely before seed-37 update 14. Raw evidence is retained; none is a method outcome. The [direct-M0 amendment](rfc-direct-m0-stage-a-origin.md) retires the task-specific seed without weakening a gate. |
| Stage-A calibration | **CURRENT GATE — DIRECT M0** | B-S and B-G now branch weights-only from the same M0 with fresh optimizers. The existing 10-update LR screen is also B-G's rollout-health check; no new calibration campaign is added. |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | If B-S/B-G separates reproducibly, first test supervised replay of a prospectively frozen verifier-correct subset of B-G rollouts; then decide whether conditional B-O or optional G-U adds enough value. G-B is a duplicate of direct-M0 B-G and R-G's teacher-allocation question is retired. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no B-S/B-G stability–future-learnability result is
claimed.

The three pre-Pilot gates have caps of `$25`, `$120`, and `$300`. Remote
execution for the smoke and boundary gates is complete. All three teacher-
exposure attempts are stopped and retained as calibration evidence. Stage-A
calibration from M0 is the current acquisition-calibration gate; its Stage-A-
only launch cap is `$153.32` inside the unchanged `$300` lifetime acquisition-
calibration cap. Pilot 0 is separately capped at `$600` and remains unstarted.
