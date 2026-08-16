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
| Teacher-dose / Stage-A calibration | **READY TO LAUNCH** | Frozen panel and prompt sources authenticate. The lag-cleared provider usage export and protected-reserve reconciliation pass; the existing execution path remains capped at `$300`. |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | If B-S/B-G separates reproducibly, first test supervised replay of a prospectively frozen verifier-correct subset of B-G rollouts; then use the reconnaissance to decide whether B-O or optional G-B/R-G/G-U adds enough value. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no B-S/B-G stability–future-learnability result is
claimed.

The three live pre-Pilot gates have caps of `$25`, `$120`, and `$300`. Remote
execution for the smoke and boundary gates is complete; acquisition calibration
would be the next paid gate only after the panel design is resolved. Pilot 0 is
separately capped at `$600`.
