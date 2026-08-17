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
| Teacher-exposure calibration | **CURRENT GATE — M1** | The first run remains failed diagnostic evidence. The [dose-2 bounded repair](rfc-teacher-exposure-progressive-repair.md) also found no joint pass: update 8 was the last complete checkpoint in both orientations and passed format but failed the mixed-group-rate gate; seed-17 update 12 passed the mixed-group-rate gate but missed format by 3/768, so collection stopped after 67/96 sentinel generations and seed-37 update 12 was never launched. The run is truthfully marked interrupted, raw evidence is retained, and billing is pending. The current gate is the accepted [dose-8 M1 amendment](rfc-teacher-seed-dose8-low-repetition.md): one `1e-4` continuous 6/10/14-update trajectory in each orientation with unchanged panels, decoder, manifests, and gates. |
| Stage-A calibration | **NOT STARTED** | Begins only if one teacher-exposure checkpoint passes every existing gate in both orientations. |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | If B-S/B-G separates reproducibly, first test supervised replay of a prospectively frozen verifier-correct subset of B-G rollouts; then use the reconnaissance to decide whether B-O or optional G-B/R-G/G-U adds enough value. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no B-S/B-G stability–future-learnability result is
claimed.

The three pre-Pilot gates have caps of `$25`, `$120`, and `$300`. Remote
execution for the smoke and boundary gates is complete. The first acquisition-
calibration attempt and the bounded dose-2 repair are both stopped; the first
run's terminal artifact and all raw repair evidence remain unchanged. The
repair run's authoritative billing reconciliation is pending. The dose-8 M1
amendment is the current acquisition-calibration gate. Pilot 0 is separately
capped at `$600` and remains unstarted.
