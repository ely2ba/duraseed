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
| Three-cohort panel freeze | **UNRESOLVED** | Exact old/current replay passed. It found 49 observation-eligible candidates and selected 24 matched families, but only 12/24 passed the required single-family test split. No panels were published; this is a structural feasibility outcome, not a B-S/B-G result. |
| Teacher-dose / Stage-A calibration | **BLOCKED ON PANELS** | The local `$300` execution path is ready, but it cannot run without a valid frozen panel artifact. |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | Use the reconnaissance to decide and, where justified, run B-O and optional G-B/R-G/G-U before freezing the confirmatory matrix. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no B-S/B-G stability–future-learnability result is
claimed.

The three live pre-Pilot gates have caps of `$25`, `$120`, and `$300`. Remote
execution for the smoke and boundary gates is complete; acquisition calibration
would be the next paid gate only after the panel design is resolved. Pilot 0 is
separately capped at `$600`.
