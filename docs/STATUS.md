# Project status

| Phase gate | Status | Public evidence |
| --- | --- | --- |
| M0 + format training | **DONE** | Common M0 was selected after the [accepted canonical format-selection RFC](rfc-format-selection-estimand.md). |
| MAPS calibration | **DONE** | Ratified `shortest2_cap2` / `3e-4` / step-480 recipe; see the [public provenance record](../provenance/v0-map.md#ratified-stage-b-decision). |
| Boundary scan cohort 1 | **DONE** | The [frozen finalist gate](rfc-boundary-finalist-gate.md) produced 15 of 38 passing families, locked by the [extension protocol](rfc-boundary-family-extension.md). |
| Replay equivalence | **DONE** | The [v0 → v1 replay](../provenance/replay-equivalence.md) completed with zero divergences. |
| v1 rebuild, runtime, and live runners | **DONE** | Runtime and local execution paths for the smoke, boundary extension, acquisition calibration, and Pilot 0 pass locally; the [boundary extraction](rfc-boundary-protocol-extraction.md) and [calibration closure](rfc-calibration-instrument-closure.md) were equivalence-verified. |
| Live smoke gate | **NOT STARTED** | This is the next engineering gate. It is capped at `$25` and must demonstrate real-data reward parity, the stop contract, full-state resume, and weights-only branching before scientific runs begin. |
| Boundary extension cohorts 2–3 | **IN PROGRESS** | Cohort-2 broad and refinement evidence is complete. Cohort-2 confirmation and cohort 3 remain, followed by the final three-cohort panel freeze. |
| Teacher-dose / Stage-A calibration | **IN PROGRESS** | The local `$300` execution path is complete. After the panels freeze, this gate will select teacher dose and allocation, Stage-A learning rates and duration, one common RL objective, and one shared completion cap under the [pre-Pilot estimands RFC](rfc-prepilot-estimands-and-selection.md). |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | Use the reconnaissance to decide and, where justified, run B-O and optional G-B/R-G/G-U before freezing the confirmatory matrix. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no scientific result is claimed.

The remaining path to the Pilot-0 freeze is capped at `$25 + $120 + $300 =
$445`; Pilot 0 is separately capped at `$600`. Each live gate closes with a
cost reconciliation before the next one begins.
