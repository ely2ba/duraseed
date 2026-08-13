# Project status

| Phase gate | Status | Public evidence |
| --- | --- | --- |
| M0 + format training | **DONE** | Common M0 was selected after the [accepted canonical format-selection RFC](rfc-format-selection-estimand.md). |
| MAPS calibration | **DONE** | Ratified `shortest2_cap2` / `3e-4` / step-480 recipe; see the [public provenance record](../provenance/v0-map.md#ratified-stage-b-decision). |
| Boundary scan cohort 1 | **DONE** | The [frozen finalist gate](rfc-boundary-finalist-gate.md) produced 15 of 38 passing families, locked by the [extension protocol](rfc-boundary-family-extension.md). |
| Replay equivalence | **DONE** | The [v0 → v1 replay](../provenance/replay-equivalence.md) completed with zero divergences. |
| v1 rebuild, runtime, and live runners | **DONE** | Runtime plus fail-closed smoke, boundary-extension, acquisition-calibration, and Pilot-0 runners pass locally; the [boundary extraction](rfc-boundary-protocol-extraction.md) and [calibration closure](rfc-calibration-instrument-closure.md) were accepted and equivalence-verified. |
| Live smoke gate | **NOT STARTED** | Engineering-only gate capped at `$25`: its billing preflight is reconciled, but SDK credentials are not available to the runner. Pass requires real data, exact online/offline reward parity, a verified stop contract, full-state resume, and weights-only branching. |
| Boundary extension cohorts 2–3 | **IN PROGRESS** | Cohort-2 broad/refinement evidence is preserved; its confirmation and cohort 3 await a passing live smoke and authorization. Panels remain unresolved. |
| Teacher-dose / Stage-A calibration | **IN PROGRESS** | The fail-closed `$300` live runner is complete. Scientific execution awaits frozen panels, authorization, and resolution of the explicitly Proposed [common max-token ratification](rfc-acquisition-max-token-ratification.md) and [panel-split extraction](rfc-calibration-panel-split-extraction.md). The gate must freeze dose, allocation, Stage-A LR/duration, one common RL objective/entropy-collapse decision, and one shared max-tokens value under the [pre-Pilot estimands RFC](rfc-prepilot-estimands-and-selection.md). |
| Pilot 0 | **NOT STARTED** | No pilot execution or sealed-test access. |
| Variance reconnaissance | **NOT STARTED** | After Pilot 0, extend B-S/B-G to 3–4 paired pilot seeds before setting confirmatory power. |
| Mechanism/context pilot decision | **NOT STARTED** | Use the reconnaissance to decide and, where justified, run B-O and optional G-B/R-G/G-U before freezing the confirmatory matrix. |
| Preregistration | **NOT STARTED** | Begins only after variance reconnaissance and the mechanism/context decision freeze and power the confirmatory design. |
| Confirmatory study | **NOT STARTED** | Requires the preregistered design and separate authorization. |

Pilot 0 has not started; no scientific result is claimed.

The committed path to the Pilot-0 freeze is `$25 + $120 + $300 = $445`;
Pilot 0 is separately capped at `$600`. The console balance was refreshed
before the smoke; the `$120` boundary launch additionally requires the passing
smoke and its lag-cleared post-smoke billing reconciliation.
