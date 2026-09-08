# Replay preparation: evidence audit

2026-09-05. Local evidence and code, not the earlier assistant assessment, are
the basis of this audit. Pilot 0 remains frozen. No new training or sampling
has occurred for replay-v1. Earlier gauge work is dropped and excluded.

## What exists

| Evidence class | State and use |
| --- | --- |
| Calibration and capability-dose work | Completed preparatory work; establishes the Pilot recipe and operational history, not additional Pilot seeds or confirmatory replications. Retained under the original run directories and linked from `docs/STATUS.md`. |
| Pilot pair 1, seed 11 | Completed; selected B-S@140/B-G@30, 31/96 each on the matching panel; both MAPS continuations reached update 480. |
| Pilot pair 2, seed 29 | Completed; selected B-S@40/B-G@20, 17/96 each on the matching panel; both MAPS continuations reached update 480. |
| Original offline analyses | Retrospective uncertainty, failure anatomy, baseline concentration and adapter summaries. Their definitions and outputs are preserved. |
| Pair-2 mechanical prediction | Prospective geometry-based prediction recorded before outcome inspection; the later explicit scoring rule preceded release. Original record and scoring stay unchanged in `docs/results/pilot0-pair2-prediction.md`. Neither is a randomized intervention on factor scale. |
| Software checks/recovery | Engineering evidence for execution and integrity, not scientific replications. Documented recovery does not create an extra seed. |
| Replay-v1 | Local audit, corpora, implementation and tests only. No experimental outcomes yet. |
| Gauge/local-GPU plans | Unexecuted and now dropped; existing unrelated working files were not erased or incorporated. |

Source runs are `runs/pilot0/pilot0-pair1-seed11-20260825T125100Z` and
`runs/pilot0/pilot0-pair2-seed29-20260830T204211Z`. Each retains original
`pilot-inputs/` manifests, preflight/M0 lineage, acquisition generations,
rewards, update metrics, call journals, matching, selected F3 profiles and
Stage-B evaluations. Original public reductions are under
`artifacts/pilot0-pair{1,2}-{readout,offline-analysis}/`. Source and recovery
billing records remain attributable to their DuraSeed runs; unrelated account
usage is not inferred from console deltas.

## Reproduced results and new existing-data analysis

The new package authenticates 88 evaluations: 39,424 item-checkpoint records
and 428,032 completion records. All four stored F1/F2 cells reproduce exactly.
All 28 original bootstrap intervals reproduce with maximum discrepancy 0.0,
using their original seeds and estimands. The original `raw_gain_auc` field
name does not make it raw Pass@1: those stored cells use posterior scores.
The new report keeps these distinct.

All contrasts below are B-G minus B-S on **raw** Pass@1:

| Quantity | Pair 1 | Pair 2 |
| --- | ---: | ---: |
| Targeted retention AUC0–20 | +0.01966146 | +0.03382161 |
| Absolute MAPS AUC0–40 | −0.00926514 | +0.00254974 |
| Own-baseline MAPS gain AUC0–40 | −0.05919189 | −0.05091705 |
| Absolute MAPS AUC0–480 | −0.01056824 | +0.11391080 |
| Own-baseline MAPS gain AUC0–480 | −0.06049500 | +0.06044401 |

The targeted early-retention difference survives this audit. A general claim
that B-S has better early *absolute* new-task performance does not: the pair-2
sign reverses when the inherited baseline is not subtracted. Gain and absolute
performance answer different questions. Early TCES decline is not solely a
format failure: B-S's output-valid targeted completions increase from update 0
to 2 while exact successes decrease in both pairs. This does not identify a
cause of the difference between training procedures.

The complete [analysis](../artifacts/replay-v1/pilot-audit/README.md) includes
raw/posterior trajectories, gain decompositions, paired item-trajectory
intervals, chronological downstream-performance paths, all four fixed
first-attainment sensitivities and unavailable cases, failure-code tables,
overlapping diagnostic rates and conditional/unconditional denominators.
Its notebook, machine-readable counts and figure data reproduce those outputs.

## Replay data availability

All 50 B-G acquisition updates per block are locally complete and authenticated.
The 12,800 original draw records reproduce their stored text and exact verifier
fields; each rollout's generating sampler predates its optimizer step.

| Block | Sampled unique prompts | Correct uncapped draws | Shared R-S/R-P prompts | Stage-A R-S/R-P train tokens |
| --- | ---: | ---: | ---: | --- |
| 11 | 800 | 1,479 | 579 | 1,889,739 / 15,548,318 |
| 29 | 800 | 1,416 | 557 | 1,893,391 / 15,101,772 |

No corpus or family was replenished. The success-conditioned distribution is
reported, not balanced away. Both arms present each ordered prompt at exactly
the same updates, but their response lengths and trained token totals differ.
Full traces require at most 4,217 rendered tokens. Explicit full-length
conversion avoids the old 1,024-token default without changing Pilot code.

## Launch direction and remaining limitations

1. **Credit and storage liability.** The exact token schedule plus proposed
   bounded 30-day storage gives a conditional $2,125.11 package ceiling,
   including the ≤$20 smoke and 10% recovery reserve. The signed-in 2026-09-05
   console shows $3,743.52 available. Other commitments and organization backup
   retention remain unverified. The 2GB-state/0.5GB-sampler reservations exceed
   existing same-rank file sizes but are not a provider guarantee. After disclosure
   the owner directed launch, twice. Execution uses this later direction and the
   $2,125.11 operational cap without presenting unknowns as verified or describing
   the allowance as an unconditional provider-invoice maximum. Existing per-call
   caps, checkpoint size/expiry checks and no-ambiguous-retry rules remain active.
2. **Provider identity/availability.** Original metadata identifies the Tinker
   model and M0 paths, not an exact upstream weight revision. Local tokenizer
   compatibility with stored text is not evidence that the public Qwen snapshot
   was Tinker's original. The new package restores the original M0 path and
   requires full datum-token agreement; no local public-base substitution.
   Current M0 availability and full-length acceptance remain untested remotely.
3. **Scientific availability.** Fixed candidate matching can legitimately
   produce `NO_MATCH`. No extra nominees, training, tolerance changes or samples
   are authorized to obtain a match. The intervention can test a trace-source
   bundle, not isolate objective, length, strategy or parameter scale.

The existing data cannot identify whether successful archived-policy traces
under the same supervised acquisition procedure produce different subsequent
retention. That is the bounded paid contrast, if it is explicitly approved.
Related work already includes SFT on RL traces and effects of initialization
on later learning; the [five-paper full-text audit](replay-v1-related-work.md)
records these overlaps rather than claiming an unqualified first.
