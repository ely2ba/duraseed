# Phase 0b billing reconciliation

The delayed raw Tinker billing export covers
`2026-08-09T00:00:00Z` through `2026-08-11T16:00:00Z` for project
`7727a6e3-fadb-4b07-9801-721221235e1e`.

- Raw events: 156
- Attributed sessions: 24
- Prefill tokens: 3,570,928
- Cached-prefill tokens: 1,064,373
- Sample tokens: 12,579,816
- Train tokens: 10,681,651
- Checkpoint operations: 128
- Storage: 948.1852751426779 GB-hours
- Token cost at pinned prices: `$43.221298049000005`
- Storage cost at `$0.10/GB-month`: `$0.13169239932537194`
- Reconciled total at the pinned price snapshot: `$43.35299044832538`

Price snapshot: `tinker-qwen3.5-9b-base-2026-08-09`, with per-million-token
rates of `$0.66` prefill, `$0.132` cached prefill, `$1.995` sample, and
`$1.463` train. The export contains every non-null session identity retained
in the local run records, plus 13 historical sessions identified by Tinker's
run metadata.

After this reconciliation cutoff, retention was removed from all 109
non-smoke artifact-referenced checkpoints. Their recorded aggregate size is
42.776 decimal GB, implying approximately `$4.28` per month of ongoing remote
storage at the frozen `$0.10/GB-month` rate. This prospective storage charge
is not included in the historical `$43.35` total above.

Preserved evidence:

- Raw usage:
  `../DuraSeed-v1-preservation/billing/usage-20260809T000000Z-20260811T160000Z.json`
  — SHA-256
  `b8c98651d71e9acf9a4c366fb4f12c8efacff2f43afb5daa12fc7033fae72ee5`.
- Session-level reconciliation:
  `../DuraSeed-v1-preservation/billing/reconciliation.json`
  — SHA-256
  `efdbdcb0710818ac27317624518251e82e4e508c3df00704e40f59e27ccfead1`.

No console dollar value was invented. The reconciliation uses the service's
raw usage events and the frozen price snapshot; the older per-run
`pending_raw_usage` files are preserved unchanged as historical snapshots.

## Pre-launch ledger snapshot

Against the `$5,000` grant ceiling, the reconciled historical spend above
leaves a ledger balance of `$4,956.64700955167462` before post-cutoff storage
or usage. The committed path to the Pilot-0 freeze is `$445`: `$25` for the
live smoke gate, `$120` for boundary completion/panel freeze, and `$300` for
acquisition calibration. Reserving that path would leave
`$4,511.64700955167462`; the separately capped `$600` Pilot 0 would leave
`$3,911.64700955167462`.

Twenty percent of the current ledger balance is `$991.329401910334924`. Even
holding that conservative amount fixed, the balance after the `$445` path and
Pilot 0 remains `$2,920.317607641339696` above it. This arithmetic does not
replace the confirmatory-launch reserve check, which is recomputed from the
then-uncommitted balance.

## Console refresh before the live smoke

On 13 August 2026 UTC, the signed-in Tinker console reported `$44.34` spent in
the August billing period and a current balance of `$4,955.66`. Its exported
usage CSV contains 32,975,005 tokens and 3.41 GB-months; its rounded line items
sum to `$44.33`, one cent below the authoritative dashboard total because each
line is rounded independently. The preserved export is
`../DuraSeed-v1-preservation/billing/usage-console-2026-08.csv`, SHA-256
`db9bd8166d67d4489af5891d099a4d8b28644f9e8fd1843a7957ad72a073a270`.

The console balance exactly reconciles to the grant ceiling minus its displayed
spend: `$5,000.00 - $44.34 = $4,955.66`. The `$445` path to the Pilot-0 freeze
would leave `$4,510.66`; the separately capped `$600` Pilot 0 would leave
`$3,910.66`. Twenty percent of the current uncommitted balance is `$991.132`,
so even holding that protected amount fixed leaves `$2,919.528` above it after
the pre-Pilot path and Pilot 0. The reserve must still be recomputed at any
later confirmatory launch.
