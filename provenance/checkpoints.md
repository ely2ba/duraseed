# Phase 0b checkpoint preservation

Status: complete with one documented service limitation. TTL removal is
verified in the saved post-change inventory. Every sampler checkpoint was
downloaded and hashed; the service does not expose archives for either full
training-state checkpoint. No training or sampling call was made.

## Preservation scope

The frozen scope is the sorted union of:

- 18 keys in the format candidate TTL audit;
- 7 keys in the final M0-recovery TTL audit (3 overlap the format set, so 4
  additional paths); and
- 87 unique `sampler_checkpoint_path` values in the six MAPS-probe checkpoint
  ledgers.

This produces exactly 109 unique paths. The authoritative path list is
`../DuraSeed-v1-preservation/checkpoint-inventory/checkpoint_paths.txt`, SHA-256
`46ddd461ef6b26ebff3794d52d221a7fb3e1cff294668673cead1ec40c82b329`.
The saved inventories contain the same 109 targets: 107 sampler checkpoints and
2 training/full-state checkpoints, totaling 42,775,627,382 bytes (39.838 GiB).

An exhaustive recursive scan of `runs/` and `artifacts/` contains 120 unique
Tinker paths. The 109-path scope is exactly every non-smoke reference. The 11
excluded paths belong only to the two obsolete engineering-smoke runs; those
runs did not produce a scientific checkpoint or calibration decision.

## TTL audit

| Snapshot | Expiring | Non-expiring | Earliest expiry | Latest expiry |
|---|---:|---:|---|---|
| Before TTL removal | 107 | 2 | 2026-08-16T17:31:37.660287Z | 2026-09-08T19:14:08.336404Z |
| After TTL removal | 0 | 109 | — | — |

The post-change inventory records `expires_at: null` for every target. Key
scientific paths were preserved as follows:

- Raw-audit source M0 sampler: expiry changed from
  `2026-09-08T19:14:06.321660Z` to non-expiring.
  `tinker://7a1bd1bf-2ac0-5cf2-9be1-34fe80139271:train:0/sampler_weights/m0-calibration-20260809T124248Z-05075ce8-candidate-m0-sampler`
- Raw-audit source M0 full state: expiry changed from
  `2026-09-08T19:14:06.873150Z` to non-expiring.
  `tinker://7a1bd1bf-2ac0-5cf2-9be1-34fe80139271:train:0/weights/m0-calibration-20260809T124248Z-05075ce8-candidate-m0-state`
- Selected format source candidate (LR `1e-4`, step 2): expiry changed from
  `2026-09-08T19:14:07.369454Z` to non-expiring.
  `tinker://058bebf2-6569-52d6-bd2c-cdb8e3296080:train:0/sampler_weights/format-training-20260809T171110Z-lr-1e-04-step-2-sampler`
- Selected reconstructed scientific M0 sampler and full state were already
  non-expiring and remain so.
  `tinker://334a5523-f111-5ca2-9607-58ae27cc941d:train:0/sampler_weights/m0-recovery-20260809T191351Z-reconstructed-sampler`
  `tinker://334a5523-f111-5ca2-9607-58ae27cc941d:train:0/weights/m0-recovery-20260809T191351Z-reconstructed-state`
- Frozen MAPS `shortest2_cap2`, LR `3e-4`, step 480 sampler: expiry changed
  from `2026-08-17T15:35:34.030033Z` to non-expiring.
  `tinker://68f62ddf-cb2a-5256-b4cc-2c9dbf74fa1d:train:0/sampler_weights/maps-shortest2-20260810T144606Z-shortest2_cap2-lr-3e-04-step-480`

## Format intervention and scientific M0

- Source model: `Qwen/Qwen3.5-9B-Base`; LoRA rank 32; renderer
  `role_colon`; seed 5; Tinker SDK 0.25.0; Tinker cookbook 0.5.3.
- Recorded resolved-config hash:
  `af1ba6165b89f100a879926dd82c917ef3276066cc180134d7e9d81a2c6ab3f8`.
  The literal `duraseed_pilot_config.yaml` file SHA-256 is
  `814c5293606ed59ab7f3d7e1344a4b7061759733d38fa1829e6fe00af62bb5a0`;
  these are different hash domains, not a claimed byte-hash match.
- Format-training manifest:
  `sha256:36095b3846360f7ad80efb7bb9b1cb38c40f409319de0044f522570ffc637c88`;
  1,024 authenticated task-agnostic format records; TCES/MAPS grammar absent.
- Full-format manifest:
  `sha256:19ad5eee8a9688e07bee246bd38f06ef2b5d6db9ab6539d46004a34086f72c66`;
  256 items. The source format run also records paired-TCES manifest
  `sha256:af6c8237d97aac3dae09c2aada5d47da18e4fa59891e1e53acc2df05c172b253`.
- Format training used LR grid `1e-4, 3e-4, 1e-3`, checkpoints
  `1, 2, 4, 8, 16, 32`, batch size 32, canonical order without shuffling,
  at most one epoch, and weights-only restore with a fresh optimizer. Source
  commit: `c80a0387e012863744f15af3a2f55db5c3b1d502`.
- At step 2 all three LRs passed the 64-item quick screen; the frozen tie-break
  selected the smallest LR, `1e-4`. Its screen metrics were exactly-one-tag
  `63/64 = 0.984375`, exact literal copy `64/64 = 1.0`, and nonempty payload
  `64/64 = 1.0`; task grammar was absent.
- Reused full-format evidence for that source checkpoint recorded canonical
  wrapper compliance `250/256 = 0.9765625`, exact literal copy and nonempty
  payload `253/256 = 0.98828125`, and 4 length stops. Terminations were 251
  stop-sequence, 1 EOS, and 4 malformed.
- The two-update reconstruction completed at LR `1e-4` (4,098 train tokens).
  Its 256-item confirmation recorded canonical wrapper compliance
  `251/256 = 0.98046875`, exact literal copy and nonempty payload
  `255/256 = 0.99609375`, and 2 length stops. Terminations were 252
  stop-sequence, 2 EOS, and 2 malformed.
- The recovery artifact records `scientific_m0_selected: true`, selected step
  2, and the reconstructed sampler/state paths above. Recovery commit:
  `cd04512d7c69a453306f8d8a8075d495243da364`. Pilot 0 remained unstarted.

Factual caveat: the final recovery selection artifact contains no sampled
paired-TCES diagnostic. Its `teacher_forced_logprob_diagnostic` is explicitly
`not_computed`, with the format confirmation treated as authoritative. This
preservation record therefore makes no TCES equivalence, noninferiority, or
uncertainty claim.

## Local export

- Export root:
  `../DuraSeed-v1-preservation/checkpoints`
- Exportable sampler checkpoints: 107
- Full-state checkpoints retained remotely but not exportable: 2
- Verified complete checkpoint directories: 107
- Verified payload files: 321 (107 safetensors, 107 adapter configs, and 107
  `checkpoint_complete` markers), totaling 40,489,713,138 bytes (37.709 GiB)
- Recursive payload manifest:
  `../DuraSeed-v1-preservation/checkpoint-inventory/export_payload_files.sha256`
- SHA-256 of recursive payload manifest:
  `54dc7ca7d00b1d36e40f70aaadbcce03a574db2c41843351d2fdda4fcc6209ae`
- Complete 109-row machine inventory:
  `../DuraSeed-v1-preservation/checkpoint-inventory/checkpoints.tsv`
  — SHA-256
  `894399809ca056373cfda3992ec56a53d9222b952a220639fe1e50f510b93cd1`

The installed Tinker 0.25.0 download command uses the checkpoint-archive
endpoint. That endpoint rejected the selected reconstructed M0 full-state path
with HTTP 400 because it supports only checkpoint IDs beginning with
`sampler_weights/`. The raw-audit source M0 full state has the same `weights/`
type. Both states are verified present and non-expiring, and remain remotely
loadable; the export limitation does not affect their retention. The exact
service errors are preserved with the download logs. No unsupported alternate
export path was invented.

## Evidence sources

- Scope list:
  `../DuraSeed-v1-preservation/checkpoint-inventory/checkpoint_paths.txt`
- Before inventory (SHA-256
  `24f9d794bc6742ab2f1b28a2ac3e16439b34f886447fa0171b69c66cb0852d91`):
  `../DuraSeed-v1-preservation/checkpoint-inventory/checkpoint_inventory_before.json`
- After-removal inventory (SHA-256
  `562fbb0a3e7ac14805f58783f9d70bab8d551fdb1e87875e49359445a0d2d85e`):
  `../DuraSeed-v1-preservation/checkpoint-inventory/checkpoint_inventory_after.json`
- Format candidate TTL audit (SHA-256
  `5ee2d7df4963de73f64cc6b375ec1daabef1adb740b1f024d486e08f43246295`):
  `runs/tinker-calibration/format/format-training-20260809T171110Z/candidate_checkpoint_ttl_audit.json`
- Final recovery TTL audit (SHA-256
  `a4b4959990a44c185c70ff9940ec7306741582b96038ff2f62b9ab2fb8af3047`):
  `runs/tinker-calibration/format-recovery/m0-recovery-20260809T191351Z/ttl_audit.json`
- M0 selection decision (SHA-256
  `06e3adcf8a954d8dd89edb6f641cdb88b182821027679207e8ca62475bcae3fd`):
  `runs/tinker-calibration/format-recovery/m0-recovery-20260809T191351Z/selection_decision.json`
- Format training audit (SHA-256
  `f3cb0bb6ec7a41c3a8851dc9e54cb5f8d1bef1c9ff6e31329403569c5770a211`):
  `runs/tinker-calibration/format/format-training-20260809T171110Z/training_data_audit.json`
- MAPS checkpoint ledgers:
  `runs/tinker-calibration/maps-probe/*/probe_checkpoints.jsonl` (six files,
  87 unique paths).

The before and after inventories were produced by authenticated read-only
checkpoint listings around the authorized TTL changes. Download and retention
operations used only the installed checkpoint/REST boundary; no ServiceClient,
training client, sampling client, generation, or optimizer operation was used.
