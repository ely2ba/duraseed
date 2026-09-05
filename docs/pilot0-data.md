# Pilot-0 raw scientific data

Two downloads cover the completed DuraSeed Pilot-0 pairs: seed 11 and seed 29.
They contain recorded observations, not regenerated samples or a new analysis.
The supervised trace-replay follow-up now running is **not** included.

## Downloads

| Archive | Recorded completions | Verified evaluations | Download size |
| --- | ---: | ---: | ---: |
| [Pair 1 · seed 11](https://github.com/ely2ba/duraseed/releases/download/pilot0-data-v1/pilot0-pair1-data.tar.gz) | 259,712 | 82 | 798.6 MB |
| [Pair 2 · seed 29](https://github.com/ely2ba/duraseed/releases/download/pilot0-data-v1/pilot0-pair2-data.tar.gz) | 259,712 | 82 | 665.2 MB |

Each archive has 378 scientific files, including matching rewards for every
completion and 1,304 training-metric rows. Counts include acquisition rollouts
and evaluations, not just the later-task curves. The
[release page](https://github.com/ely2ba/duraseed/releases/tag/pilot0-data-v1)
keeps these downloads separate from the code repository.

## Contents

Each archive preserves the original repository-relative `runs/pilot0/...`
layout and contains:

- all stored Stage-A rollouts and cadence evaluations, selected-checkpoint
  evaluations, and Stage-B evaluations;
- full prompt and completion text, completion token IDs and log probabilities,
  sample/item IDs, sampling seeds/settings, stop reasons, and verifier rewards;
- evaluation item counts, matching selections, F1/F2 results and F3 profiles;
- scientific training metrics and segment boundaries; and
- the six used training/validation manifests and Stage-A prompt-pool allocation.

Failed completions are retained. No output is shortened, repaired, selected for
presentation, or excluded because of its reward. For every included evaluation,
the generation/reward join and item counts are checked against the stored result.
Native B-S teacher-training files were not logged: those deterministic targets
can be reconstructed from the included manifests, prompt allocation, and
repository code. Stage-B teacher programs are in the training manifest. This
release does not mislabel newly reconstructed targets as recorded model outputs.

## Privacy and provenance

These are **public projections**, not byte-identical copies of private runs.
Prompt/completion text, tokens, log probabilities, rewards, and scientific
numbers are unchanged. Provider checkpoint addresses become stable anonymous
`checkpoint:pairN:...` references; they preserve equality within each pair but
are not usable remote paths. Billing/authorization records, account identifiers,
remote-call journals, backend timing counters, local machine paths, sealed test
sets, and model/adapter weights are excluded.

`data-index.json` lists every exported scientific file, its source and public
byte hashes, and JSONL row counts. References to generation/reward hashes in
summaries and profiles point to the exported bytes. Manifest IDs and remaining
scientific source fingerprints retain their original meaning; not every
historical calibration source is bundled. Internal whole-run audit hashes are
omitted rather than presented as checksums of these reduced archives.

Tar metadata contains no local owner name, UID, or timestamps. Both archives
include the export script and repository Apache-2.0 license; no model weights
are redistributed.

## Use with the repository

Extract the downloads, then copy their `runs/` directories into the root of a
DuraSeed checkout. They populate only the two completed Pilot run directories.
Use a fresh checkout rather than overwriting a private evidence collection.
The existing readout and uncertainty scripts can then read the same relative
locations as the publication reports. Adapter-geometry recomputation still
requires the separately retained tensors; their derived geometry tables are
already in the repository.

Generation and reward rows join on `sample_id`; items join on `task_id`.
`item_counts` in each evaluation's `result.json` supplies successes and trials
per task. `method`, `checkpoint_stage`, `training_step`, `panel_role`, and the
containing directory locate each observation. Stage-B update-0 arithmetic
evaluations live in `selected-pre-b/`; later arithmetic monitoring is named
`a-retention/`.

To rebuild this public export from the original local evidence, from the
repository root:

```sh
python tools/export_pilot0_data.py --output artifacts/pilot0-data-export
```

The output location must be new. Export is offline and does not instantiate a
remote client, change source files, renew checkpoints, or touch live runs.
