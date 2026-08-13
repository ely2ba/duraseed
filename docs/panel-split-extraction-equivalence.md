# Panel-split extraction equivalence

This is fixture-level evidence for the narrow extraction accepted in
[`rfc-calibration-panel-split-extraction.md`](rfc-calibration-panel-split-extraction.md).
It is not production panel-split authorization. Production equivalence remains
deferred until the completed three-cohort boundary source exists, so the
calibration launch gate remains closed.

## Comparison

On 2026-08-13 UTC, archived v0 commit
`da709622ff7ee77bcdb1ac5a3a5185244ec3eeae` was exported to a clean package
tree. Its `tinker_teacher_dose._build_panel_split_manifest` and the extracted
v1 `data.panel_split_manifest.build_panel_split_manifest` were executed in
separate Python processes, each with only its own package tree on `PYTHONPATH`.
The archived source file SHA-256 was
`5e461b768f7b00deac02a101a272242bd3fff910e222f8c8c8a2426a6827944e`;
the extracted module SHA-256 was
`f8dc0c60ee4fdae7adccb5862c8c10171f346a0e3e3bd5fed5929409446c5a9a`.

Both processes rebuilt the existing v0 fixture: TCES generator parameters
3 operands, values 2–20, depth 3, 5 AST nodes, and 512 attempts; three broad
families with two items each; the first two sorted families as one-family
panels; root seed 5; panel schedule seeds 17 and 37. The train builder received
the same ordered 14-record forbidden sequence (6 broad, then 8 confirmation
records). The gate builder received those 14 records followed by the four
generated train records. Each builder scanned the same deterministic per-family
stream with the frozen 32× limit.

## Result

| Split | Old rows | New rows | Old bytes | New bytes | Old/new canonical SHA-256 | Full object | Canonical bytes |
|---|---:|---:|---:|---:|---|---|---|
| `a_seed_train` | 4 | 4 | 4,384 | 4,384 | `942ff8bc3e0d909ea2049cf474f5b2745e11edc2034aa4168b8daa1077a171b8` | equal | equal |
| `a_seed_gate` | 4 | 4 | 4,415 | 4,415 | `5d42e5fbf9e1acec338084890211d2e629c2aaed9456b6a622f7a73ce7eab6bc` | equal | equal |

The manifest IDs were respectively
`sha256:2abc457358889349784c95edbc95faa9dc0f9a448761c486dc3bf7e4d2fc4129`
and
`sha256:e1304fb4d8d849c267851f7e953d329e4b2d33591f59470b24f1ded1aa281192`.
The combined canonical comparison was 8,831 bytes with SHA-256
`2f8d1a47d1eb01534cf471101a09db34bd8187b0a2d9f53d41165aa5ab6a32c0`.

Result: zero differing manifest objects, zero differing canonical bytes, and
zero differing records. The extracted unit test imports no archived runner.
