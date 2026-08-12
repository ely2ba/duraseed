# Calibration extraction equivalence

Verified locally on 2026-08-12 with no Tinker, network, training, or sampling
access. The v0 and v1 module graphs ran in separate Python processes. Every
reported digest is SHA-256 over the repository's canonical JSON bytes.

## Immutable decision reducers

- `teacher_dose.py` is byte-identical to v0: SHA-256
  `c30ad32dc6efa7b1b8c86b6e69dbfbcbf49db0ff3530d22a0fadc7fae5e16619`.
- `stage_a_calibration.py` is byte-identical to v0: SHA-256
  `6055c54649dd2cd7a17385a04445e5fa5f8c9ea44a3d6f4adf93a7611d2f3b7f`.
- The 12 teacher-dose tests were split at a test boundary to satisfy the v1
  file cap. All 12 test-function ASTs are identical to v0. Only the shared raw
  fixture replaces archived runner-private boundary builders with their
  already-equivalence-verified v1 reducers. The old and new fixture objects
  are identical: 199,268 bytes,
  `sha256:c79de3d7a092ba4d8747d2d3033bd12e4e46ed2308db2e182da90d14a743a934`,
  with 64 targeted and 8+8 paired sentinel generations.
- The exact Stage-A test file is immutable carry-over: SHA-256
  `95700b2a9000c65c9da9c99279a9cca1a4b687e84d0d671c9d673bf4a53f78e7`.
- Dose decisions, both learning-rate decisions, and the duration decision are
  identical: 265,839 bytes,
  `sha256:42db58553425ce160d9c6ef6e903e6296b69ed6c133e33c560bd85235f27cdff`.
  The outcomes are verification-required then dose 2 selected; B-S `1e-4`,
  B-G `1e-5`; duration frozen at 50 updates.

## Teacher candidate and crossed allocation

The same deterministic 240-record fixture was reconstructed in separate v0
and v1 processes. V0's tokenizer constructor alone was replaced by the same
local role-colon measurement callback used by v1 (3 prompt and 2 target
tokens); source records, candidate derivation, dose selection, matching, and
result composition remained authoritative.

- Canonical candidate: 210,128 bytes,
  `sha256:cbfc6a1f361cbf71f4e62376c56c93e725d0d742a371f88438384cc623bafe28`.
- Dose-2 crossed allocations: 217,511 bytes,
  `sha256:fc314e39d1995618b1e06a77fcbd98adb9e9a5669c13284318020ba772bf00ac`.
  Both orientations selected with 13 matcher states each.
- A fixture where manifest item order differs from task-ID order confirms v1
  retains v0's exact dose-builder ordering and per-manifest tie break.
- Adversarial derived token ledgers are ignored and remeasured in both paths.

## Random-universe freezer

The deterministic harness supplied identical authenticated object inputs and
candidate streams to v0 and v1. It replaced only path/run authentication and
the expensive record-generation/matching leaves; each repository's actual
prefix loop, manifest creation, seven-manifest leakage audit, terminal-state
handling, and complete result composition ran unchanged. Full canonical
`TeacherAllocationFreezeResult` objects matched:

| Case | Prefix | Random rows | Bytes | Canonical SHA-256 |
|---|---:|---:|---:|---|
| first joint selection | 2 | 32 | 26,559 | `45b34a6de5e0d4a12c7530b37226197ca651f057ab4504e7a5b3269b1ef61fb9` |
| panel-A search exhausted | 1 | 16 | 13,945 | `db14b04072b13dc06a02fb3c32a7fb94ea1c2fe7cfef17ae5df55ca7c4deed15` |
| later panel-A exhaustion (stale-B regression) | 2 | 32 | 26,265 | `2ab493e6ec1c797cdcc311aa4946da5bd6c80af98479e78bfcf804507cd45eb3` |
| panel-B search exhausted | 1 | 16 | 14,030 | `01b151afc25f7ab792b1fd59551510ceb3936ad5cd93f36aec55b86a64458ee1` |
| `a_seed_train` cap | 257 | 4,096 | 3,186,212 | `92721e94d2e7bba30f1396d3fc896ffc0d7c42bffd6d264e5e5ff8d3179e17bc` |
| zero accepted at ceiling | 3 | 0 | 1,571 | `fe06150d08b1f863ca62a77d576d1e25e86dcceb6f4a562393ee11e5bb630999` |
| trailing exclusions at ceiling | 3 | 16 | 14,066 | `c98dd563bc730264a1e6b8c5ab871fb517a0d02501da58aad8b331eb2a0f2e76` |

Result: zero divergence. The v1 freezer owns only object validation and pure
local reduction. Completed-run/preflight authentication and the role-colon
token measurer remain Phase-5 runner/runtime responsibilities.
