# Adapter validation

Status: **Technical qualification passed: 15 CPU tests, both real-checkpoint debug transports, four valid native simulator episodes, and all eight exact replay/backend checks. [PR #234](https://github.com/XPolicyLab/XPolicyLab/pull/234) is open. The checkpoint is public and ungated; no evaluator credential handoff is required. The technical report and training/pruning/export code are planned for November 20, 2026. Full official benchmark verification remains pending.**

| Check | Evidence and scope |
|---|---|
| Static/template audit | 50 Python files compiled, five shell scripts checked, clean diff, exact PR-template headings |
| CPU regression | 15 tests passed: compact architecture metadata, corrupt artifacts, blending, observation timing, reset, backend settings, lossless array integrity and comparison failures |
| Real video tensors | Strict CPU loading passed; 12,468,409,664 compact video parameters |
| Artifact integrity | All six bundle files and nine auxiliary assets matched pinned sizes/hashes |
| Fresh policy installation | Python 3.10.21, PyTorch 2.11.0+cu130; dependency consistency check passed |
| Source preservation | 30 numerical modules and 16 policy functions match the frozen reference after namespace normalization; factory and prompt match |
| Simulator preflight | Official doctor: 12 passes, zero failures; Isaac and conda checks intentionally skipped at this non-GPU stage |
| Plain and encoded debug | Passed with real checkpoint: job 53609, exit 0, both evaluation-finished markers |
| Simulator smoke | 4/4 valid, original layouts 0/1, seed 1: Insert tubes 2/2 successes, 100/100 partial; Store laptop/headphones 1/2 successes, 60/100 partial |
| Numerical action comparison | Passed: all eight checks, zero recorded differences; job 53612 completed with exit 0 |

XPolicyLab base: `6baccd57ff958570d0db1588370289ee36c8bb70`.
Simulator validation checkout: `f57e1633ffed59761bc8e568cfe7530c4b33381c`.

Two real recordings of 209 observations / 14 action chunks each were replayed in A/B/A order, giving 42 chunks per implementation. Reference-versus-package, original-capture-versus-replay, reset and backend checks all passed with zero recorded differences. Evidence: `evidence/gpu-validation.json` and `evidence/action-comparison.json`. Equality is established for these fixtures on PyTorch 2.11.0+cu130 and NVIDIA RTX PRO 6000 Blackwell, not every input or backend.

The original Store laptop/headphones layout 1 failed with 0.2/1 partial and is retained. Debug completion is an interface check; four simulator smoke episodes do not establish full-suite performance. Historical research scores are separately documented in EVALUATION_RESULTS.md.

## Reviewer workflow audit (November 7, 2026)

A fresh regular-file snapshot of submitted commit
`a08a40fb01664cfe5411a02d723f9e0af7233683` passed all 15 CPU contract tests with
zero skips after the published metadata downloader fetched and verified the
four pinned JSON files. This check used only the dedicated reviewer credential
and an empty Hugging Face cache; no owner credential fallback was used.

Authenticated metadata access also passed for both checkpoint tensors and all
nine public dependency files, matching pinned sizes and the appropriate remote
LFS SHA-256 or Git blob identity. Large tensors were not downloaded again in this
audit. CPU checks used Python 3.12.7 / PyTorch 2.11.0+cpu with existing test
dependencies; they do not replace the earlier fresh Python 3.10.21 / CUDA 13.0
installation and GPU qualification. Inference code and settings are unchanged.

That audit used private authentication; the checkpoint is now public, as verified
below. A complete fresh-machine simulator/assets installation, different
GPU types, and a full five-suite packaged-adapter run have not been qualified by
this audit.

## Public checkpoint access (November 7, 2026)

The checkpoint and all required encoder/tokenizer assets are publicly accessible
without authentication. The published metadata downloader passed in an isolated
empty cache with the submission-pinned `huggingface-hub==0.36.0` and no token.
All four checkpoint JSON files matched their SHA-256 checksums; all 15 asset URLs
passed anonymous ranged downloads and matched pinned remote identities and sizes.
The checkpoint revision, hashes and inference settings are unchanged. Evidence:
[public-checkpoint-access.json](evidence/public-checkpoint-access.json).

## Reproduce the transport checks

Use the policy README's debug command with the actual checkpoint. Repeat it with `DEBUG_OBS_ENCODED=1`. Both runs must finish normally and print the evaluation-finished marker.

## Optional lossless capture

`BAGEL_VALIDATION_TRACE=/absolute/path/to/new-capture-directory` enables `validation_server.py` through the normal server launcher. The wrapper uses the official server and records already-decoded observations, inference inputs, history fingerprints, raw predictions, blending and commands. It does not modify actions, seeds, task termination or scoring.

Capture is limited to the first 209 observations per episode. The default server path does not import or enable the recorder. Array blobs are local test artifacts and are excluded from the PR.

`validation_replay.py` replays two fixtures in A/B/A order. `validation_compare.py` checks exact equality of reference versus packaged outputs, capture versus replay, history and resets, and verifies matched GPU/backend settings. The original frozen research implementation is needed for the reference comparison and is not redistributed in this inference package.

All planned qualification stages completed. Failed attempts remain preserved in the outer verification packet; full camera fixtures and videos remain on the validation machine and are excluded from the PR.

## Packaging repair

The first debug attempt (53541) failed before opening the server because serialized processor targets still referred to the original `vpp2` namespace and the processor modules were missing. Its dependent simulator job 53542 was cancelled before execution. The repair vendors the evaluated processor/transform modules unchanged except import namespace, and recursively maps Hydra targets into `bagel_cp1000`. Four new CPU cases test the real processor metadata and legacy target formats while rejecting research-package imports. Failed evidence is preserved in the outer review packet.

## Shared simulator installation

The local qualification environment reuses an existing shared Isaac Sim installation. Attempt 53610 stopped at its noninteractive startup prompt; attempt 53611 then stopped on read-only shared cache permissions. Neither reached a scored episode. Attempt 53612 restores the same noninteractive flag and user-owned portable Kit directory used by the frozen evaluation runner. A recorded one-line change to the validation checkout's launch script permits the portable-root argument; task, physics, layout and scoring code remain unchanged. This launch-only patch is in the outer verification packet and is not part of the policy PR. See [Kit portable configuration](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/configuring.html).
