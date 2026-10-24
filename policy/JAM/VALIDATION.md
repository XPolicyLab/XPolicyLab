# JAM Piper X adapter validation

Code-review scope: `policy/JAM` on XPolicyLab `e0f05c13de1671ba83b6e31ee6bd8a09c3b539b7`, audited on 2026-10-17. Model source comes from [JAM@0eaec826](https://github.com/y0urOy/JAM/tree/0eaec826faba36d6f2e447a3c7cc6d6db1dd1a71). The original inference implementation, fixed statistics and supplied checkpoints are unchanged.

## Current CPU/static checks

| Check | Result |
|---|---|
| Required adapter files / standard deploy keys | Passed: all 11 files and 12 reference keys |
| Model contract, checkpoint identity and RGB boundary | Passed in the contract suite |
| Downloads | Passed: 12 anonymous commit-pinned sources checked by size and content SHA256; actual resumable downloader fetched the 6,623-byte tokenizer metadata file and verified its final path/hash |
| Shell/Python syntax | Passed: 8 shell scripts and 72 Python files |
| CPU contracts | Passed: 8 tests, including policy selection, fixed inference overrides, Hugging Face directory layout and no-I/O dry-run |
| Current WebSocket transport | Passed: 2 CPU test cases, each exercising both manifest identities, raw/encoded RGB, batch order and reset; neural inference is an explicit deterministic test double |
| CPU doctor | Passed: current imports, Hydra configs, processor, relocation and normalization; Python 3.10.20, torch 2.7.1+cpu, CUDA not initialized; full asset hashing not run by doctor |
| Native install/data/train command construction | Passed: 4 dry runs; no data conversion or training started |
| Runtime preservation | Passed: 73 runtime files compared; 49 byte-identical, 24 whitespace-only cleanups. All executable ASTs match; 20 full ASTs match and 4 differ only in docstring whitespace |

The CPU transport tests do not load weights or estimate policy performance. They contain no production inference bypass. Tests use an isolated Python 3.10 environment with `requirements/test.txt`; `pip check` passed. The CPU doctor reused the earlier isolated CPU-torch validation environment, distinct from the newly installed torch-free transport-test venv.

Run from the repository root after preparing the small workspace templates:

```bash
python3.10 policy/JAM/setup_workspace.py
# Use the CPU Python environment installed from policy/JAM/requirements/test.txt:
python policy/JAM/tests/test_contracts.py
python policy/JAM/tests/test_transport.py
```

### Decode/encode audit

`model.py` accepts decoded RGB arrays and never decodes or swaps channels. XPolicyLab source HDF5 image buffers reach the shared official `decode_image_bit` through its converter. No `cv2.imdecode`, `cv2.imencode`, `COLOR_BGR2RGB` or `COLOR_RGB2BGR` occurs in the submitted Python files. All other mechanical decode hits were inspected:

- `runtime/scripts/robodojo_piperx_prepare.py` and vendored LeRobot PIL readers consume the official converter's RGB PNG/video or native LeRobot storage, not raw XPolicyLab trajectory JPEG buffers.
- `runtime/src/fastwam/runtime.py` opens an optional standalone user image, outside this adapter's observation path.
- `runtime/src/fastwam/datasets/foresight_cache.py` uses `np.frombuffer` on cached bf16 feature tensors, not images.

`deploy.py` matches the current reference adapter. `eval.sh` retains the standard ten arguments, server/wait/client/cleanup flow and completion marker; its documented client is simulator-free debug only. Native data/train wrappers use the named options shown in the README.

## Historical actual-weight checks

[Sanitized historical evidence](validation/historical_transport.json) retains the original report hashes and selected results. On **2026-10-16**, candidate `piperx-local-20261016-b2cf99f5a4e2` at XPolicyLab `f6e309e8ece20bfa33c0e5d7b05a5f3287522757` passed actual-checkpoint checks on one GH200 (job `3569244`, exit 0):

| Fixed model | Raw | Encoded | Batch | Max difference from saved actions | Local inference chunks |
|---|---|---|---|---:|---:|
| JAM-Wan | passed | passed | passed | 0.0 | 4 |
| FastWAM-PiperX | passed | passed | passed | 0.0 | 4 |

Reference: `cap_pen_ep008_frame00000`. The five shared framework files covering RGB conversion, checkpoint resolution, WebSocket server/client and debug client are byte-identical between that candidate and the current upstream base. Runtime executable AST is preserved, with the format-only differences recorded in `PROVENANCE.json`. This strengthens provenance; it is **not** a new GPU execution on this PR or a fresh platform installation.

## Not run / pending

- Full actual-checkpoint `EVAL_ENV_TYPE=debug bash eval.sh ...` on the PR head or platform GPU: not run in this CPU-only contribution round.
- Fresh CUDA/x86/RTX installation, platform hardware capacity/latency and remote robot client: unverified.
- Portable data conversion/retraining, simulation and real robot trials: not run here.
- Staged six-task/two-model acceptance, repetitions/seeds/success definitions, reset/stop protocol and schedule: await platform agreement.
- Verified overall leaderboard acceptance and public result/video scope or release timing: not promised. Piper X-only coverage does not satisfy the full three-embodiment requirement.

**Verdict: ready for PR code review.** Physical evaluation remains pending: trials **0**, success rate **null**. The earlier 240 held-out action chunks per model are not robot trials.
