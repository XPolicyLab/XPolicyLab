# CSU-AI-05 Submission Checklist

Checked on: 2026-08-11

This checklist follows the repository-level requirements in `CONTRIBUTING.md`.

## Policy package

- [x] Policy directory is named `policy/CSU-AI-05`.
- [x] Required entry files are present: `__init__.py`, `model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `install.sh`, `setup_eval_policy_server.sh`, `setup_eval_env_client.sh`, and `train.sh`.
- [x] A standard `process_data.sh` entry is present and executable.
- [x] `download_checkpoint.sh` supports the configured Hugging Face dataset folder plus safe archive downloads, keeps `CSU_AI_05.pt` as the canonical weight name, validates the expected layout, and applies the submitted SHA-256 manifest.
- [x] The downloader default and policy-server resolver both use the Git-ignored local path `policy/CSU-AI-05/checkpoints/RoboDojo-CSU-AI-05-arx_x5-joint-0`; runtime weights cannot be staged with the policy code.
- [x] The exact GalaxeaVLA source state used by the running evaluation is vendored under `G05/` and documented as an upstream derivative.
- [x] G0.5 `LICENSE-G0.5`, `NOTICE`, third-party notices, and prominent modification notices are retained.
- [x] No checkpoint, dataset, cache, or other large generated artifact is included in the policy package.

## Model contract

- [x] `model.py` provides the required `Model` interface and supports dynamic import with the hyphenated policy name.
- [x] Image inputs are accepted as decoded `numpy.ndarray` values; `model.py` does not decode image bytes.
- [x] Robot state slicing and action dimensions are obtained from registered robot metadata rather than hardcoded to 14 dimensions.
- [x] The CSU-AI-05 checkpoint bundle resolver selects `checkpoints/*.pt` and does not mistake auxiliary tokenizer files for the main checkpoint.
- [x] The adapter resolves the action tokenizer, dataset statistics, HF processor, and nullable Hydra environment fields from the CSU-AI-05 bundle.
- [x] The evaluated `policy/CSU-AI-05/G05` snapshot is the default implementation root; `G05_ROOT` is retained only for explicit compatibility tests.
- [x] FM action routing, batch size 8, observation offset 0, and the SDPA vision backend match the running CSU-AI-05 evaluation defaults.

## Deployment protocol

- [x] `deploy.py` uses the official WebSocket batch payload contract (`obs=env_idx_list`).
- [x] Unsupported extra keyword arguments are not sent to the batch client.
- [x] `deploy.yml` does not hardcode an action dimension.
- [x] `setup_eval_policy_server.sh` validates robot registration before resolving the action dimension.
- [x] `setup_eval_env_client.sh` supports `XPOLICYLAB_BENCH_ROOT` when the submission checkout is not nested under RoboDojo/RoboTwin, plus explicit client-Python and Conda executable overrides for non-interactive launchers.

## Lifecycle scripts

- [x] `install.sh` installs both XPolicyLab and the evaluated vendored G05 package in editable mode.
- [x] `process_data.sh` validates the expected LeRobot v3 dataset layout and documents that RoboDojo data needs no extra conversion.
- [x] `train.sh` validates Python and dataset inputs and contains no contributor-specific W&B account default.
- [x] All shell entry points pass `bash -n`.
- [x] `model.py` and `deploy.py` pass `python -m py_compile`.

## Documentation

- [x] README contains the contributor, paper, arXiv-status, and original-code header.
- [x] README includes the repository's shared conventions and leaderboard pointer.
- [x] README uses the fixed pointer paragraph and keeps the core sections in the required order: Installation, Data Processing, Training, Evaluation; policy-specific Model Assets, Configuration, and Notes are also included.
- [x] README provides the required command templates plus CSU-AI-05 examples and documents the exact CSU-AI-05 bundle layout, checkpoint path, hashes, configuration, and evaluation commands.
- [x] Model hash and size manifests use portable bundle-relative paths and include the HF processor sidecars.
- [x] README describes the temporal-history deployment behavior that differs from the upstream one-frame path.

## Verification summary

- [x] Static, lightweight contract, license, secret, portability, and package-hygiene checks passed.
- [x] Canonical `ckpt_name=CSU-AI-05` debug closed loop completed with the final `https://huggingface.co/datasets/ShaoRun/vla_ro/tree/main/checkpoints/CSU-AI-05` bundle: 10 episodes x 20 action steps, `[MAIN] eval finished`, exit code 0, zero tracebacks.
- [x] Native `stack_bowls` simulator evaluation completed through `policy/CSU-AI-05/eval.sh` with the same v2 bundle on policy GPU 0 / environment GPU 1: 25 episodes, 19 success / 6 failure / 0 unstable (76%), `[MAIN] eval finished`, exit code 0, zero tracebacks or fatal restarts.
- [x] Detailed test artifacts are backed up outside the submitted policy directory.
- [x] Execute `install.sh` in a clean disposable environment to confirm dependency installation end to end. Static validation has passed.

## External release steps

- [x] README and the downloader use the final CSU-AI-05 checkpoint folder: `https://huggingface.co/datasets/ShaoRun/vla_ro/tree/main/checkpoints/CSU-AI-05`.
- [x] The final local v2 bundle matches the manifest: checkpoint `13,506,086,463` bytes, SHA-256 `7330d644a90a0da313c5aff19ad7987fed6a3d084d789eda0651aea2696182e9`; action tokenizer `169,705,637` bytes, SHA-256 `673b59a7ce63aff22b89d8ccb4f75b21ed906dc32682a342bea11dbb62766388`.
- [x] The final local v2 bundle contains all 12 required files, including all eight `processor/` sidecars, and passes the complete submitted SHA-256 manifest.
- [x] Make the Hugging Face dataset anonymously downloadable; unauthenticated file resolution returned HTTP 401 on 2026-08-11.
- [x] Verify a fresh anonymous download against the complete documented SHA-256 manifest after public access is enabled.
- [x] Stage and inspect the complete `policy/CSU-AI-05` directory.
- [x] Commit to a contributor branch, push it, and open the pull request.
