# H5 online-vision release validation

Release: the three `h5-online-joint14-init256013-q01q99-clip5-step_040000-20261019.pt` files pinned in `phoenix_release.json`. These checks concern the H5 online-vision checkpoints and PIL preprocessing shipped in this revision.

## Recorded-observation replay

On 2026-10-19, one task and one episode per robot were randomly selected before inference (seed 20261019), with no error-based reselection. All three requested 40000-step checkpoints were verified by SHA256. Full replay used seed 0, BF16, horizon/replan 32, 10 denoising steps, `training_cache` text and no outer autocast. Each chunk sees only its current recorded RGB/state. The last repeated terminal action was excluded.

| Robot | Task / episode | Valid action timestamps | Chunks | Arm MAE (rad / degrees) | Arm RMSE (rad) | Max absolute error (rad) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ARX X5 | `pack_and_pour_fruit` / 41 | 935 | 30 | 0.01117 / 0.6401° | 0.05452 | 0.67900 |
| Piper | `stand_up_bottles` / 15 | 609 | 20 | 0.00943 / 0.5405° | 0.04192 | 0.49682 |
| Piper X | `classify_objects` / 60 | 940 | 30 | 0.00831 / 0.4759° | 0.04009 | 0.82988 |

Metrics cover the 12 arm angles, excluding grippers. Total: **2,484 action timestamps / 80 inference calls**. Predictions were finite and curves were not smoothed. The mean errors are small, but transitions have large local discrepancies: ARX left joint 3 at 15.80 s, Piper right joint 2 at 18.97 s, and Piper X left joint 4 at 3.56 s. These are recorded training episodes, not held-out task success rates or physical closed-loop trials.

## Training/inference alignment

At the start, middle and tail of each episode (nine windows), deterministic image tensors, normalized state, cached text context/masks and current visual latents matched the actual training path exactly, with maximum absolute difference zero. Training-only random color jitter was disabled for this comparison. Shared XPolicyLab image decoding matched the legacy training JPEG decoding on these H5 observations.

At the first window of each robot, actual training-loss/input-builder MoT inputs matched inference with the same controlled noise and timestep. Full ten-step sampling in train/eval mode also matched exactly for all nine windows. This scoped numerical comparison does not imply equality between stochastic training augmentation and inference, or between every possible input.

The deployed visual path is PIL bicubic aspect resize **640×480 → 341×256**, then center crop **256×256**. The single current frame is repeated nine times through both complete WAN VAE and DINO/STVAE branches before taking the first temporal latent. Runtime initialization calls `set_runtime_device`, including BF16 WAN VAE mean/std/scale. Each checkpoint restores 1,650 MoT tensors and three proprio encoder tensors exactly.

## Public adapter checks

The portable package passed **38 current CPU tests**, including all three real H5 checkpoint metadata pairs; 15 historical z-score tests were skipped. Shell syntax, Python compilation, required deploy keys, source hashes, the codec/channel-swap scan and `deploy.py` parity with `demo_policy` passed.

On RTX PRO 5000 72GB Blackwell GPUs, all three public-package loaders restored the pinned checkpoints and generated finite 32×14 actions. The first 32-action chunk per family was byte-for-byte equal to the corresponding audited H5 replay above (maximum difference zero); action acknowledgments and reset passed. These checks used only the portable policy source, the declared Python dependencies, weights and external robot definitions.

The actual `eval.sh` completed **ten debug episodes with plain RGB arrays and ten with encoded observations**, using the official batched WebSocket server/client. Both runs exited zero, reached `[MAIN] eval finished`, and contained no traceback:

```text
[TestEnv] Episode finished
[MAIN] eval finished
```

The reproducible commands are:

```bash
# From the parent of XPolicyLab; PHOENIX_H5_TEST_WEIGHTS is optional for CPU tests.
PHOENIX_H5_TEST_WEIGHTS=/absolute/path/to/downloaded/weights python -m unittest \
  XPolicyLab.policy.Phoenix_MWA.test_phoenix \
  XPolicyLab.policy.Phoenix_MWA.test_joint96 \
  XPolicyLab.policy.Phoenix_MWA.test_joint96_weights \
  XPolicyLab.policy.Phoenix_MWA.test_h5 -v

python -m XPolicyLab.policy.Phoenix_MWA.joint96_assets \
  /absolute/path/to/downloaded/weights --verify-remote

cd XPolicyLab/policy/Phoenix_MWA
WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-mwa base
DEBUG_OBS_ENCODED=1 WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-mwa base
```

The validation host used the existing `fastwam-pretrain` environment for both the policy and client (PyTorch 2.7.1+cu128, torchvision 0.22.1, Pillow 12.0.0, NumPy 1.26.4 and transformers 5.12.1). It set `WAM_CHECKPOINT_PATH` to the prepared weights directory and ran the two commands above with `fastwam-pretrain fastwam-pretrain` as the final environment arguments and GPU 3.

The debug client uses a synthetic instruction, so its tests explicitly use `online_t5`. Default `training_cache` accepts the shipped instruction list. The two modes are not claimed to generate bitwise-identical text features.

## Published weights

All seven files in `phoenix_release.json` are published in `ssuuuuu/phoenix-wam`; their ModelScope metadata matches the pinned SHA256 and byte sizes. Anonymous HTTP Range downloads of the first 64 KiB of each of the three new policy checkpoints returned HTTP 206, the expected full file size, and bytes identical to the local verified checkpoint. This checks public availability and sampled download bytes; it is not a fresh full download of all weights.

## Limits

No simulator success rate or real-robot closed-loop evaluation is claimed. Native data rates are ARX/Piper 30 Hz and Piper X 25 Hz; hardware timing and arm limits remain controller responsibilities. No training code, training trajectory dataset or optimizer state is included in the policy submission. A fresh conda installation has not been rerun for this release.

## H5 filename update (2026-10-19)

The three checkpoint filenames were simplified to `phoenix-wam-<robot>-h5-online-joint14-init256013-q01q99-clip5-step_040000-20261019.pt`. Checkpoint contents, SHA256 hashes and all inference settings are unchanged. The download manifest, README and filename assertions use the new names. After the rename, all seven published policy/shared files passed the manifest SHA256/size checks. The 38 current CPU tests passed again; 15 historical tests were skipped. This naming update does not constitute a new GPU inference evaluation.

## Policy naming update (2026-10-20)

Renamed the public package and deployment identifier to `Phoenix_MWA`; module imports, shell entry points and documented commands use the new name. ModelScope repository IDs, checkpoint filenames/hashes, inference configuration, normalization statistics and text features are unchanged. Under the renamed package, 38 current CPU tests passed with the actual checkpoint metadata; 15 historical tests were skipped. Shell syntax, Python parsing, source-manifest hashes and diff whitespace checks passed. GPU inference and debug episodes were not rerun for this naming-only update; the earlier results above remain historical validation.
