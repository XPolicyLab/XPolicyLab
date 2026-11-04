# Edge25 online release validation

This revision downloads and loads three robot-specific **40000-step** checkpoints from `run20261021_edge25_01`, initialized at step 256013. The exported model architecture, processor and normalization are unchanged; the saved training configuration records online H5 visuals, edge25 temporal trimming and training-only shared color jitter. Inference keeps full-view bicubic resize256, repeat9 encoding, replan 32 and 10 denoising steps.

The loader pins the complete per-family training contracts in `phoenix_source/edge25_training_contracts.json`, including their trim/augmentation source identities. It checks the exported configuration, checkpoint step, family, normalization and exact restored tensors. It never instantiates the private training dataloader or applies training augmentation.

## Current 40000-step checks (2026-10-21)

- **43 CPU tests passed**, with 15 historical tests skipped (58 total), including all three actual checkpoint/configuration pairs and rejection of altered family, step, normalization, geometry, trim and augmentation metadata.
- **All three GPU policies loaded and generated finite 32×14 joint-action chunks** from recorded observations, using cached training text. Each loader verified all 1650 `mot` tensors and 3 `proprio_encoder` tensors against the checkpoint. Action acknowledgments and reset passed; replan=32 and denoising steps=10.
- **Standard `eval.sh` passed 10 plain-RGB and 10 encoded-observation debug episodes** for ARX X5 with `online_t5`. Both exited zero, reached `[MAIN] eval finished` and had no traceback. These exercise the official WebSocket server/client and server-side decoding; they are synthetic protocol checks, not physical robot success evaluation.
- **Concurrent tokenizer initialization regression passed.** The earlier 35000-step update fixed a partial-copy checksum race. This release retains verified temporary copies and atomic replacement and reruns the regression.
- Shell syntax, Python parsing, whitespace and checked-in source/configuration hashes passed. All seven public ModelScope policy/shared files matched the pinned SHA256 and sizes. The three uploaded policy files previously passed anonymous first/last 64-KiB range checks; no fresh complete download was repeated.

Machine-readable evidence: [edge25_40000_20261021.json](validation_results/edge25_40000_20261021.json). Tests used the existing `fastwam-pretrain` environment on NVIDIA RTX PRO 5000 72GB Blackwell GPUs (driver 580.126.09). No new environment installation, simulator success evaluation, physical rollout or strict numerical training/inference equivalence is claimed.

Reproduce CPU and public download checks from the parent of `XPolicyLab`:

```bash
PHOENIX_H5_TEST_WEIGHTS=/absolute/path/to/weights python -m unittest \
  XPolicyLab.policy.Phoenix_MWA.test_phoenix \
  XPolicyLab.policy.Phoenix_MWA.test_joint96 \
  XPolicyLab.policy.Phoenix_MWA.test_joint96_weights \
  XPolicyLab.policy.Phoenix_MWA.test_h5 \
  XPolicyLab.policy.Phoenix_MWA.test_resize_concat -v
python -m XPolicyLab.policy.Phoenix_MWA.joint96_assets /absolute/path/to/weights --verify-remote
```

For current download and debug commands, see [README.md](README.md).

The previous 35000-step smoke-test summary is retained in `validation_results/edge25_20261021.json` for history; it is not evidence for the current weights. This release supersedes only the three matching edge25 35000-step policy files in ModelScope; unrelated releases and shared encoders remain unchanged.

## Historical paired-cache release (2026-10-19/20)

**Everything in this section concerns the previous paired-cache release, not the current edge25 weights. Historical commands must be used with the corresponding historical code and weights.**

The previous revision selected three robot-specific **40000-step** checkpoints from the 2026-10-19 no-crop, temporal-trim, paired-cache training runs, initialized from step 256013. Filenames and hashes are pinned in `phoenix_release.json`. The earlier center-crop release and its validation results do not apply to these weights.

### Training/inference comparison

The source deployment was audited on 2026-10-19 using the preceding 35000-step checkpoint. Six windows covered all three robot families, including temporal-trim boundaries. Image tensors and cached text context/masks matched the training reader exactly; normalized states also matched at four ARX windows. Inverse-normalized labels matched next-retained-state targets within 1.2e-7 (float32 rounding). The actual cache generator produced exactly the same RGB tensors at the four ARX windows.

Both complete visual branches encode nine repeated current frames before the first latent is selected. That first latent matched fresh encoding of the real nine-frame clip at all six windows. However, **persisted training-cache features did not exactly match online encoding**: maximum first-latent absolute differences were 0.046875 for WAN and 0.9873046875 for DINO/STVAE. At the first ARX window, the training-loss input builder and inference had equal action tokens, context, masks and timing inputs, but visual tokens had a relative L2 discrepancy of 1.71% (maximum absolute difference 1.259765625).

Controlled tests using the actual cache encoder loader reproduced this residual. Changing TF32 or WAN VAE statistics precision did not remove it. Repeating the probe with the training launcher's CUDA/conda library paths reproduced the same residual. The cause remains unresolved; strict numerical training/cache/inference equivalence is **not claimed**. The 40000-step checkpoints do not by themselves resolve this encoder discrepancy.

A controlled diagnostic on the **40000-step ARX** model substituted the stored current latent for the online current latent at four recorded windows, keeping RGB/state/text, seed and sampling settings fixed. Across each 32-action chunk's 12 arm joints, mean absolute action differences ranged from **0.000330 to 0.000906 rad**, with maximum **0.008831 rad** across the four windows. This measures the downstream effect on those inputs only; it neither identifies the cause nor establishes equivalence across tasks or robots.

### Earlier recorded-episode replay (35000 step)

An ARX X5 `store_in_safe` episode (36) was chosen before inference using seed 20261019 among episodes with reviewed temporal cuts. It had 782 original frames, 17 removed waiting frames and 764 valid next-state action timestamps. Replay used BF16, seed 0, cached training text, 10 denoising steps and horizon/replan 32 (24 chunks). No future observations were supplied and curves were not smoothed.

Across 12 arm joints, excluding grippers: MAE **0.007423 rad**, RMSE **0.024002 rad**, P95 **0.026045 rad**, maximum **0.267308 rad**. These numbers concern **35000 step only**, not this PR's 40000-step release. Recorded training-episode replay is not held-out task success or physical closed-loop evaluation.

### Public adapter validation

The portable package passed **42 current tests**, including all three real 40000-step checkpoint/configuration pairs; 15 historical z-score tests were skipped. Geometry tests verify retained image edges, RGB channel order and three separate camera views. Negative tests reject altered cache/trim identities, crop settings, camera order, family, step and normalization.

All three actual GPU loaders restored the checkpoints and produced finite 32×14 actions. For each family, the public adapter's first chunk was exactly equal to the original server deployment using the same checkpoint/input (maximum difference **zero**). Action acknowledgments and reset passed. This is packaging parity, distinct from the unresolved training-cache discrepancy above.

The actual `eval.sh` passed **10 plain-RGB debug episodes and 10 encoded-observation debug episodes** through the official batched WebSocket server/client. Both runs exited zero, reached `[MAIN] eval finished` and had no traceback. These used the existing `fastwam-pretrain` environment on RTX PRO 5000 72GB Blackwell GPUs, with PyTorch 2.7.1+cu128 and transformers 5.12.1. `WAM_CHECKPOINT_PATH` selected the verified weights directory; both environment arguments were `fastwam-pretrain`, using GPU 3. Shell syntax, Python compilation, whitespace, source/configuration hashes and `deploy.py` parity with the reference adapter passed.

Commands from the parent of `XPolicyLab`:

```bash
PHOENIX_H5_TEST_WEIGHTS=/absolute/path/to/weights python -m unittest \
  XPolicyLab.policy.Phoenix_MWA.test_phoenix \
  XPolicyLab.policy.Phoenix_MWA.test_joint96 \
  XPolicyLab.policy.Phoenix_MWA.test_joint96_weights \
  XPolicyLab.policy.Phoenix_MWA.test_h5 \
  XPolicyLab.policy.Phoenix_MWA.test_resize_concat -v

python -m XPolicyLab.policy.Phoenix_MWA.joint96_assets \
  /absolute/path/to/weights --verify-remote

cd XPolicyLab/policy/Phoenix_MWA
WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-mwa base
DEBUG_OBS_ENCODED=1 WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-mwa base
```

The synthetic debug instruction requires `online_t5`; the default `training_cache` mode accepts the shipped instruction list. The two text modes are not claimed to generate bitwise-identical features.

### Published weights

All three 40000-step no-crop policy checkpoints and all four shared encoder weights are published in `ssuuuuu/phoenix-mwa`. The public repository's SHA256 and byte size matched every entry in `phoenix_release.json`. Each new policy checkpoint is 12,045,488,406 bytes.

Anonymous HTTP range downloads of the first 64 KiB of each new checkpoint returned HTTP 206, reported the expected full file size, and matched the corresponding local bytes exactly. This checks public access and sampled content; it is not a fresh full-file download. The download script verifies the full SHA256 after downloading. The updated ModelScope README was also read back anonymously and matched the published content.

### Limits

No simulator success rate or physical robot evaluation is claimed. Native data rates are ARX/Piper 30 Hz and Piper X 25 Hz; controller timing and arm limits remain hardware-client responsibilities. No training entry point, trajectory dataset or optimizer state is included. A fresh conda installation has not been rerun for this release.

### Canonical filename cleanup (2026-10-19)

The three checkpoint entries now use `h5-trim-cached-resize256-nocrop` and `run20261019_01`. Their SHA256 hashes and sizes are unchanged. Three duplicate no-crop aliases were removed from ModelScope. The three separate center-crop checkpoints were restored with their original filenames and SHA256 hashes and remain available for PR #165. After cleanup, the downloader verified all seven retained policy/shared files against the release manifest; the ModelScope README passed anonymous readback. The 42 current CPU tests passed again, with 15 historical tests skipped. This filename-only change did not rerun GPU inference or physical evaluation.

### Policy naming update (2026-10-20)

Renamed the public package and deployment identifier to `Phoenix_MWA`; module imports, shell entry points and documented commands use the new name. ModelScope repository IDs, checkpoint filenames/hashes, inference configuration, normalization statistics and text features are unchanged. Under the renamed package, 42 current CPU tests passed with the actual checkpoint metadata; 15 historical tests were skipped. Shell syntax, Python parsing, source-manifest hashes and diff whitespace checks passed. GPU inference and debug episodes were not rerun for this naming-only update; the earlier results above remain historical validation.
