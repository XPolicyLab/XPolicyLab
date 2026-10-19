# No-crop paired-cache release validation

This revision selects three robot-specific **40000-step** checkpoints from the 2026-10-19 no-crop, temporal-trim, paired-cache training runs, initialized from step 256013. Filenames and hashes are pinned in `phoenix_release.json`. The earlier center-crop release and its validation results do not apply to these weights.

## Training/inference comparison

The source deployment was audited on 2026-10-19 using the preceding 35000-step checkpoint. Six windows covered all three robot families, including temporal-trim boundaries. Image tensors and cached text context/masks matched the training reader exactly; normalized states also matched at four ARX windows. Inverse-normalized labels matched next-retained-state targets within 1.2e-7 (float32 rounding). The actual cache generator produced exactly the same RGB tensors at the four ARX windows.

Both complete visual branches encode nine repeated current frames before the first latent is selected. That first latent matched fresh encoding of the real nine-frame clip at all six windows. However, **persisted training-cache features did not exactly match online encoding**: maximum first-latent absolute differences were 0.046875 for WAN and 0.9873046875 for DINO/STVAE. At the first ARX window, the training-loss input builder and inference had equal action tokens, context, masks and timing inputs, but visual tokens had a relative L2 discrepancy of 1.71% (maximum absolute difference 1.259765625).

Controlled tests using the actual cache encoder loader reproduced this residual. Changing TF32 or WAN VAE statistics precision did not remove it. The cause remains unresolved; strict numerical training/cache/inference equivalence is **not claimed**. The 40000-step checkpoints do not by themselves resolve this encoder discrepancy.

## Earlier recorded-episode replay (35000 step)

An ARX X5 `store_in_safe` episode (36) was chosen before inference using seed 20261019 among episodes with reviewed temporal cuts. It had 782 original frames, 17 removed waiting frames and 764 valid next-state action timestamps. Replay used BF16, seed 0, cached training text, 10 denoising steps and horizon/replan 32 (24 chunks). No future observations were supplied and curves were not smoothed.

Across 12 arm joints, excluding grippers: MAE **0.007423 rad**, RMSE **0.024002 rad**, P95 **0.026045 rad**, maximum **0.267308 rad**. These numbers concern **35000 step only**, not this PR's 40000-step release. Recorded training-episode replay is not held-out task success or physical closed-loop evaluation.

## Public adapter validation

Release-specific test results are recorded below after execution. Commands from the parent of `XPolicyLab`:

```bash
PHOENIX_H5_TEST_WEIGHTS=/absolute/path/to/weights python -m unittest \
  XPolicyLab.policy.Phoenix_WAM.test_phoenix \
  XPolicyLab.policy.Phoenix_WAM.test_joint96 \
  XPolicyLab.policy.Phoenix_WAM.test_joint96_weights \
  XPolicyLab.policy.Phoenix_WAM.test_h5 \
  XPolicyLab.policy.Phoenix_WAM.test_resize_concat -v

python -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  /absolute/path/to/weights --verify-remote

cd XPolicyLab/policy/Phoenix_WAM
WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-wam base
DEBUG_OBS_ENCODED=1 WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-wam base
```

The synthetic debug instruction requires `online_t5`; the default `training_cache` mode accepts the shipped instruction list. The two text modes are not claimed to generate bitwise-identical features.

## Limits

No simulator success rate or physical robot evaluation is claimed. Native data rates are ARX/Piper 30 Hz and Piper X 25 Hz; controller timing and arm limits remain hardware-client responsibilities. No training entry point, trajectory dataset or optimizer state is included. A fresh conda installation has not been rerun for this release.
