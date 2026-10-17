# Phoenix_WAM validation — 2026-10-17

Release: all three robot-specific `init256013-q01q99-clip5-step_040000-20261017.pt` checkpoints in `phoenix_release.json`. Checks used RTX PRO 5000 72GB Blackwell GPUs, PyTorch 2.7.1+cu128, BF16, seed 0, action horizon/replan 32 and 10 denoising steps. Dummy inference was disabled.

## Public release and portable inference

- ModelScope metadata verified all three policy weights and all four shared weights against the release's exact SHA256 and size. Download tests cover selecting one robot, all robots, offline reuse, and rejecting missing/corrupt files. The large files were reused locally; a fresh full download was not repeated.
- 31 current CPU tests passed. 15 historical z-score saved-weight tests were skipped; they are not tests of the new release. Run from the XPolicyLab parent workspace:

  ```bash
  python -m unittest XPolicyLab.policy.Phoenix_WAM.test_joint96 \
    XPolicyLab.policy.Phoenix_WAM.test_joint96_weights \
    XPolicyLab.policy.Phoenix_WAM.test_phoenix -v
  ```

- `bash -n`, Python compilation, required file/deploy-key checks, the RGB codec/channel-swap scan and `git diff --check` passed. `deploy.py` is identical to `demo_policy/deploy.py`.
- Each public checkpoint loaded through the default release-directory path, using only the checked-in runtime/config/statistics and downloaded weight files. All 1,653 trainable tensors (1,650 MoT + 3 proprio encoder tensors) matched the checkpoint after restore. Each robot predicted a finite 32x14 action chunk from a real recorded observation, acknowledged all actions and reset successfully.
- The actual current-frame visual wrapper recorded one input frame, nine encoded frames, full latent shape `[1,96,3,16,48]` and first-latent shape `[1,96,1,16,48]`.
- The public path uses online T5. Compared with the training text cache on one real instruction/window per robot, text embedding mean absolute differences were approximately 0.001166 (ARX), 0.000294 (Piper), 0.000406 (Piper X). Maximum native Joint14 action differences were 0.004533, 0.004589 and 0.004202 respectively. These include gripper dimensions and are not arm-only radian metrics. The two text execution paths are not claimed to be bitwise identical.
- The official WebSocket `PolicyServer` and batched `debug_env_client.py` completed both plain-array and encoded-image episodes with the new ARX checkpoint. Both client logs ended with:

  ```text
  [TestEnv] Action Step: 19 / 20 (step_limit)
  [TestEnv] Check Episode End: True
  [TestEnv] Check Episode End: True
  [TestEnv] Episode finished
  ```

  This exercised the shared server decoder and real model output. The combined `eval.sh` convenience launcher was not rerun for this update.

## Recorded-observation replay

The same q01/q99 backend with the training text cache completed five full training episodes, sampled before inference with seed 20261017: two uniformly sampled tasks per ARX/Piper, one for Piper X, then one random episode per selected task. There was no error-based reselection.

| Robot | Task | Episode | Valid action timestamps |
| --- | --- | --- | --- |
| ARX X5 | make_bread | 28 | 1081 |
| ARX X5 | cover_blocks | 23 | 1128 |
| Piper | stand_up_bottles | 62 | 765 |
| Piper | fill_pen_holder | 67 | 943 |
| Piper X | pack_objects_into_backpack | 66 | 663 |

Total: 145 prediction calls and 4,580 valid action timestamps. Raw HDF5 and LeRobot numeric labels were exactly equal for the selected episodes. At each chunk start, only the current recorded RGB/state and instruction were passed to the policy. Predictions were compared with `action[t] = state[t+1]`; the final repeated terminal action was excluded. Full time-series plots were produced without smoothing.

This is training-pool recorded-observation inference, not held-out task-success evaluation or a physical closed loop. The five-episode curves used cached text; the separate public-path checks above used online T5. Simulator success rates, hardware control timing, device joint limits and fresh-environment installation remain unverified.
