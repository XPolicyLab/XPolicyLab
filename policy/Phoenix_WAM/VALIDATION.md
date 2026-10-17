# Phoenix_WAM validation — 2026-10-17

Release: all three robot-specific `init256013-q01q99-clip5-step_040000-20261017.pt` checkpoints in `phoenix_release.json`. Checks used RTX PRO 5000 72GB Blackwell GPUs, PyTorch 2.7.1+cu128, BF16, seed 0, action horizon/replan 32 and 10 denoising steps. Dummy inference was disabled.

## CUDA image preprocessing

The current policy uses FFmpeg CUDA bicubic resizing and NV12-aligned cropping. For 640x480 RGB, the intermediate size is 342x256 and the crop starts at (42,0), producing 256x256. The previous PIL path produced 341x256 and used a different crop origin/interpolator.

- 35 CPU regressions and one opt-in CUDA image comparison passed; 15 historical z-score tests were skipped. The comparison sends identical raw RGB inputs through an independent reference using the upstream default `scale_cuda` interpolation and its unmodified center-crop expression. Output bytes matched exactly for 640x480, 480x640 and 1280x720, including repeated calls with changing content and distinct camera colors. Mixed-size camera ordering and the unchanged 256x256 bypass were checked too.
- All three latest checkpoints also produced finite 32x14 chunks from original 640x480 HDF5 camera frames, decoded through XPolicyLab's shared helper. Acknowledgment and reset passed. The official WebSocket server/debug client completed both plain and encoded 640x480 observation episodes with the new preprocessing and ARX checkpoint. An initial test-harness import-path error was corrected before the successful protocol run.
- Tests used FFmpeg 4.4.2. Across the three model processes, first-use image processing took 0.44–1.18 seconds to start CUDA/FFmpeg; five subsequent three-view calls each took approximately 4.8–7.3 milliseconds, including normalization and transfer to the model device. These measurements cover preprocessing only, not action inference or controller timing.
- To rerun the comparison on a CUDA host, from the XPolicyLab parent workspace:

  ```bash
  PHOENIX_FFMPEG=/usr/bin/ffmpeg PHOENIX_TEST_CUDA_RGB=1 \
    python -m unittest XPolicyLab.policy.Phoenix_WAM.test_phoenix_rgb -v
  ```

This comparison verifies the resize/crop pipeline for the same RGB input. It does not reproduce the training MP4's compressed pixels or its CPU-fallback resize path. Already-prepared 256x256 replay frames bypass the new conversion; the earlier five-episode curves below do not validate preprocessing of raw 640x480 inputs.

## Public release and portable inference

- ModelScope metadata verified all three policy weights and all four shared weights against the release's exact SHA256 and size. Download tests cover selecting one robot, all robots, offline reuse, and rejecting missing/corrupt files. The large files were reused locally; a fresh full download was not repeated.
- The current CPU suite has 35 passing tests. 15 historical z-score saved-weight tests are skipped; they are not tests of the new release. The CUDA image comparison skips unless explicitly enabled. Run from the XPolicyLab parent workspace:

  ```bash
  python -m unittest XPolicyLab.policy.Phoenix_WAM.test_joint96 \
    XPolicyLab.policy.Phoenix_WAM.test_joint96_weights \
    XPolicyLab.policy.Phoenix_WAM.test_phoenix \
    XPolicyLab.policy.Phoenix_WAM.test_phoenix_rgb -v
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
