# Phoenix_WAM

**Contributor:** Phoenix_WAM contributors | **Paper:** Not provided | **arXiv:** Not provided | **Original code:** [InternW0-Delta](https://github.com/InternRobotics/InternW0-Delta)

Phoenix_WAM is an **evaluation-only** adapter for RoboDojo real-robot ARX X5, Piper and Piper X, using absolute dual-arm joint actions (`action_type=joint`). Each robot has its own **40000-step posttraining checkpoint initialized from step 256013**, with q01/q99 normalization. `joint96_source/` contains the inference subset of the upstream runtime and encoders; `phoenix_source/` contains the new inference configuration and per-robot statistics. No training code or training data is required to run this adapter.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, CUDA and BF16 support are required. From this directory:

```bash
bash install.sh phoenix-wam
conda activate phoenix-wam
bash download_checkpoint.sh checkpoints/phoenix-wam-init256013-step40000
```

The download script fetches the three latest policy weights plus the shared DINO, STVAE, WAN VAE and T5 weights from [ModelScope](https://www.modelscope.cn/models/ssuuuuu/phoenix-wam/files). To download one robot and the shared dependencies, append `--robot-type piper` (or `arx_x5` / `piper_x`). Model configuration, tokenizer and normalization statistics are included in Git; no source/resource archive is downloaded.

Old installations must rerun the download command. Old z-score checkpoints cannot be used with the new q01/q99 statistics. Existing files with incorrect hashes are rejected rather than silently replaced.

## Data Processing

This submission does not provide data conversion. Posttraining used RoboDojo real Joint14 labels and a verified visual cache built from a prepared **LeRobot v3.0** MP4 export, including the corrected RGB videos. That export uses `action.joint`, `observation.joint_state` and `observation.images.cam_high/cam_left_wrist/cam_right_wrist`; it is a supplementary Joint14 export, not the official EEF-only converter output. Neither the training export nor the visual cache is required for inference. `process_data.sh` reports that this stage is unsupported.

## Training

本版本仅支持评测。`train.sh` reports that training is unsupported. A public training release date has not been announced. The bundled configurations are inference metadata, not a training entry point.

## Evaluation

The parent workspace must contain RoboDojo's official `env_cfg/` robot definitions. Standard same-machine invocation:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline protocol check: no simulator or robot motion.
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks phoenix-wam-init256013-step40000 \
  arx_x5 joint 0 0 0 phoenix-wam base
# Exercise encoded observations through the shared server decoder too.
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks phoenix-wam-init256013-step40000 \
  arx_x5 joint 0 0 0 phoenix-wam base
```

For real hardware, RoboDojo supplies the environment client and controller. Start the matching policy server, for example:

```bash
bash setup_eval_policy_server.sh RoboDojo cover_blocks phoenix-wam-init256013-step40000 \
  arx_x5 joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo fill_pen_holder phoenix-wam-init256013-step40000 \
  piper joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo cap_pen phoenix-wam-init256013-step40000 \
  piper_x joint 0 0 phoenix-wam 9000 0.0.0.0
```

Choose one command per server. `env_cfg_type` selects the checkpoint, statistics and soft-prompt slot. The shared checkpoint resolver uses `ckpt_name`; `WAM_CHECKPOINT_PATH` may override it with an absolute weights directory. See the root README for client setup and split-machine deployment.

## Model Assets

All files are in ModelScope repository `ssuuuuu/phoenix-wam`, revision `master`; `phoenix_release.json` pins every filename, size and SHA256.

| `env_cfg_type` | Latest checkpoint | Slot | Native data rate |
| --- | --- | --- | --- |
| `arx_x5` | `phoenix-wam-arx-x5-openwam-joint14-init256013-q01q99-clip5-step_040000-20261017.pt` | 2 | 30 Hz |
| `piper` | `phoenix-wam-piper-openwam-joint14-init256013-q01q99-clip5-step_040000-20261017.pt` | 4 | 30 Hz |
| `piper_x` | `phoenix-wam-piper-x-openwam-joint14-init256013-q01q99-clip5-step_040000-20261017.pt` | 15 | 25 Hz |

Keep the downloaded filenames and nested `base_models/` layout. The loader verifies source metadata, checkpoint identity, per-family normalization, cached-visual provenance, frozen encoder paths and all restored trainable tensors. It does not import a training checkout or read server-specific paths.

The trained action and state layout is left arm 6 + left gripper 1 + right arm 6 + right gripper 1, mapped to canonical80 active dimensions `[0,1,2,3,4,5,16,40,41,42,43,44,45,56]`. Public robot action dimensions still come from XPolicyLab's shared helpers.

## Configuration

`deploy.yml` defaults to `runtime_mode=joint96_real_phoenix`, BF16, action horizon 32, replan 32, 10 denoising steps and CPU random noise. Seed comes from the standard launch arguments. `action_hz=null` records the native data rate above; actual hardware command timing is controlled by the client.

- Inputs are RGB uint8 arrays supplied by XPolicyLab. The policy does not decode or swap channels. Three views are ordered main, left wrist, right wrist; the image path uses aspect-preserving bicubic resize and center crop to 256x256, then float32 normalization to [-1,1]. Prepared 256x256 training MP4 frames preserve their geometry; live-image processing is not claimed to be bitwise identical to compressed MP4 frames.
- `current_visual_mode=repeat9_full_joint96`: repeat the current frame nine times, run both complete WAN and DINO/STVAE branches, concatenate Joint96 latents, then retain the first temporal latent. No future observation is required.
- Action/state normalization uses each family's q01/q99 statistics, forward clipping to [-5,5] and no inverse normalization clip. Piper/Piper X input grippers are checked against [-0.05,1.05] and clipped to [0,1]; ARX inputs retain their raw values. Output grippers are clipped to [0,1] with an audit record; arm angles are not globally clipped.
- Public inference encodes the instruction with the pinned T5 weights and tokenizer, using the training prompt template and 128-key padding convention. It does not download or require the training text cache.
- Text CFG is 1, negative prompt is empty, tiling is disabled, and the saved video/action noise schedules are used. No sigma-shift override, memory or VLM conditioning is supported by this release.
- `allow_dummy_policy=false` is the default. `audit_dir` receives loading, first-action and gripper records. `WAM_AUDIT_DIR`, `WAM_EXPECTED_STEP`, `WAM_CHECKPOINT_PATH` and `WAM_CKPT_SETTING` are optional server-script overrides.

## Notes

The validation record is in [VALIDATION.md](VALIDATION.md). Recorded-observation training-episode replay is not a held-out evaluation or a real-robot success rate. This release has not been verified in a physical closed loop, and it does not implement hardware-specific arm position/velocity limits.
