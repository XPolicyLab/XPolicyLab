# Phoenix_WAM

**Contributor:** Phoenix_WAM contributors | **Paper:** Not provided | **arXiv:** Not provided | **Original code:** [InternW0-Delta](https://github.com/InternRobotics/InternW0-Delta)

Phoenix_WAM is an **evaluation-only** adapter for RoboDojo real-robot ARX X5, Piper and Piper X, using absolute dual-arm joint actions (`action_type=joint`). The current **H5 no-crop paired-cache release** provides one **40000-step posttraining checkpoint per robot, initialized from step 256013**. `joint96_source/` contains the inference subset of the upstream runtime and encoders; `phoenix_source/` contains inference configuration, per-robot q01/q99 statistics and fixed task-instruction features. No training checkout or trajectory dataset is required.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, CUDA and BF16 support are required. The installer pins PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.8. Image preprocessing uses PIL/torchvision; a CUDA-enabled FFmpeg build is not required. From this directory:

```bash
bash install.sh phoenix-wam
conda activate phoenix-wam
bash download_checkpoint.sh checkpoints/phoenix-wam-h5-nocrop-init256013-step40000-20261019
```

The downloader fetches the three H5 policy weights and shared DINO, STVAE, WAN VAE and T5 weights from [ModelScope](https://www.modelscope.cn/models/ssuuuuu/phoenix-wam/files). For one robot plus shared weights, append `--robot-type piper` (or `arx_x5` / `piper_x`). It downloads individual weight files only. Configuration, tokenizer, statistics and the small instruction feature table are included in Git; no resource archive is needed.

Use the new weights directory when upgrading. Center-crop, legacy visual-cache and z-score policy checkpoints are incompatible and rejected. Existing files with incorrect hashes are rejected rather than silently replaced. Shared weights with matching hashes may be reused.

## Data Processing

This submission does not provide data conversion. These checkpoints were posttrained using paired visual features cached from raw RoboDojo HDF5 RGB observations and native Joint14 labels. Each complete camera image is directly resized to 256×256; no spatial crop is applied. A reviewed temporal trim removes waiting frames before caching. Action targets are the next retained state, and statistics remain the original per-family q01/q99 statistics. No LeRobot MP4 export is used. The training dataset and cache are not needed for inference. `process_data.sh` reports that this stage is unsupported.

Offline readers of RoboDojo trajectories must use `decode_image_bit` from `XPolicyLab.utils.process_data`. The shared server already decodes live observations into RGB arrays; the policy performs no image decoding or channel swaps.

## Training

本版本仅支持评测。`train.sh` reports that training is unsupported. A public training release date has not been announced. The bundled configuration and instruction features are inference assets, not a training entry point or episode data.

## Evaluation

The parent workspace must contain RoboDojo's official `env_cfg/` robot definitions. Standard same-machine invocation:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Offline protocol check; online_t5 accepts the debug client's synthetic instruction.
WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-wam base
DEBUG_OBS_ENCODED=1 WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-wam-h5-nocrop-init256013-step40000-20261019 arx_x5 joint 0 0 0 phoenix-wam base
```

For real hardware, RoboDojo supplies the environment client and controller. Choose one matching policy server command:

```bash
bash setup_eval_policy_server.sh RoboDojo cover_blocks phoenix-wam-h5-nocrop-init256013-step40000-20261019 \
  arx_x5 joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo fill_pen_holder phoenix-wam-h5-nocrop-init256013-step40000-20261019 \
  piper joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo cap_pen phoenix-wam-h5-nocrop-init256013-step40000-20261019 \
  piper_x joint 0 0 phoenix-wam 9000 0.0.0.0
```

`env_cfg_type` selects the checkpoint, statistics and soft-prompt slot. The shared checkpoint resolver uses `ckpt_name`; `WAM_CHECKPOINT_PATH` may override it with an absolute weights directory. See the root README for client setup and split-machine deployment.

Default `text_mode=training_cache` preserves the exact training instruction features used in the replay validation. Instructions must match the text in [instructions.json](phoenix_source/instructions.json); unknown strings fail explicitly. For arbitrary instructions, set `WAM_TEXT_MODE=online_t5` before launching the server (or `text_mode: online_t5` in `deploy.yml`). Online T5 uses the pinned model/tokenizer and the same prompt template, but is not claimed to produce bitwise-identical cached features.

## Model Assets

Files are in ModelScope repository `ssuuuuu/phoenix-wam`, revision `master`. `phoenix_release.json` pins every filename, size and SHA256. The `20261019` suffix identifies this publication; the no-crop training runs began on 2026-10-19. These filenames distinguish this release from the earlier center-crop H5 weights.

| `env_cfg_type` | H5 no-crop checkpoint | Slot | Native data rate |
| --- | --- | --- | --- |
| `arx_x5` | `phoenix-wam-arx-x5-h5-nocrop-trim-cached-joint14-init256013-q01q99-clip5-step_040000-20261019.pt` | 2 | 30 Hz |
| `piper` | `phoenix-wam-piper-h5-nocrop-trim-cached-joint14-init256013-q01q99-clip5-step_040000-20261019.pt` | 4 | 30 Hz |
| `piper_x` | `phoenix-wam-piper-x-h5-nocrop-trim-cached-joint14-init256013-q01q99-clip5-step_040000-20261019.pt` | 15 | 25 Hz |

Each policy checkpoint is approximately 12.05 GB. Keep the downloaded filenames and nested `base_models/` layout. The loader validates the H5 training contract, selected robot, normalization, encoder resources and all restored trainable tensors. It does not import a training checkout or resolve server-specific paths. External `training_config` and `stats_package` overrides are rejected.

The native layout is left arm 6 + left gripper 1 + right arm 6 + right gripper 1, mapped to canonical80 active dimensions `[0,1,2,3,4,5,16,40,41,42,43,44,45,56]`. Public robot dimensions come from XPolicyLab's shared helpers.

## Configuration

`deploy.yml` defaults to `runtime_mode=joint96_real_phoenix`, BF16, action horizon 32, replan 32, 10 denoising steps and CPU random noise. Seed comes from the standard launch arguments. `action_hz=null` records the native data rate above; actual command timing is controlled by the hardware client.

- Inputs are RGB uint8 arrays, ordered head, left wrist, right wrist. Each **full view is resized directly to 256×256** using PIL/torchvision bicubic interpolation with antialiasing, then normalized to float32 `[-1,1]`. A 640×480 input becomes **256×256**, changing aspect ratio and preserving the image edges. Views use a 256×768 horizontal transport layout and are split before visual encoding; their latents are joined horizontally. There is no spatial crop or image augmentation.
- `current_visual_mode=repeat9_full_joint96`: repeat the current frame nine times, run both complete WAN VAE and DINO/STVAE branches, concatenate Joint96 latents, then retain the first temporal latent. No future observation is required.
- Runtime initialization calls `model.set_runtime_device(device)` and uses BF16 WAN VAE statistics, as in the source deployment. There is no additional outer autocast. The paired-cache encoder used FP32 VAE statistics; controlled tests found that this difference does not explain the measured DINO/STVAE residual. Strict numerical equality with persisted training cache is **not established**; see VALIDATION.md.
- Action/state normalization uses each family's q01/q99 statistics, forward clipping to `[-5,5]` and no inverse normalization clip. Piper/Piper X input grippers are checked against `[-0.05,1.05]` and clipped to `[0,1]`; ARX inputs retain raw values. Output grippers are clipped to `[0,1]` with an audit record; arm angles are not globally clipped.
- `text_mode` is `training_cache` or `online_t5`. Both use the training prompt template and 128-key padding convention. T5 weights are downloaded with the shared weights so either mode is available.
- Text CFG is 1, negative prompt is empty, tiling is disabled, and the saved video/action noise schedules are used. No sigma-shift override, memory or VLM conditioning is supported. Dummy actions and legacy runtime modes are rejected.
- `audit_dir` receives loading, first-action and gripper records. `WAM_AUDIT_DIR`, `WAM_EXPECTED_STEP`, `WAM_CHECKPOINT_PATH`, `WAM_CKPT_SETTING` and `WAM_TEXT_MODE` are optional server-script overrides. `timing_enabled` controls session timing; `default_instruction` is only used when the observation contains no instruction and must exist in the cache when using `training_cache`.

The client calls `reset` at episode start, uploads an observation, requests a chunk, then uploads one observation after each executed action. All 32 actions must be acknowledged before the next chunk; configure a smaller `replan_steps` explicitly if the client executes fewer. Batch sessions are isolated by `env_idx`.

## Notes

See [VALIDATION.md](VALIDATION.md) for measured checks and limitations. The earlier 35000-step audit found exact RGB/state/text preparation but a residual between stored training visual features and online encoding. This remains an unresolved validation item; a 40000-step checkpoint does not by itself resolve it. Recorded training-episode replay is not a held-out evaluation or a real-robot success rate. No physical closed-loop evaluation is claimed. This adapter outputs joint targets and does not implement hardware-specific arm position/velocity limits.
