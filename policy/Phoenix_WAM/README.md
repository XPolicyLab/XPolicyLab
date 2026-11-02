# Phoenix_WAM

## Current weights: run 20261101_02 (2026-11-02 release)

This package selects the three **30000-step** checkpoints trained at **5e-5**,
initialized from step **256013**. It retains head-camera cover and
`phoenix_stall_seed=-1` random resampling during stalls. Install the matching
adapter and weights together; older local 40000-step files do not satisfy this
release's filename, size, SHA256 and step checks.

The active ModelScope repository is **ssuuuuu/phoenix-mwa**. The obsolete
`ssuuuuu/phoenix-wam` address is not used. The deployment folder is still
`policy/Phoenix_WAM`; it has not been renamed to Phoenix_MWA.

## Head-camera stall recovery with resampling (2026-11-02)

This release uses the 2026-11-01 paired-cache 30000-step weights and retains
normalization, normal-motion sampler seed, 32-step horizon/replan and 10 denoising steps.
It implements the temporary **cover** intervention from the supplied
2026-10-27 stall analysis: sustained measured inactivity triggers two chunks
with a black head-camera image, then one chunk with the real image. This cycle
repeats only while inactivity persists. Arm or gripper movement restores the
real image at the next prediction boundary; already returned actions are not
cancelled. Wrist images, measured joint state and task instruction are retained.

During **every stalled prediction**, `phoenix_stall_seed=-1` also requests a
fresh random seed, including the real-image chunk of the two-cover/one-real
cycle. The frozen sampler passes seeds to `torch.Generator.manual_seed`, where
literal `-1` is deterministic. The adapter therefore resolves the `-1` sentinel
to a fresh nonnegative 63-bit seed before each stalled call, avoiding the normal
seed and the previous draw. It logs the actual seed for replay. Each session's
normal seed is restored in `finally`, including after an inference exception;
normal motion and reset retain the original seed and batch environments do not
share an override. Neither the frozen sampler nor global RNG state is modified.

Only decoded `uint8` head-camera pixels are replaced with zero **before** the
existing resize/normalization/visual encoder. The original observation is never
modified, including read-only decoded arrays. This is not `camera_is_pad`, latent
zeroing or a forced joint movement. Each batch environment has
its own measured-state history and cover cycle; `reset()` clears both.

Detection uses actual joint feedback from acknowledged control steps, not
predicted actions or the number of inference requests. The window must contain
all configured steps. Every arm channel must remain within 0.02 rad and every
gripper within 0.05 native units. Compared with the report's endpoint formula,
this patch checks the range over the whole window, so a robot that moves out
and back is not mistaken for stationary. Uploading duplicate observations at
the same executed step does not advance the window.

The patched `deploy.yml` enables recovery. Missing recovery keys in an external
old config default to disabled. Configuration:

| Key | Patched default | Meaning |
| --- | --- | --- |
| `phoenix_stall_fix` | `cover` | `none` restores the original inference path |
| `phoenix_stall_seconds` | `3.0` | Nominal duration used to derive control steps |
| `phoenix_stall_steps` | `null` | `ceil(seconds * action_hz)`; an explicit positive integer overrides duration, `0` disables recovery |
| `phoenix_stall_cover_chunks` | `2` | Black chunks followed by one real-image chunk |
| `phoenix_stall_seed` | `-1` | Fresh random seed for each stalled chunk; `null` retains the previous cover-only behavior; a nonnegative integer uses that fixed seed during stalls |
| `phoenix_stall_arm_threshold` | `0.02` | Maximum measured arm range in radians |
| `phoenix_stall_gripper_threshold` | `0.05` | Maximum measured gripper range in native units |

`action_hz` must reflect the **actual hardware client's control frequency**.
It does not change the client's speed. With the existing native fallback,
ARX/Piper use 90 steps at 30 Hz and Piper X uses 75 steps at 25 Hz. Detection
occurs at a replan boundary, so the first possible trigger with replan 32 is
step 96. If the client executes at 10 Hz, set `WAM_ACTION_HZ=10` to obtain the
report's 30-step window. These are control-time durations, not a wall-clock
timeout including inference latency. Two covered chunks contain 64 actions:
about 2.13 s at 30 Hz, 2.56 s at 25 Hz or 6.4 s at 10 Hz.

The existing server script accepts `WAM_ACTION_HZ`, `WAM_STALL_FIX`,
`WAM_STALL_SECONDS`, `WAM_STALL_STEPS`, `WAM_STALL_COVER_CHUNKS` and
`WAM_STALL_SEED` as overrides. `WAM_STALL_SEED=null` selects cover-only behavior;
`WAM_STALL_FIX=none` disables both interventions.
For example, prefix the usual server command with `WAM_ACTION_HZ=10` for a
10 Hz controller, or `WAM_STALL_FIX=none` for the original behavior. Restart
the policy server after changing code or configuration; this patch does not
restart an active robot process.

`audit_dir/stall_recovery.jsonl` records each successful prediction's effective
thresholds, measured movement, environment index and cover decision. The
`stalled` counter counts inactive prediction chunks and `changed` counts covered
chunks. `get_stall_rollout()` exposes the current episode statistics. Episode
records also contain `last_requested_seed` (`-1` for random resampling),
`last_effective_seed` (the actual sampler seed), and `reseeded` (successfully
sampled chunks with a new random seed). A real-image chunk can therefore have
`last_covered=false` and `last_requested_seed=-1` while still stalled. Episode
summaries are written and printed at `reset()`; logging failure does not discard
a successfully queued action chunk.

CPU regression command, from the parent of `XPolicyLab`:

```bash
python -m unittest -v XPolicyLab.policy.Phoenix_WAM.test_stall_recovery
```

The source report describes different robots, checkpoints and execution rates;
its success numbers do not validate this patch. See the accompanying validation
record for tests actually run on this version. The intervention changes model
conditioning and does not guarantee a safe or successful recovery from every
stall. This adapter continues to rely on the existing hardware client's limits.

**Contributor:** Phoenix_WAM contributors | **Paper:** Not provided | **arXiv:** Not provided | **Original code:** [InternW0-Delta](https://github.com/InternRobotics/InternW0-Delta)

Phoenix_WAM is an **evaluation-only** adapter for RoboDojo real-robot ARX X5, Piper and Piper X, using absolute dual-arm joint actions (`action_type=joint`). The current **H5 no-crop paired-cache release** provides one **30000-step posttraining checkpoint per robot, initialized from step 256013**. `joint96_source/` contains the inference subset of the upstream runtime and encoders; `phoenix_source/` contains inference configuration, per-robot q01/q99 statistics and fixed task-instruction features. No training checkout or trajectory dataset is required.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, CUDA and BF16 support are required. The installer pins PyTorch 2.7.1 / torchvision 0.22.1 with CUDA 12.8. Image preprocessing uses PIL/torchvision; a CUDA-enabled FFmpeg build is not required. From this directory:

```bash
bash install.sh phoenix-wam
conda activate phoenix-wam
bash download_checkpoint.sh checkpoints/phoenix-mwa-h5-nocrop-init256013-step30000-20261101
```

The downloader fetches the three H5 policy weights and shared DINO, STVAE, WAN VAE and T5 weights from [ModelScope](https://www.modelscope.cn/models/ssuuuuu/phoenix-mwa/files). For one robot plus shared weights, append `--robot-type piper` (or `arx_x5` / `piper_x`). It downloads individual weight files only. Configuration, tokenizer, statistics and the instruction feature table ship with this adapter. The matching adapter archive is `releases/20261101_02/Phoenix_WAM_run20261101_02.zip` in the same repository; install its `policy/Phoenix_WAM/` directory into the XPolicyLab checkout before downloading weights. No training-data or optimizer archive is needed.

Use the new weights directory when upgrading. Center-crop, legacy visual-cache and z-score policy checkpoints are incompatible and rejected. Existing files with incorrect hashes are rejected rather than silently replaced. Shared weights with matching hashes may be reused.

After downloading, verify public metadata and then the full local checkpoint contract
from the parent of the XPolicyLab checkout:

```bash
python -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  /absolute/path/to/weights --verify-remote
python -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  /absolute/path/to/weights --robot-type arx_x5 --expected-step 30000
```

Repeat the second command with `piper` and `piper_x`. Downloading verifies full
file SHA256; preflight verifies metadata, robot identity, normalization and step.
A successful preflight alone is not GPU inference or physical robot evaluation.

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
  phoenix-mwa-h5-nocrop-init256013-step30000-20261101 arx_x5 joint 0 0 0 phoenix-wam base
DEBUG_OBS_ENCODED=1 WAM_TEXT_MODE=online_t5 EVAL_ENV_TYPE=debug bash eval.sh RoboDojo cover_blocks \
  phoenix-mwa-h5-nocrop-init256013-step30000-20261101 arx_x5 joint 0 0 0 phoenix-wam base
```

For real hardware, RoboDojo supplies the environment client and controller. Choose one matching policy server command:

```bash
bash setup_eval_policy_server.sh RoboDojo cover_blocks phoenix-mwa-h5-nocrop-init256013-step30000-20261101 \
  arx_x5 joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo fill_pen_holder phoenix-mwa-h5-nocrop-init256013-step30000-20261101 \
  piper joint 0 0 phoenix-wam 9000 0.0.0.0
bash setup_eval_policy_server.sh RoboDojo cap_pen phoenix-mwa-h5-nocrop-init256013-step30000-20261101 \
  piper_x joint 0 0 phoenix-wam 9000 0.0.0.0
```

`env_cfg_type` selects the checkpoint, statistics and soft-prompt slot. The shared checkpoint resolver uses `ckpt_name`; `WAM_CHECKPOINT_PATH` may override it with an absolute weights directory. See the root README for client setup and split-machine deployment.

Default `text_mode=training_cache` preserves the exact training instruction features used in the replay validation. Instructions must match the text in [instructions.json](phoenix_source/instructions.json); unknown strings fail explicitly. For arbitrary instructions, set `WAM_TEXT_MODE=online_t5` before launching the server (or `text_mode: online_t5` in `deploy.yml`). Online T5 uses the pinned model/tokenizer and the same prompt template, but is not claimed to produce bitwise-identical cached features.

## Model Assets

Files are in ModelScope repository `ssuuuuu/phoenix-mwa`, revision `master`. `phoenix_release.json` pins every filename, size and SHA256. The `run20261101_02` suffix identifies the selected training run. All three source `slot_6.pt` files contain actual step **30000**, initialized from step **256013**, with learning rate **5e-5**. These are different weights from the 2026-10-19 40000-step and 2026-10-21 edge25 releases.

| `env_cfg_type` | H5 no-crop checkpoint | Slot | Native data rate |
| --- | --- | --- | --- |
| `arx_x5` | `phoenix-mwa-arx-x5-h5-trim-cached-resize256-nocrop-joint14-init256013-q01q99-clip5-step_030000-run20261101_02.pt` | 2 | 30 Hz |
| `piper` | `phoenix-mwa-piper-h5-trim-cached-resize256-nocrop-joint14-init256013-q01q99-clip5-step_030000-run20261101_02.pt` | 4 | 30 Hz |
| `piper_x` | `phoenix-mwa-piper-x-h5-trim-cached-resize256-nocrop-joint14-init256013-q01q99-clip5-step_030000-run20261101_02.pt` | 15 | 25 Hz |

The release manifest pins unique `h5-trim-cached-resize256-nocrop` filenames for run `20261101_02`, full SHA256 values and exact sizes. Existing historical ModelScope artifacts remain separate. Both download entry points read this manifest, including the shared DINO/STVAE names under the current `phoenix-mwa` repository. The adapter directory and `policy_name` remain **Phoenix_WAM** for compatibility with this checkout.

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

See [VALIDATION.md](VALIDATION.md) for measured checks and limitations. The earlier 35000-step audit found exact RGB/state/text preparation but a residual between stored training visual features and online encoding. This remains an unresolved validation item; a new checkpoint does not by itself resolve it. Historical numbers are kept in [VALIDATION_20261019.md](VALIDATION_20261019.md) and are not new-weight results. Recorded training-episode replay is not a held-out evaluation or a real-robot success rate. No physical closed-loop evaluation is claimed. This adapter outputs joint targets and does not implement hardware-specific arm position/velocity limits.
