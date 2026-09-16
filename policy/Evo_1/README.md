# Evo-1

**Contributor:** MINT-SJTU | **Paper:** Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment (CVPR 2026) | **arXiv:** [2511.04555](https://arxiv.org/abs/2511.04555) | **Original code:** [MINT-SJTU/Evo-1, evo1-flash](https://github.com/MINT-SJTU/Evo-1/tree/evo1-flash)

This adapter connects Evo-1 to the current RoboTwin / XPolicyLab evaluation stack, with joint-space actions on `arx_x5` or `aloha_agilex`. It reuses the upstream network and normalizer, and provides data preparation and training entry points. Installation downloads the upstream source into ignored `upstream/`, pinned to `5fd14b015013c4fd0aacf5f8f48f868ca9b870a2`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use a separate CUDA policy environment. FlashAttention is part of the upstream evaluation recipe.

```bash
conda create -n evo1-xpl python=3.10 -y
conda activate evo1-xpl
cd XPolicyLab/policy/Evo_1
bash install.sh
```

The installer uses PyTorch 2.5.1, torchvision 0.20.1, transformers 4.39.0 and FlashAttention 2.7.4.post1; CUDA development tools are needed when no compatible FlashAttention wheel exists. `EVO1_MAX_JOBS=4` controls build parallelism. The adapter-owned requirements use `einops` (the upstream requirements spell it `einop`) and obtain shared runtime dependencies through XPolicyLab.

To reuse an existing **pinned** source checkout, export `EVO1_SOURCE_DIR=/absolute/path/to/Evo-1` in both installation and server shells, or set `upstream_dir` in `deploy.yml`. `bash install.sh --source-only` checks/prepares source without installing dependencies. Existing checkouts at another revision are rejected rather than changed.

The backbone `OpenGVLab/InternVL3-1B` is also required when constructing the network. Its assets are downloaded by the upstream loader; for offline use, cache them first or set `vlm_path` to a local complete snapshot.

## Data Processing

Training consumes **LeRobot v2.1**, with the keys produced by the official `scripts/transform_lerobot_v21_format.py` converter:

| Feature | Layout |
| --- | --- |
| `observation.state`, `action` | 14 values: left arm, left gripper, right arm, right gripper |
| `observation.images.cam_high` | RGB head camera video |
| `observation.images.cam_left_wrist` | RGB left wrist camera video |
| `observation.images.cam_right_wrist` | RGB right wrist camera video |

Convert **one canonical task per dataset** to retain the released policy's per-task normalization. Use prepared official exports, or run the official converter from the XPolicyLab root inside the benchmark workspace:

```bash
python scripts/transform_lerobot_v21_format.py 'RoboTwin.adjust_bottle.arx_x5' \
  --repo_id robotwin_adjust_bottle
```

Then prepare a named dataset collection; each trailing argument maps a task name to its dataset root:

```bash
cd policy/Evo_1
bash process_data.sh RoboTwin clean50 arx_x5 joint \
  adjust_bottle=/absolute/path/to/robotwin_adjust_bottle \
  click_bell=/absolute/path/to/robotwin_click_bell
```

`process_data.sh` writes `processed_data/RoboTwin-clean50-arx_x5-joint/config.yaml`, maps tasks to `aloha_joint/robotwin_<task>`, and runs the upstream statistics calculation. It copies metadata and links data/videos so the upstream writer cannot overwrite source metadata. No adapter-owned image decoding or conversion is performed. Each output collection must have a new name; existing collections are preserved. Caches pad state/actions to 24 dimensions and retain absolute joint targets. Upstream-native datasets with different camera names must be exported to the official keys or supplied through a separately prepared Evo-1 YAML via `EVO1_DATA_CONFIG`.

## Training

The wrapper calls the unchanged upstream Accelerate trainer, sets the requested seed, and writes a new XPolicyLab run directory. It uses full-model fine-tuning defaults; pass native trainer arguments to select the training recipe. These defaults are an integration example, **not a reproduction claim for the released RoboTwin checkpoint**.

```bash
# Fine-tune from an upstream STANDARD checkpoint.pt directory into a NEW run.
bash train.sh RoboTwin clean50 arx_x5 joint 0 0 \
  --resume --resume_pretrain --resume_path /absolute/path/to/standard_checkpoint/step_2 \
  --lr 6e-5 --max_steps 20000 --batch_size 16
```

The launcher uses the active `python` for Accelerate. Training requires a local filesystem supporting symlinks for prepared metadata/data links. Logs stay local by default: the upstream trainer uses offline W&B, and `SWANLAB_MODE=disabled` is the wrapper default (set it explicitly to opt into another mode).

GPU IDs may be comma-separated (e.g. `0,1,2,3`). The default launcher uses Accelerate with BF16 and standard PyTorch/DDP checkpointing. Without `--resume --resume_pretrain`, training starts from the InternVL backbone and a newly initialized action head. The upstream training/data documentation covers two-stage pretraining and alternative recipes.

Checkpoints land under `checkpoints/RoboTwin-clean50-arx_x5-joint-0/step_<N>/` (plus `step_best` / `step_final`). Pass the **exact step directory** as `ckpt_name` for evaluation. The inference loader accepts the upstream standard `checkpoint.pt` / `model_state_dict` format and released DeepSpeed `mp_rank_00_model_states.pt` / `module` format, with strict loading. With the default non-DeepSpeed training launcher, `--resume_path` must contain a standard `checkpoint.pt`; a released DeepSpeed weights-only directory is supported for inference but is not a standard training-resume checkpoint. Omit `--resume_pretrain` to restore optimizer state and continue the saved step counter. Use a new run name for each invocation: the wrapper rejects existing run directories because the upstream trainer overwrites checkpoint tags.

## Evaluation

Use the current [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) workspace with `scripts/eval_policy_xpolicylab.py` and `env_cfg/`, and put this checkout at `RoboTwin/XPolicyLab`. The upstream Evo-1 plugin for RoboTwin `stable_2.0` uses a different interface; copying this adapter into that old `policy/` directory is insufficient. Install RoboTwin's simulator, CuRobo and assets in its separate environment following its own documentation.

```bash
cd RoboTwin/XPolicyLab/policy/Evo_1
conda activate evo1-xpl
bash download_checkpoint.sh

# The clean-trained checkpoint can be evaluated on either scene setting.
EVO1_EVAL_TASK_CONFIG=demo_clean bash eval.sh \
  RoboTwin adjust_bottle Evo1_RoboTwin2_clean arx_x5 joint 0 0 0 evo1-xpl RoboTwin

EVO1_EVAL_TASK_CONFIG=demo_randomized bash eval.sh \
  RoboTwin adjust_bottle Evo1_RoboTwin2_clean arx_x5 joint 0 0 0 evo1-xpl RoboTwin
```

The downloader uses [MINT-SJTU/Evo1_RoboTwin2_clean](https://huggingface.co/MINT-SJTU/Evo1_RoboTwin2_clean). Optionally pass a destination, HF revision and repository: `bash download_checkpoint.sh /absolute/path/to/checkpoint <revision> <repo_id>`. The repository defaults to the clean checkpoint above. Each checkpoint directory needs `config.json`, `norm_stats.json`, and its weight file. You can pass that absolute directory as `ckpt_name`, or use `checkpoint_path` in `deploy.yml` (shared resolver precedence applies).

For the RoboTwin550 checkpoint, download [MINT-SJTU/Evo1_RoboTwin2_datascale](https://huggingface.co/MINT-SJTU/Evo1_RoboTwin2_datascale) at the verified revision:

```bash
bash download_checkpoint.sh checkpoints/Evo1_RoboTwin2_datascale \
  f2bca91d0f230cd300b447d0d332154c592334ce MINT-SJTU/Evo1_RoboTwin2_datascale
```

Set `dataset_key_suffix: _clean` in `deploy.yml` for `demo_clean`, or `_rand` for `demo_randomized`, and pass `Evo1_RoboTwin2_datascale` as `ckpt_name`. Restore the empty suffix when using RoboTwin50. If Hugging Face is unreachable, `HF_ENDPOINT=https://hf-mirror.com` selects the mirror used for this verification.

`EVO1_EVAL_TEST_NUM=1` limits simulator smoke checks; omit it for the benchmark's default episode count. For the required wiring checks with real weights, run both modes:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin adjust_bottle Evo1_RoboTwin2_clean arx_x5 joint 0 0 0 evo1-xpl RoboTwin
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboTwin adjust_bottle Evo1_RoboTwin2_clean arx_x5 joint 0 0 0 evo1-xpl RoboTwin
```

## Configuration

- `action_horizon: 37` is the number of actions executed; the released model predicts 50.
- `num_inference_timesteps: 50` controls flow-matching denoising.
- `smoothing_kernel: 9` applies the original Gaussian smoother to the **complete denormalized chunk before truncation**, including its boundary renormalization and float64 arithmetic.
- Camera order is head, left wrist, right wrist. XPolicyLab supplies RGB directly; the old JSON client's RGB→BGR→RGB transport round trip is removed. Resize remains OpenCV linear to 448×448, followed by float32 scaling to [0, 1].
- State and action order is `6+1+6+1`; dimensions come from the shared robot metadata. The action mask selects those 14 entries from the 24-D model output. Joint targets and grippers remain absolute, with no extra thresholding or clipping.
- `arm_key: aloha_joint`; the default dataset key is `robotwin_<task_name>`. The current RoboTwin client's `prepare_case` hook switches tasks before `reset`. Unknown tasks fail explicitly. For checkpoints whose statistics have setting suffixes, choose `dataset_key_suffix: _clean` or `_rand` explicitly. **A clean-trained checkpoint keeps the empty suffix even in `demo_randomized`.** `dataset_key` is an optional explicit override.
- `request_timeout_s: 180.0` sets the WebSocket request timeout in the shared debug client; simulator clients use their own timeout settings.
- `seed` initializes Torch RNG once when loading; episode resets clear observations without reseeding. `eval_batch: false`; run a separate server per concurrently active rollout.

## Validation and limits

The CPU contract suite uses the native Evo-1 normalizer and a deterministic test network to check camera/state mapping, smoothing, task switching, reset behavior, raw/encoded-image WebSocket calls, and data/launcher behavior:

```bash
# Run from XPolicyLab root after installing the policy environment and source.
EVO1_SOURCE_DIR=/absolute/path/to/Evo-1 python -m unittest discover -s policy/Evo_1/tests -v
```

The CPU suite alone does not establish checkpoint-backed inference or simulator success.

Real release weights passed both raw and encoded-image debug closed loops. Full evaluation on RTX 4090 GPUs then completed 50 tasks × 100 episodes per setting for each checkpoint (20,000 episodes total):

| Checkpoint | Scene setting | Successful / evaluated episodes | Success rate |
| --- | --- | --- | --- |
| RoboTwin50 | `demo_clean` | 3192 / 5000 | **63.84%** |
| RoboTwin50 | `demo_randomized` | 630 / 5000 | **12.60%** |
| RoboTwin550 | `demo_clean` | 4258 / 5000 | **85.16%** |
| RoboTwin550 | `demo_randomized` | 4139 / 5000 | **82.78%** |

Evaluations used RoboTwin `6dde57155eafa3e4ebf6ad1f93a7cf7d5d41a755`, seed 0, h37/N50/Gaussian k9, and the official expert feasibility filter and success checks. Clean used seen instructions, randomized used unseen instructions. Each group ran on eight GPUs with one environment per GPU; zero-success tasks were retained. RoboTwin50 kept unsuffixed task keys; RoboTwin550 used `_clean` / `_rand` statistics. The scores are completed-task aggregates; independent verification of all episode/video/archive artifacts remains pending.

The tested policy environment used Torch 2.8.0, transformers 4.39.0 and FlashAttention 2.8.0.post2. The separate simulator used SAPIEN 3.0.0b1 and CuRobo v0.7.8 compiled for RTX 4090 (`sm_89`). The installer's default Torch 2.5.1 / FlashAttention 2.7.4.post1 recipe has not been validated from a fresh environment. An end-to-end training check passed on H20 using RoboTwin demonstration data: statistics recomputation, training with decreasing loss, checkpoint saving, optimizer resume, and strict adapter reload followed by inference. This validates the training/save/resume/inference pipeline, not convergence or reproduction of the released checkpoint training recipe. Historical Evo-1 benchmark results used `stable_2.0`; these results do not establish controlled equivalence across versions.

Tests used local release packages. Weight SHA-256: RoboTwin50 `3c1226e9ea197de575287989a787b3c9c71e7d1b1fad16b16cd0912986ef15e6`; RoboTwin550 `5ad97a3e69e3e20b331a9b5e7a892dfa4ea4f7a4be03ca0af7e02bf5baed6776`. The public RoboTwin550 package at revision `f2bca91d0f230cd300b447d0d332154c592334ce` was downloaded completely via hf-mirror.com: all 1,554,370,173 weight bytes, config and normalization statistics match the evaluated local release. Direct huggingface.co access timed out. The clean repository is `MINT-SJTU/Evo1_RoboTwin2_clean`; its public LFS weight hash matches the evaluated RoboTwin50 weight.

This adapter covers RoboTwin joint control; RoboDojo, end-effector control, batched rollouts and real hardware are not validated.

The Gaussian smoother is derived from MINT-SJTU/Evo-1. Its MIT license and copyright notice are retained in [LICENSE.upstream](LICENSE.upstream).
