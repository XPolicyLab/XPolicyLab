# SLA

**Contributor:** Fuhao Li | **Paper:** SLA: Scaling Spatial-Language-Action Model via Continual Pre-training | **arXiv:** [2611.08373](https://arxiv.org/abs/2611.08373) | **Original code:** [SLA](https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA)

SLA adds spatial representation alignment to a vision-language-action policy. This adapter supports data preparation, training, and evaluation for **RoboTwin / arx_x5 / ee**, following the released [train_robotwin configuration](openpi/src/openpi/external_config/train_robotwin.py) and initializing post-training from the pretrained **SLA_base** checkpoint.


## Installation

Use Linux, an NVIDIA GPU compatible with the upstream CUDA 12 / JAX 0.5.3 stack, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The policy uses its own Python 3.11 virtual environment; the environment client uses the benchmark's Conda environment.

```bash
cd XPolicyLab/policy/SLA
bash install.sh
source openpi/.venv/bin/activate
```

The install script synchronizes the upstream uv project and installs XPolicyLab. For data preparation and evaluation, the benchmark workspace must contain `env_cfg/` **beside** the XPolicyLab checkout. Dimensions come from its shared robot-registration helper; the training launcher uses XPolicyLab's `utils/get_action_dim.sh` and checks that its dimensions agree with the prepared dataset.

Evaluation argument 9 is `uv` for this adapter's `openpi/.venv`, or the path to another uv project containing `.venv/bin/python`. Advanced users can set `SLA_PYTHON` to an existing compatible interpreter; adapter scripts still import the bundled SLA source. Do not share this interpreter with another OpenPI variant.

## Data Processing

Consumes **LeRobot v2.1**, with the official XPolicyLab keys: `observation.state`, `action`, and `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`. State/action vectors contain left arm joints, left gripper, right arm joints, right gripper. These are **joint data**, although SLA predicts `ee` actions after its upstream transforms.

Use an export produced by `scripts/transform_lerobot_v21_format.py` ([official conversion](../../README.md#official-lerobot-conversion)), or the upstream RoboTwin v2.1 export below. For raw XPolicyLab trajectories, run the official converter first. `process_data.sh` only validates and links prepared datasets and computes normalization statistics using SLA's FK / SO(3) / delta-action transforms. It does not decode raw trajectories or re-encode videos.

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  <lerobot_directory_or_quoted_glob> [--batch-size N] [--num-workers N] [--max-frames N]

# A merged official export, or a parent containing one dataset per task:
bash process_data.sh RoboTwin cotrain arx_x5 ee /datasets/robotwin_lerobot

# Select specific task directories through a quoted glob:
bash process_data.sh RoboTwin subset arx_x5 ee '/datasets/robotwin_lerobot/stack_*'
```

The upstream public data can be obtained with:

```bash
hf download TianxingChen/RoboTwin2.0 lerobot_dataset/RoboTwin_lerobot_v21.zip \
  --repo-type dataset --local-dir /datasets/robotwin_download
unzip /datasets/robotwin_download/lerobot_dataset/RoboTwin_lerobot_v21.zip \
  -d /datasets/robotwin_lerobot
```

Pass the extracted directory that contains `meta/info.json`, or its parent of task datasets. Dataset links and a source manifest are stored under `data/<bench>-<ckpt>-<env>-<action>/`; statistics are stored under `assets/<same-name>/`. Reusing a name for a different source is rejected. `--max-frames` is for limited statistics/smoke checks; omit it for a training run's full statistics. Defaults are batch size 256 and 32 workers.

All images remain RGB. The policy server decodes runtime observations, and the official converter handles both XPolicyLab image encodings through the shared helpers.

## Training

Download the pretrained `haofuly/SLA_base` weights and the OmniVGGT/DINOv2 training assets, then run RoboTwin post-training with the same `ckpt_name` used for data preparation. The default weight loader reads `checkpoints/SLA_base/params`, matching `train_robotwin.py`:

```bash
bash download_checkpoints.sh training
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [options]

# Released recipe: 256 samples, 8 GPUs, 30,000 updates, spatial alignment enabled.
bash train.sh RoboTwin cotrain arx_x5 ee 0 0,1,2,3,4,5,6,7
```

The model settings in `train_robotwin.py` are `pi05=True`, `paligemma_variant="gemma_2b_lora"`, and `action_expert_variant="gemma_300m"`. These specify the architecture and LoRA parameterization; the pretrained initialization is selected separately by `CheckpointWeightLoader("checkpoints/SLA_base/params")`.

The wrapper calls the released `scripts/train_align.py`, preserving that model configuration and freeze filter, EMA (`0.999`), state dropout (`0.25`), and Spatial Forcing alignment with OmniVGGT. FSDP and teacher device counts default to the number of selected GPUs. Useful overrides:

| Option | Default / purpose |
| --- | --- |
| `--batch-size`, `--num-workers` | `256`, `32`; batch size must divide evenly across visible GPUs. |
| `--num-train-steps`, `--save-interval` | `30000`, `30000`. |
| `--fsdp-devices`, `--align-vggt-devices` | Number of selected GPUs; shared teacher/student devices. |
| `--align-feature-batch-size` | `32`; teacher microbatch size. |
| `--gradient-accumulation-steps` | `1`; effective batch size is batch size times this value. |
| `--base-checkpoint` | `checkpoints/SLA_base`; pretrained SLA initialization, loaded from its `params/` subdirectory. |
| `--omnivggt-checkpoint` | `checkpoints/omnivggt`, containing `OmniVGGT.safetensors`. |
| `--dinov2-checkpoint` | `checkpoints/dinov2/dinov2_vitl14_reg4_pretrain.pth`; OmniVGGT's offline encoder initialization. |
| `--resume` | Resume an existing run; existing outputs are never overwritten implicitly. |
| `--smoke-test` | Random student initialization, for integration tests only; run name must start with `smoke`. |

Weights and their training-time statistics are saved together:

```text
checkpoints/RoboTwin-cotrain-arx_x5-ee-0/<step>/
├── params/
├── assets/RoboTwin-cotrain-arx_x5-ee/norm_stats.json
└── train_state/
```

Normalization and checkpoint names are built with XPolicyLab's shared naming helper. Training does not require the outer `env_cfg/` tree. Weights & Biases reporting is disabled by the wrapper. No benchmark accuracy is implied by the random-initialization smoke mode.

## Evaluation

Evaluation loads the post-trained `SLA_robotwin_ckpt` from `haofuly/SLA_benchmark_ckpts`, or a checkpoint produced by `train.sh`. Model structure and observation/action transforms come from `train_robotwin`; parameters and normalization statistics come from the selected evaluation checkpoint.

```bash
bash download_checkpoints.sh robotwin
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <uv_or_project_path> <eval_env_conda_env>

# Public checkpoint, within an installed RoboTwin workspace:
bash eval.sh RoboTwin beat_block_hammer SLA_robotwin_ckpt arx_x5 ee 0 0 0 uv robotwin

# A locally trained run; selects the latest numeric checkpoint step:
bash eval.sh RoboTwin beat_block_hammer cotrain arx_x5 ee 0 0 0 uv robotwin

# Offline wiring check, no simulator:
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin beat_block_hammer SLA_robotwin_ckpt arx_x5 ee 0 0 0 uv robotwin
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboTwin beat_block_hammer SLA_robotwin_ckpt arx_x5 ee 0 0 0 uv robotwin
```

`ckpt_name` accepts the short run name, full run-directory name, or a checkpoint path. Relative paths are resolved from this adapter. The public download creates `checkpoints/SLA_robotwin_ckpt/{params,assets}` and excludes optimizer states and other benchmarks. For split-machine evaluation use the standard setup scripts and [deployment flow](../../README.md#-deployment-flow).

Observations must include all three cameras and joint/gripper states even when evaluating with `action_type=ee`. Upstream inference converts joint observations to its common EEF frame, restores absolute actions from the predicted deltas, and returns RPY orientations. This adapter applies the original RoboTwin base transform and emits world-frame `left_ee_pose` / `right_ee_pose` in quaternion `wxyz` order plus scalar gripper arrays. There is no TCP offset or channel swap. The default chunk has 50 actions. Batch calls use sequential inference with explicit environment-index matching.

## Configuration

The standard fields in `deploy.yml` follow XPolicyLab's launch conventions. Adapter-specific fields:

| Key | Default / purpose |
| --- | --- |
| `train_config_name` | `train_robotwin`; defines model structure and transforms. Evaluation weights come from the selected checkpoint. |
| `checkpoint_path` | `null`; explicit checkpoint/run path, using the shared resolver. |
| `checkpoint_step` | `null`; latest numeric step, or specify an exact integer under the resolved run. |
| `norm_stats_asset_id` | `null`; requires one unique `assets/*/norm_stats.json` in the selected checkpoint. Set explicitly when several exist. |
| `action_chunk_size` | `50`; may be reduced to 1–50 executed steps. |
| `num_inference_steps` | `10`; diffusion/flow sampling steps. |
| `robot_base_pose` | `[0,-0.65,0,0.707,0,0,0.707]`, translation and `wxyz`, matching the original RoboTwin evaluation. |
| `request_timeout_s` | `600`; allows first-inference JAX compilation. |

The released checkpoint stores statistics under `assets/omnisf_12.2_jax/`, despite the public configuration being named `train_robotwin`. The adapter always loads statistics from the chosen checkpoint and validates their dimensions. It never falls back to unrelated local training statistics. The common EEF origin comes from the supplied URDF.

## Notes

The supplied kinematics asset, `openpi/assets/embodiedments/arx5_description_isaac.urdf`, comes from the contributor's original training and evaluation installations; both copies have SHA-256 `097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a`. Original XML attribution is retained. Pinocchio loads only its kinematics, so visual meshes are not required.

The vendored source includes a dependency lockfile and an offline DINOv2 path override through `SLA_DINOV2_WEIGHTS`, set by the training wrapper. Upstream tests and four unused native Aloha converters are omitted. Public model and teacher downloads use fixed Hugging Face revisions recorded in `download_checkpoints.py`.
