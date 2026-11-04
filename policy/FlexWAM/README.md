# FlexWAM

**Contributor:** FlexWAM Contributors

`FlexWAM` adapts the FlexWAM world-action model to XPolicyLab/RoboDojo (`arx_x5`, `piper`, `piper_x`; absolute EE commands; delta rot6d training). Integration scripts live at this directory level. The vendored implementation lives in `flexwam/`. `infer_action` is single-environment, so `get_action_batch` runs the active envs one after another.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Create a Python >= 3.10 environment, then install a CUDA PyTorch wheel before the adapter packages:

```bash
conda create -n flexwam python=3.10 -y
conda activate flexwam
pip install -U pip
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/FlexWAM
bash install.sh
```

`install.sh` editable-installs XPolicyLab and the vendored `flexwam/` package.

## Data Processing

Training reads a LeRobot v2.1 dataset. Camera keys match the [official converters](../../README.md#official-lerobot-conversion): `observation.images.cam_high`, `cam_left_wrist`, `cam_right_wrist`. State and action use a different layout:

| Key | Content |
| --- | --- |
| `observation.ee_pose`, `action.ee_pose` | 20-D, left then right: `xyz(3) + rot6d first two rows(6) + gripper(1)` in each arm's robot base |

`process_data.sh` only checks that the directory exists (`FLEXWAM_DATASET_DIR`, otherwise `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`). Point it at a dataset that already has these keys. The official joint-space export (`observation.state` / `action`) will not load.

```bash
cd XPolicyLab/policy/FlexWAM
export FLEXWAM_DATASET_DIR=/path/to/robodojo_lerobot_ee_pose
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

## Model Assets

Download [Zzming/FlexWAM-base](https://huggingface.co/Zzming/FlexWAM-base) once. Set `FLEXWAM_MODEL_BASE_PATH` and `DIFFSYNTH_MODEL_BASE_PATH` to that directory. `arx_x5`, `piper`, and `piper_x` share it. Trained DiT weights stay in each checkpoint's `final.pt`.

| Path | Content |
| --- | --- |
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | Wan2.2 VAE |
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors` | UMT5 encoder |
| `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/` | UMT5 tokenizer |

| `env_cfg_type` | Checkpoint |
| --- | --- |
| `arx_x5` | [Zzming/FlexWAM-RoboDojo-Real-ARX-X5](https://huggingface.co/Zzming/FlexWAM-RoboDojo-Real-ARX-X5) |
| `piper` | [Zzming/FlexWAM-RoboDojo-Real-Piper](https://huggingface.co/Zzming/FlexWAM-RoboDojo-Real-Piper) |
| `piper_x` | [Zzming/FlexWAM-RoboDojo-Real-Piper-X](https://huggingface.co/Zzming/FlexWAM-RoboDojo-Real-Piper-X) |

Each checkpoint directory has `final.pt`, `dataset_stats.json`, and `config.yaml`. Pass the directory name as `ckpt_name`.

```bash
cd XPolicyLab/policy/FlexWAM
mkdir -p checkpoints
huggingface-cli download Zzming/FlexWAM-base \
  --local-dir checkpoints/FlexWAM-base
huggingface-cli download Zzming/FlexWAM-RoboDojo-Real-ARX-X5 \
  --local-dir checkpoints/FlexWAM-RoboDojo-Real-ARX-X5
huggingface-cli download Zzming/FlexWAM-RoboDojo-Real-Piper \
  --local-dir checkpoints/FlexWAM-RoboDojo-Real-Piper
huggingface-cli download Zzming/FlexWAM-RoboDojo-Real-Piper-X \
  --local-dir checkpoints/FlexWAM-RoboDojo-Real-Piper-X

export FLEXWAM_MODEL_BASE_PATH="$PWD/checkpoints/FlexWAM-base"
export DIFFSYNTH_MODEL_BASE_PATH="$FLEXWAM_MODEL_BASE_PATH"
```

`setup_eval_policy_server.sh` uses `checkpoints/FlexWAM-base` when those variables are unset. All three embodiments must resolve to this directory.

A locally trained run can be linked the same way:

```bash
ln -sfn /path/to/run checkpoints/RoboDojo-cotrain-arx_x5-ee-0
```

## Training

```bash
cd XPolicyLab/policy/FlexWAM
export FLEXWAM_DATASET_DIR=/path/to/robodojo_lerobot_ee_pose
export FLEXWAM_MODEL_BASE_PATH="$PWD/checkpoints/FlexWAM-base"
export DIFFSYNTH_MODEL_BASE_PATH="$FLEXWAM_MODEL_BASE_PATH"
export FLEXWAM_ACTION_DIT_PATH=/path/to/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example
bash train.sh RoboDojo cotrain arx_x5 ee 0 0
```

`FLEXWAM_MODEL_BASE_PATH` is the frozen base above. `FLEXWAM_ACTION_DIT_PATH` is only the ActionDiT initialization used by training; it is not inside FlexWAM-base, and deployment does not load it. `env_cfg_type` selects the Hydra task: `arx_x5` → `wan/robodojo_arx5`, `piper` → `wan/robodojo_piper`, `piper_x` → `wan/robodojo_piperx`. Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Extra Hydra overrides go in `FLEXWAM_TRAIN_OVERRIDES`.

## Evaluation

`action_type` is `ee`. `task_name` is the RoboDojo task, for example `stack_bowls`. `env_cfg_type` is `arx_x5`, `piper`, or `piper_x`.

```bash
cd XPolicyLab/policy/FlexWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Wiring check only. No robot and no weights; the server holds the current pose.
EVAL_ENV_TYPE=debug FLEXWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls dummy arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Real robot. Loads checkpoints/FlexWAM-RoboDojo-Real-ARX-X5/final.pt.
EVAL_ENV_TYPE=real BASE_CFG=/path/to/base_cfg \
  bash eval.sh RoboDojo stack_bowls FlexWAM-RoboDojo-Real-ARX-X5 arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

Released checkpoints are for the real robot: set `EVAL_ENV_TYPE=real` and `BASE_CFG`. Leave `EVAL_ENV_TYPE` unset to start the simulator.

`ckpt_name` resolves in order: `FLEXWAM_CKPT_DIR`; an absolute path; a path containing `/`, relative to `policy/FlexWAM`; `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` if that directory exists; otherwise `checkpoints/<ckpt_name>/`. A release loads `final.pt`. A training run loads the highest `checkpoints/weights/step_*.pt`. Split-machine deployment follows the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `eval_batch`, `action_type`, `device`, `ckpt_dir`, `flexwam_root`, `delta`, `replan_steps`, `num_inference_steps`, `cuda_graph`, `allow_dummy_policy`.

| Variable | Notes |
| --- | --- |
| `FLEXWAM_DATASET_DIR` | LeRobot root with `ee_pose` keys for `process_data.sh` / `train.sh`. |
| `FLEXWAM_CKPT_DIR` | Explicit eval run directory. |
| `FLEXWAM_CHECKPOINT_PATH` | Explicit `step_*.pt`, instead of the latest file in the run directory. |
| `FLEXWAM_DATASET_STATS` | Explicit `dataset_stats.json`. |
| `FLEXWAM_ROOT` | Optional source override; defaults to the vendored `flexwam/`. |
| `FLEXWAM_MODEL_BASE_PATH` | [Zzming/FlexWAM-base](https://huggingface.co/Zzming/FlexWAM-base): frozen VAE, UMT5 encoder, and tokenizer. `DIFFSYNTH_MODEL_BASE_PATH` must be the same directory. Default `checkpoints/FlexWAM-base`. |
| `FLEXWAM_TASK` | Optional Hydra task override. |
| `FLEXWAM_TRAIN_OVERRIDES` | Extra Hydra overrides forwarded to training. |
| `FLEXWAM_ALLOW_DUMMY_POLICY` | Debug-only: skip checkpoint load and return hold-position chunks. |
| `FLEXWAM_CUDA_GRAPH` | Override `cuda_graph`. Default `true` for the real-robot server. |

`delta: true` integrates the trained `pose6d_delta_bimanual` output and returns absolute EE poses in the same robot-base frame as the observation. Real `arx_x5`, `piper`, and `piper_x` poses are already in that frame, so evaluation skips the simulation env-origin transform. `replan_steps: 12` executes the first 12 steps of each 32-step chunk. `cuda_graph: true` captures one environment, three cameras, and `192×224`.
