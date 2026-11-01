# OpenWAM

**Contributor:** OpenWAM Contributors | **Paper:** An Open, Modular Exploration Towards Systematic World–Action Model Pretraining | **arXiv:** [2609.07398](https://arxiv.org/abs/2609.07398) | **Original code:** https://github.com/OpenWAM-Official/OpenWAM

`OpenWAM` adapts the OpenWAM world-action model to XPolicyLab/RoboDojo (`arx_x5`, absolute EE control, batched inference), plus the WUJI dual dexterous-hand platform (`tianji_wuji`) in the 58-D hybrid representation — see [WUJI 58-D hybrid](#wuji-58-d-hybrid-tianji_wuji). Integration scripts live at this directory level; the vendored upstream implementation lives in `OpenWAM/`. Official OpenWAM does not expose batch inference; the vendored tree adds `generate_batch` for `eval_batch: true`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Create a Python >= 3.10 environment, then install the official PyTorch CUDA wheel before the adapter packages:

```bash
conda create -n openwam python=3.10
conda activate openwam
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/OpenWAM
bash install.sh
```

`install.sh` editable-installs XPolicyLab and the vendored `OpenWAM/` package. Cosmos-Predict2.5 extras are not required for the RoboDojo Wan checkpoint.

## Data Processing

OpenWAM consumes native RoboDojo HDF5 (`<dataset_dir>/<task>/<embodiment>/data/episode_*.hdf5`). There is no LeRobot conversion.

```bash
cd XPolicyLab/policy/OpenWAM
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
export OPENWAM_DATASET_DIR=/path/to/robodojo_data
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

If `OPENWAM_DATASET_DIR` (or `policy/OpenWAM/data/<bench>-<ckpt>-<env>-<action>/`) already exists, the script exits. Otherwise it launches the official interactive downloader `OpenWAM/scripts/download_assets/download_benchmark_data.py` — select RoboDojo.

## Model Assets

Released RoboDojo SFT checkpoints are self-contained (`config.yaml`, `checkpoint_step_*.safetensors`, `normalization_stats.npy`). Download from [Hugging Face OpenWAM](https://huggingface.co/OpenWAM) or with the official helper:

```bash
cd XPolicyLab/policy/OpenWAM/OpenWAM
python scripts/download_assets/download_openwam_checkpoints.py
```

Place or symlink the checkpoint directory where eval can resolve it:

```bash
cd XPolicyLab/policy/OpenWAM
mkdir -p checkpoints
ln -sfn /path/to/New_OpenWAM_RoboDojo_SFT_60k checkpoints/New_OpenWAM_RoboDojo_SFT_60k
```

Training from scratch also needs the Wan video backbone (`python scripts/download_assets/download_video_backbone.py` inside `OpenWAM/`). Fine-tuning a released checkpoint uses `OPENWAM_FINETUNE_CKPT_PATH`.

## Training

```bash
cd XPolicyLab/policy/OpenWAM
export OPENWAM_DATASET_DIR=/path/to/robodojo_data

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: RoboDojo SFT on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 ee 0 0
```

This wraps official `OpenWAM/scripts/train.sh` with `dataloader=robodojo`. Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Extra Hydra overrides go in `OPENWAM_TRAIN_OVERRIDES`.

## Evaluation

```bash
cd XPolicyLab/policy/OpenWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline wiring check (no simulator, no checkpoint load)
EVAL_ENV_TYPE=debug OPENWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls dummy arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a released checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls New_OpenWAM_RoboDojo_SFT_60k arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. `ckpt_name` may be the short directory name, the full 5-tuple run directory, or an absolute path. Override with `OPENWAM_CKPT_DIR`. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## WUJI 58-D hybrid (`tianji_wuji`)

The adapter also serves the WUJI dual-arm dexterous-hand platform (`tianji_wuji`: two 7-DOF arms carrying two 20-DOF Wuji hands) in the 58-D hybrid representation defined by WUJI SimBench (`wrist_rel_te2`) and owned by `XPolicyLab.utils.wuji58`. It is selected by the run tuple alone — `env_cfg_type=tianji_wuji` (registered as `arm_dim [7, 7]`, `ee_dim [20, 20]`) with `action_type=ee` — and is a separate representation from EEF20: an `arx_x5` checkpoint and a `tianji_wuji` checkpoint are not interchangeable, and `model.py` refuses a checkpoint whose `dataloader.action_mode` does not match the run.

| Slice | Content | State | Action |
|---|---|---|---|
| `[0:20]` | left hand joints (finger-major) | absolute | absolute target |
| `[20:29]` | left wrist `xyz + rot6d` | absolute, in the wrist frame | delta vs. the same frame's measured wrist |
| `[29:49]` | right hand joints | absolute | absolute target |
| `[49:58]` | right wrist `xyz + rot6d` | absolute, in the wrist frame | delta vs. the same frame's measured wrist |

The wrist frame is the fixed WUJI SimBench TE2 head camera by default (`wrist_frame: te2`; `world` keeps world coordinates). There is no dual-X5 base calibration and no unify scatter in this mode: the raw width 58 is the model width (`model.architecture.action_dim=58`, `state_dim=58`, `unify_action=false`).

**Data.** Native RoboDojo HDF5 at `<dataset_dir>/<task>/tianji_wuji/data/episode_*.hdf5` with `state/{left,right}_ee_poses` `(T, 7)` and `state/{left,right}_ee_joint_states` `(T, 20)`; when the episode also carries the same keys under `action/`, those are the targets, otherwise the next measured state is (the EEF20 convention). Reader: `OpenWAM/openwam/dataloader/robodojo_wuji58.py` (`read_wuji58`, torch-free), wired into `robodojo.py` as `action_mode: wuji58`.

**Normalization stats.** Auto-built at `<dataset_dir>/meta/robodojo_wuji58_normalization_stats.npy` (or `python -m openwam.dataloader.utils.stats_computation.robodojo_stats_computation --config configs/dataloader/robodojo_wuji58.yaml --output ...`). The file carries two blocks because state and action rows differ: `wuji58` (action rows, used to unnormalize model output) and `wuji58_state` (state rows, used to normalize proprio — the deploy loader picks it up as the `<action_mode>_state` sibling). The rot6d dims `23..28` and `52..57` are pinned to identity in both, and the metadata records the frame contract fingerprint, so a stats file built under another `wrist_frame` is rejected.

**Train / eval.**

```bash
export OPENWAM_DATASET_DIR=/path/to/robodojo_wuji_data
bash process_data.sh RoboDojo wuji tianji_wuji ee
bash train.sh RoboDojo wuji tianji_wuji ee 0 0            # selects dataloader=robodojo_wuji58 + 58-D model dims
bash eval.sh RoboDojo <task> wuji tianji_wuji ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`train.sh` passes `dataloader=robodojo_wuji58 model.architecture.action_dim=58 model.architecture.state_dim=58 dataloader.wrist_frame=${OPENWAM_WRIST_FRAME:-te2}` automatically for `tianji_wuji`. At eval, `model.py` builds the 58-D state with `state58_from_obs` and converts each predicted chunk with the shared `decode_action_chunk58` default (`iterative`): element 0 against the observation, each later element against the pose decoded before it, matching the per-frame deltas the dataloader writes. The returned dicts are the standard `left_ee_pose` / `left_ee_joint_state` (20) / `right_ee_pose` / `right_ee_joint_state` (20). `deploy.yml` adds `wrist_frame: te2`, which must equal the checkpoint's `dataloader.wrist_frame`. The `arx_x5` EEF20 path is unchanged.

CPU contract tests: `python -m pytest policy/OpenWAM/tests -q` from the XPolicyLab root (no torch needed).

## Configuration

`deploy.yml` keys to check before evaluation: `eval_batch`, `action_type`, `device`, `ckpt_dir`, `openwam_root`, `openwam_deploy_config`, `replan_steps`, `allow_dummy_policy`, and for `tianji_wuji` runs `wrist_frame`.

| Variable | Notes |
|---|---|
| `OPENWAM_DATASET_DIR` | Native RoboDojo HDF5 root for `process_data.sh` / `train.sh`. |
| `OPENWAM_CKPT_DIR` | Explicit eval checkpoint directory (`config.yaml` + safetensors). |
| `OPENWAM_ROOT` | Optional OpenWAM source override; defaults to the vendored `OpenWAM/`. |
| `OPENWAM_DEPLOY_CONFIG` | Optional deploy yaml override; defaults to `OpenWAM/configs/deploy.yaml`. |
| `OPENWAM_TRAIN_OVERRIDES` | Extra Hydra overrides forwarded to official `scripts/train.sh`. |
| `OPENWAM_FINETUNE_CKPT_PATH` | Warm-start directory for official `training.finetune_ckpt_path`. |
| `OPENWAM_RESUME_CKPT_PATH` | Resume directory for official `training.resume_ckpt_path`. |
| `OPENWAM_ALLOW_DUMMY_POLICY` | Debug-only: skip checkpoint load and return hold-position chunks. |
| `OPENWAM_WRIST_FRAME` | `tianji_wuji` training only: `te2` (default) or `world`, forwarded as `dataloader.wrist_frame`. |

`eval_batch: true` stacks every running env into one `engine.generate_batch` forward. `model.py` re-forces `dit_cache` / `compile` / `decode_video` off and `inference_mode=sync`. `device: cuda` loads Wan, UMT5-XXL, and ActionDiT on GPU, matching official OpenWAM deploy.
