# GR00T-N1.6 for XPolicyLab

This directory adapts NVIDIA Isaac-GR00T to the XPolicyLab policy workflow:

- convert XPolicyLab HDF5 episodes to GR00T LeRobot v2 data
- finetune from the local GR00T-N1.6 base checkpoint, configurable with `BASE_MODEL_PATH`
- run XPolicyLab debug/sim evaluation through the standard model server

The Isaac-GR00T source is kept under `Isaac-GR00T`. The XPolicyLab adaptation lives in this directory (`process_data.py`, `model.py`, `deploy.py`, and shell entrypoints).

## Environment

Install the environment according to `Isaac-GR00T/README.md`. If you use the Isaac-GR00T `uv` environment, install XPolicyLab into that environment from this policy directory:

```bash
cd XPolicyLab/policy/GR00T-N1.6/Isaac-GR00T

# Install Isaac-GR00T dependencies into the uv environment.
uv sync

# Register XPolicyLab in the same uv environment.
uv pip install -e ../../..

# Runtime packages used by XPolicyLab and this GR00T adapter.
uv pip install PyYAML h5py pyarrow pandas opencv-python tqdm

# Make bash process_data.sh/train.sh use the uv environment's python.
source .venv/bin/activate
```

Check the environment:

```bash
uv run python -c "import XPolicyLab, yaml, h5py, cv2, pandas, pyarrow, tqdm; print('ok')"
```

After activation, return to this policy directory before running the XPolicyLab entrypoints:

```bash
cd ..
```

If you are using a normal Python or Conda environment instead of `uv`, run:

```bash
cd XPolicyLab/policy/GR00T-N1.6
bash install.sh
```

`install.sh` intentionally does not create or switch conda environments. Use the environment names in `eval.sh` arguments.

Use the N1.6 release source with the N1.6 checkpoint. If you clone Isaac-GR00T yourself, use:

```bash
git clone --recurse-submodules -b n1.6-release https://github.com/NVIDIA/Isaac-GR00T
```

`n1.6-release` is a tag, so Git may report `detached HEAD`; that is normal. Do not use the default branch with the N1.6 checkpoint, because the default branch can move to newer GR00T versions and fail to recognize `Gr00tN1d6` checkpoints.

## Data Conversion

```bash
cd XPolicyLab/policy/GR00T-N1.6
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

Example:

```bash
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

Output:

```text
policy/GR00T-N1.6/data/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}
```

The converter writes:

- `data/chunk-000/*.parquet`
- `videos/chunk-000/observation.images.*/*.mp4`
- `meta/info.json`
- `meta/modality.json`
- `meta/stats.json`
- `meta/relative_stats.json` when relative joint actions are used
- `xpolicylab_gr00t_config.py`

Images are resized to `320x240`. XPolicyLab stores BGR images; training videos are written from BGR frames and inference converts observations to RGB before passing them to GR00T.

For `action_type=ee`, the 7D pose `[x, y, z, qw, qx, qy, qz]` is preserved. This integration does not do 7D-to-6D or quaternion-to-Euler conversion.

## Training

```bash
bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed}
```

Example:

```bash
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0 42
```

Defaults:

- base model: `${repo_parent}/models/GR00T-N1.6-3B`, or override with `BASE_MODEL_PATH`
- checkpoints: `policy/GR00T-N1.6/checkpoints`
- converted data: `policy/GR00T-N1.6/data`
- action horizon: `16`

Useful overrides:

```bash
BASE_MODEL_PATH=../../../../models/GR00T-N1.6-3B \
MAX_STEPS=10000 \
SAVE_STEPS=1000 \
GLOBAL_BATCH_SIZE=32 \
USE_WANDB=1 \
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0,1 42
```

The run name is:

```text
${task_name}-gr00t-${action_type}-${expert_data_num}eps-seed${seed}-${timestamp}
```

A latest marker is written to:

```text
checkpoints/${task_name}-gr00t-${action_type}-${expert_data_num}eps-seed${seed}.latest
```

## Evaluation

```bash
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${policy_gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [env_gpu_id] [MODEL_PATH]
```

Example:

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 50 joint 0 42 gr00t_env XPolicyLab 0
```

If `MODEL_PATH` is omitted, `eval.sh` uses the latest checkpoint for the run signature. To evaluate a specific checkpoint:

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 50 joint 0 42 gr00t_env XPolicyLab 0 \
  checkpoints/.../checkpoint-1000
```

`deploy.yml` defaults to `eval_env: debug`. Change it to `sim` for simulation evaluation after debug passes.

## Notes

`GR00T-N1.6` is not a valid Python module name. The eval script exposes this directory to XPolicyLab as the import alias `GR00T_N1_6` via `xpolicylab_alias/sitecustomize.py`; users should still run commands from this directory.
