# DP (Diffusion Policy)

This directory is the XPolicyLab integration layer for [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/). Core training code lives in `diffusion_policy/`; data, training, and evaluation scripts are at the top level.

| Item | Link |
|------|------|
| Paper | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/) (RSS 2023) |
| Upstream repo | [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) |
| XPolicyLab integration | RoboDojo team |

## Environment Setup

```bash
cd policy/DP
bash install.sh
conda activate <your_env>
```

`install.sh` installs PyTorch, diffusers, zarr, and related dependencies, then installs `diffusion_policy` and XPolicyLab in editable mode.

## Data Processing

Convert HDF5 under `data/<bench_name>/<ckpt_name>/<env_cfg_type>/` into DP training zarr:

```bash
bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint
```

- **Input**: `data/<bench_name>/<ckpt_name>/<env_cfg_type>/data/episode_*.hdf5`
- **Output**: `policy/DP/data/<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>.zarr`
- **Observations**: three RGB streams (head / left_wrist / right_wrist), resized to `240×320`
- **State / action**: packed into `agent_pos` and action vectors according to `action_type` (e.g. `joint`)

`ckpt_name` names the experiment and locates processed data; for single-task runs it usually matches the raw data directory name.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Example:

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0
```

If the zarr file is missing, `train.sh` calls `process_data.sh` automatically.

Checkpoints are saved to:

```text
policy/DP/checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>/<epoch>.ckpt
```

The default checkpoint is saved at epoch 600; `deploy.yml` sets `checkpoint_num: 600` to load that weight.

### Training Details (defaults)

| Parameter | Value |
|-----------|-------|
| Vision encoder | ResNet18 + GroupNorm |
| Diffusion model | Conditional UNet1D, DDPM 100 steps |
| horizon / n_obs_steps / n_action_steps | 8 / 3 / 6 |
| batch_size / lr / epochs | 128 / 1e-4 / 600 |
| EMA | enabled (inference uses EMA weights) |
| Optimizer | AdamW, cosine lr + 500 warmup steps |

Main hyperparameters are in `diffusion_policy/config/robot_dp.yaml` and can be overridden via Hydra.

## Evaluation

### Same-machine evaluation

```bash
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 50 joint 0 0 0 <policy_env> <eval_env>
```

Argument order: `bench_name task_name ckpt_name env_cfg_type expert_data_num action_type seed policy_gpu env_gpu policy_conda_env eval_env_conda_env`.

- `task_name`: simulation task to run
- `ckpt_name`: experiment name used to load the checkpoint (usually equals `task_name` for single-task runs)

### Two-machine evaluation

Start the policy server on the GPU machine:

```bash
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls stack_bowls arx_x5 50 joint 0 \
  <policy_env> 5000 0.0.0.0
```

Start the env client on the simulation machine (replace the last argument with the policy server IP):

```bash
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls stack_bowls arx_x5 joint 0 \
  <eval_env> "ckpt_name=stack_bowls,action_type=joint" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE` to control the evaluation mode: unset or `sim` for simulation, `debug` for offline shape/IO validation.

## Inference Notes

- At inference, the policy keeps `n_obs_steps=3` frames of observation history and predicts `n_action_steps=6` actions per chunk, executed sequentially.
- Images are normalized to `[0, 1]` and resized to `240×320` in `model.py`, matching training.
- `deploy.yml` sets `eval_batch: true` for batched simulation evaluation.
