# JC-Robot-VLA

**Contributor:** JC Robot Team | **Paper:** JC-Robot-VLA technical report | **arXiv:** TBD

`JC-Robot-VLA` is a Vision-Language-Action (VLA) policy designed for dual-arm robot manipulation tasks. It combines vision-language understanding with flow matching-based action generation for precise dual-arm control.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `process_data.sh` | Converts RoboDojo demonstration data into the policy-specific training format. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `openpi/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Architecture

JC-Robot-VLA uses a Vision-Language-Action architecture with the following components:

1. **Vision Encoder**: Extracts visual features from multi-camera observations (cam_high, cam_left_wrist, cam_right_wrist)
2. **Language Encoder**: Processes natural language instructions
3. **Flow Matching Action Head**: Generates action sequences using flow matching
4. **Dual-Arm Action Representation**: 14-dimensional action space (2 × (6D arm + 1D gripper))

## Key Features

### 1. Action Chunk Execution (Closed-Loop Control)
- **Parameter**: `action_chunk_exec_steps`
- **Default**: 10 (vs 50 for open-loop)
- **Benefit**: Converts open-loop to closed-loop control, re-inferring every 10 steps with fresh observations
- **Impact**: Improves performance on tasks requiring precise control

### 2. Action Clipping (Safety Bounds)
- **Parameter**: `action_clip_enabled`, `action_clip_margin`
- **Default**: enabled, margin=0.1
- **Benefit**: Clips predicted actions to training data bounds (q01/q99 ± margin)
- **Impact**: Prevents extreme joint commands during distribution shift

### 3. Temporal Action Ensembling (Optional)
- **Parameter**: `temporal_ensemble_enabled`, `temporal_ensemble_alpha`
- **Default**: disabled, alpha=0.3
- **Benefit**: Blends overlapping action chunks to reduce jitter
- **Impact**: Smoother actions, reduced jitter

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_uv_env` | `uv` to use `deploy.yml` `policy_uv_env_path`, or an explicit project path. |

```bash
cd XPolicyLab/policy/JC-Robot-VLA
# Example: install dependencies for the JC-Robot-VLA policy adapter.
bash install.sh
# `eval.sh` arg 9 is not a conda env. Pass `uv` or the project path.
source openpi/.venv/bin/activate
```

## Demo Data Processing

What it does: prepares RoboDojo demonstration data for policy training. The output name should match the training run identity so `train.sh` can find it.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Data/run identifier. Use a different value for ablations, for example `stack_bowls_50ep`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `expert_data_num` | Optional episode limit for data conversion only. It is not part of checkpoint naming. |
| `raw_task_dirs` | Optional source task directory or comma-separated task list under `data/<bench_name>/`; defaults to `ckpt_name`. |

```bash
cd XPolicyLab/policy/JC-Robot-VLA
# Template: convert all available demonstrations for one run.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: convert stack_bowls demos for arx_x5 joint control.
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: write a differently named dataset while reading all stack_bowls demos.
bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls

# Example: create a 50-episode ablation while reading from the original task data.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Model Training

What it does: starts the policy-specific training recipe through the XPolicyLab wrapper and writes checkpoints under this adapter directory.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Training run identifier, for example `cotrain`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Random seed. |
| `gpu_id` | GPU id or comma-separated GPU ids for the policy trainer. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default. |

```bash
cd XPolicyLab/policy/JC-Robot-VLA
# Template: train a policy run on one GPU or a GPU list.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: train the same run on four GPUs if the upstream trainer supports it.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

The usual checkpoint directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. During evaluation, `ckpt_name` may be the short run name from training (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory.

By default, training reads the LeRobot repo produced by `process_data.sh`: `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. Override this with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client. Use `eval.sh` for a same-machine smoke test, or split server/client scripts for debugging and multi-machine evaluation.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` or an explicit project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/JC-Robot-VLA
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

Parameters used by the split server/client flow:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` or an explicit project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=joint`. |

```bash
cd XPolicyLab/policy/JC-Robot-VLA
# Terminal 1 on the policy machine: start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_uv_env> <policy_server_port> <policy_server_host>

# Example: bind the policy server to all interfaces on port 5000.
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 uv 5000 0.0.0.0

# Terminal 2 on the environment machine: connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>

# Example: connect to a policy server reachable at <policy_server_ip>:5000.
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 <eval_env_conda_env> "ckpt_name=RoboDojo-cotrain-arx_x5-joint-0,action_type=joint" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks when the adapter supports it; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Configuration

Edit `deploy.yml` to customize optimization parameters:

```yaml
# Action chunk execution: only execute first N steps of each 50-step chunk
action_chunk_exec_steps: 10  # Recommended: 10 (closed-loop) or null (open-loop)

# Action clipping: clips predicted actions to training data bounds
action_clip_enabled: true
action_clip_margin: 0.1  # Fraction of q-range to add as safety margin

# Temporal action ensembling: blends overlapping action chunks
temporal_ensemble_enabled: false
temporal_ensemble_alpha: 0.3  # 0.0 = fully trust old, 1.0 = fully trust new
```

## Evaluation Results

### RoboDojo Benchmark (54 tasks)

| Metric | Value |
|--------|-------|
| Tasks Completed | 54/54 (100%) |
| Average Score | 18.04 |
| Overall Success Rate | 11.61% |
| Total Success | 235/2024 |

### Top 10 Tasks

| Rank | Task | Score | SR |
|------|------|-------|-----|
| 1 | put_bottles_into_dustbin | 88.0 | 82.0% |
| 2 | fold_clothes | 80.0 | 76.0% |
| 3 | stack_bowls | 76.2 | 72.0% |
| 4 | build_tower | 62.8 | 52.0% |
| 5 | make_kong | 58.0 | 58.0% |
| 6 | pour_liquid_into_cup | 48.0 | 48.0% |
| 7 | play_tic_tac_toe | 47.8 | 12.0% |
| 8 | insert_tubes | 47.4 | 31.6% |
| 9 | fold_clothes_random | 40.0 | 32.0% |
| 10 | organize_table | 30.5 | 4.0% |

## Important Parameters

Common parameter meanings used across the commands above:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_uv_env` | `uv` to use `deploy.yml` `policy_uv_env_path`, or an explicit project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_num` | Runtime or checkpoint option consumed by this adapter. |
| `result_dir` | Runtime or checkpoint option consumed by this adapter. |
| `obs_transform_pipeline` | Runtime or checkpoint option consumed by this adapter. |
| `policy_uv_env_path` | Runtime or checkpoint option consumed by this adapter. |
| `train_config_name` | Runtime or checkpoint option consumed by this adapter. |
| `repo_id` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `CONDA_BASE` | Optional override used by the local scripts or upstream runtime. |
| `GIT_LFS_SKIP_SMUDGE` | Optional override used by the local scripts or upstream runtime. |
| `HF_DATASETS_CACHE` | Optional override used by the local scripts or upstream runtime. |
| `JAX_COMPILATION_CACHE_DIR` | Optional override used by the local scripts or upstream runtime. |
| `LOCAL_CACHE_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_DATA_MODE` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to training. |
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_SRC` | Optional override used by the local scripts or upstream runtime. |
| `OPENPI_TRAIN_CONFIG_NAME` | Optional override used by the local scripts or upstream runtime. |
| `POLICY_DIR` | Optional override used by the local scripts or upstream runtime. |
| `PYENV` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
