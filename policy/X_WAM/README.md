# X_WAM

**Contributor:** RoboDojo Team | **Paper:** X-WAM: Cross-Embodiment World Action Model | **arXiv:** https://arxiv.org/abs/2604.26694 | **Original code:** https://github.com/sharinka0715/X-WAM

`X_WAM` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `INSTALLATION.md` | Required supplemental installation guide for assets, system dependencies, or multi-environment setup. |
| `process_data.sh` | Converts RoboDojo demonstration data into the policy-specific training format. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `X-WAM/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Read `INSTALLATION.md` before first use. It is intentionally kept because this policy has setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, or multi-environment runtime notes.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Name of the conda environment used by the policy runtime. |

```bash
cd XPolicyLab/policy/X_WAM
# Example: this adapter has no top-level install wrapper.
# Install the vendored upstream project, then keep this directory on PYTHONPATH.
```

## Demo Data Processing

What it does: prepares RoboDojo demonstration data for policy training. The output name should match the training run identity so `train.sh` can find it.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Data/run identifier. Use a different value for ablations, for example `stack_bowls_50ep`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation. X-WAM only supports `ee`. |
| `limit` | Optional episode limit. Leave unset to use all episodes. |
| `raw_task_dirs` | Optional source task directory or comma-separated task list when the script supports it. |

```bash
cd XPolicyLab/policy/X_WAM
# Template: convert all available demonstrations for one run.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: convert stack_bowls demos for arx_x5 EE control.
bash process_data.sh RoboDojo stack_bowls arx_x5 ee

# Example: create a 50-episode ablation while reading from the original task data.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 ee 50 stack_bowls
```

## Model Training

What it does: starts the policy-specific training recipe through the XPolicyLab wrapper and writes checkpoints under this adapter directory.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Training run identifier, for example `cotrain`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation. X-WAM only supports `ee`. |
| `seed` | Random seed. |
| `gpu_id` | GPU id or comma-separated GPU ids for the policy trainer. |
| `num_gpus` | Optional explicit process count; inferred from comma-separated `gpu_id` when omitted. |

```bash
cd XPolicyLab/policy/X_WAM
# Template: train a policy run on one GPU or a GPU list.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 using stack_bowls raw demonstrations if converted data is missing.
XWAM_RAW_TASK_DIRS=stack_bowls bash train.sh RoboDojo cotrain arx_x5 ee 0 0

# Example: train the same run on four GPUs if the upstream trainer supports it.
XWAM_RAW_TASK_DIRS=stack_bowls bash train.sh RoboDojo cotrain arx_x5 ee 0 0,1,2,3
```

The usual checkpoint directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Pass that full directory name as `ckpt_name` during evaluation.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client. Use `eval.sh` for a same-machine smoke test, or split server/client scripts for debugging and multi-machine evaluation.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation. X-WAM only supports `ee`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/X_WAM
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-ee-0 arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

Parameters used by the split server/client flow:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation. X-WAM only supports `ee`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=ee`. |

```bash
cd XPolicyLab/policy/X_WAM
# Terminal 1 on the policy machine: start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_conda_env> <policy_server_port> <policy_server_host>

# Example: bind the policy server to all interfaces on port 5000.
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-ee-0 arx_x5 ee 0 \
  0 <policy_conda_env> 5000 0.0.0.0

# Terminal 2 on the environment machine: connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>

# Example: connect to a policy server reachable at <policy_server_ip>:5000.
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-ee-0 arx_x5 ee 0 \
  0 <eval_env_conda_env> "ckpt_name=RoboDojo-cotrain-arx_x5-ee-0,action_type=ee" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks when the adapter supports it; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Important Parameters

Common parameter meanings used across the commands above:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation. X-WAM only supports `ee`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `action_dim` | Runtime or checkpoint option consumed by this adapter. |
| `exp_path` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_path` | Runtime or checkpoint option consumed by this adapter. |
| `wan_checkpoint_dir` | Runtime or checkpoint option consumed by this adapter. |
| `steps` | Runtime or checkpoint option consumed by this adapter. |
| `device` | Runtime or checkpoint option consumed by this adapter. |
| `compile_model` | Runtime or checkpoint option consumed by this adapter. |
| `denoise_steps` | Runtime or checkpoint option consumed by this adapter. |
| `action_denoise_steps` | Runtime or checkpoint option consumed by this adapter. |
| `cfg` | Runtime or checkpoint option consumed by this adapter. |
| `replan_steps` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables:

| Variable | Notes |
|---|---|
| `XWAM_RAW_DATA_ROOT` | Root containing raw benchmark directories; defaults to `<repo>/final_data`. |
| `XWAM_RAW_INPUT_DIR` | Direct raw input root containing `<task>/<env_cfg_type>/data/*.hdf5`; bypasses staging. |
| `XWAM_RAW_TASK_DIRS` | Comma-separated raw task dirs used by `train.sh` auto-conversion when data is missing. |
| `XWAM_DATASET_PATH` | Converted dataset directory used by `process_data.sh` and `train.sh`. |
| `XWAM_DATA_LIMIT` | Optional episode limit used by `train.sh` auto-conversion. |
| `XWAM_TRANSFORM_WORKERS` | Worker count for `transform_robodojo_to_xwam.py`; defaults to `16`. |
| `XWAM_WAN_CHECKPOINT_DIR` | Wan2.2-TI2V-5B base weight directory. |
| `XWAM_PRETRAINED_CHECKPOINT` | Optional pretrained X-WAM checkpoint for training initialization. |
| `XWAM_EXP_PATH` | Evaluation experiment directory containing `config.yaml` and `checkpoints/`. |
| `XWAM_CKPT_ROOT` | Evaluation checkpoint root used with `XWAM_EXP_SETTING` when `XWAM_EXP_PATH` is unset. |
| `XWAM_EXP_SETTING` | Evaluation experiment directory name; defaults to the eval `ckpt_name`. |
| `XWAM_STEPS` | Evaluation checkpoint step; defaults to `last`. |
| `XWAM_ALLOW_DUMMY_POLICY` | Set to `true` to skip checkpoint loading for protocol debugging. |
| `EVAL_ENV_TYPE` | Evaluation client mode: unset/`sim`, `debug`, or `real`. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
