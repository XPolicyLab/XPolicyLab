# RISE

**Contributor:** RoboDojo Team | **Paper:** RISE: OpenDriveLab robot policy report | **arXiv:** https://arxiv.org/abs/2602.11075 | **Original code:** https://github.com/OpenDriveLab/RISE

`RISE` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `INSTALLATION.md` | Required supplemental installation guide for assets, system dependencies, or multi-environment setup. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `process_data.sh` | Converts RoboDojo demonstration data into the policy-specific training format. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `data/` | Vendored upstream code, policy-specific assets, or helper scripts. |
| `RISE/` | Vendored upstream code, policy-specific assets, or helper scripts. |
| `weights/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Read `INSTALLATION.md` before first use. It is intentionally kept because this policy has setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, or multi-environment runtime notes.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Name of the conda environment used by the policy runtime. |

```bash
cd XPolicyLab/policy/RISE
# Example: install dependencies for the RISE policy adapter.
bash install.sh
# Example: activate the environment used later as <policy_conda_env>.
conda activate <policy_env>  # e.g. rise
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
| `expert_data_num` | Optional episode limit. Leave unset to use all episodes. |
| `raw_task_dirs` | Optional source task directory or comma-separated task list when the script supports it. |

```bash
cd XPolicyLab/policy/RISE
# Template: convert all available demonstrations for one run.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: convert stack_bowls demos for arx_x5 joint control.
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Model Training

What it does: runs the RISE offline training flow. RISE has stages: `advantage`, `policy`, or `all`. The `policy` stage expects an existing advantage dataset (`*_w_adv`); use `all` to run advantage then policy.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Training run identifier. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Random seed. |
| `gpu_id` | GPU id or comma-separated GPU ids. |
| `stage` | Optional `advantage`, `policy`, or `all`; default is `policy` unless `RISE_STAGE` overrides it. |
| `extra_args` | Optional arguments forwarded to the upstream stage script. |

```bash
cd XPolicyLab/policy/RISE
# Template: train a specific RISE stage.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [advantage|policy|all] [extra_args...]

# Example: compute advantage artifacts.
bash train.sh RoboDojo stack_bowls arx_x5 joint 42 0 advantage

# Example: train the final policy after advantage data exists.
bash train.sh RoboDojo stack_bowls arx_x5 joint 42 0 policy

# Example: run advantage then policy.
bash train.sh RoboDojo stack_bowls arx_x5 joint 42 0 all
```

Legacy stage-first usage is also supported by the script: `bash train.sh <advantage|policy|all> <gpu_id> <seed> [extra_args...]`, with dataset fields supplied through `RISE_BENCH_NAME`, `RISE_CKPT_NAME`, `RISE_ENV_CFG_TYPE`, and `RISE_ACTION_TYPE`.

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
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/RISE
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
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
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=joint`. |

```bash
cd XPolicyLab/policy/RISE
# Terminal 1 on the policy machine: start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_conda_env> <policy_server_port> <policy_server_host>

# Example: bind the policy server to all interfaces on port 5000.
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 <policy_conda_env> 5000 0.0.0.0

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
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `action_dim` | Runtime or checkpoint option consumed by this adapter. |
| `upstream_dir` | Runtime or checkpoint option consumed by this adapter. |
| `config_name` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_path` | Runtime or checkpoint option consumed by this adapter. |
| `default_prompt` | Runtime or checkpoint option consumed by this adapter. |
| `sample_data_dir` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `ADAPTER_DIR` | Optional override used by the local scripts or upstream runtime. |
| `ADV_ASSET_ID` | Optional override used by the local scripts or upstream runtime. |
| `ADV_DATASET` | Optional override used by the local scripts or upstream runtime. |
| `ADV_NORM_DIR` | Optional override used by the local scripts or upstream runtime. |
| `ADV_NORM_PATH` | Optional override used by the local scripts or upstream runtime. |
| `CLEANUP` | Optional override used by the local scripts or upstream runtime. |
| `CONDA_PREFIX` | Optional override used by the local scripts or upstream runtime. |
| `CONVERTED_DATASET` | Optional override used by the local scripts or upstream runtime. |
| `DATA_DIR` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_PI05_WEIGHTS` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_RAW_DATASET_LINK` | Optional override used by the local scripts or upstream runtime. |
| `INSTALLATION` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
