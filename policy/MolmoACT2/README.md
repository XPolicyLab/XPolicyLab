# MolmoACT2

**Contributor:** RoboDojo Team | **Paper:** MolmoAct2: Action Reasoning Models for Real-world Deployment | **arXiv:** https://arxiv.org/abs/2605.02881 | **Original code:** https://github.com/allenai/molmoact2

`MolmoACT2` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `INSTALLATION.md` | Required supplemental installation guide for assets, system dependencies, or multi-environment setup. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `deploy/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Read `INSTALLATION.md` before first use. It is intentionally kept because this policy has setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, or multi-environment runtime notes.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Conda environment name, or `uv` / a uv project path for the policy runtime. |

```bash
cd XPolicyLab/policy/MolmoACT2
# Example: install dependencies for the MolmoACT2 policy adapter.
bash install.sh
# `setup_eval_policy_server.sh` accepts `uv`, a uv project path, or a conda env.
source molmoact2/lerobot/.venv/bin/activate  # or pass `uv` as <policy_env>
```

## Model Weights

The base checkpoint is the fine-tuning starting point loaded by `train.sh`.

| Checkpoint | Use |
|---|---|
| `allenai/MolmoAct2` | Base checkpoint for RoboDojo fine-tuning (Qwen2.5-7B backbone + flow-matching action expert, `add_action_expert=true`, `max_action_dim=32`). |

`MOLMOACT2_CHECKPOINT_PATH` accepts a local directory or a Hub repo id and defaults to `allenai/MolmoAct2`. `train.sh` resolves it automatically: an existing local directory is used as-is, otherwise the checkpoint is downloaded from the Hub on first run (network required). No manual download step is needed.

- To reuse a local copy, point `MOLMOACT2_CHECKPOINT_PATH` at a directory whose `config.json` has `"model_type": "molmoact2"`.
- Concrete shared weight/dataset paths for this cluster are listed in `../POLICY_TRAINING_COMMANDS.md`.

## Demo Data Processing

What it does: prepares RoboDojo demonstration data for policy training. The output name should match the training run identity so `train.sh` can find it.

This adapter has no top-level `process_data.sh`. It expects data in the format consumed by the upstream project or by `deploy.yml`/environment variables. Use the upstream README under the vendored source tree when custom conversion is required.

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
| `gpu_id` | GPU id or comma-separated GPU ids for the policy trainer. |

```bash
cd XPolicyLab/policy/MolmoACT2
source molmoact2/lerobot/.venv/bin/activate

# Dataset is required; checkpoint defaults to allenai/MolmoAct2 (auto-downloaded).
# See ../POLICY_TRAINING_COMMANDS.md for concrete cluster paths.
export MOLMOACT2_DATASET_ROOT=<lerobot_data_root>/<dataset_repo_id>
export MOLMOACT2_DATASET_REPO_ID=<dataset_repo_id>
# Optional: reuse a local base checkpoint instead of the Hub default.
# export MOLMOACT2_CHECKPOINT_PATH=<model_weights_dir>/MolmoAct2

# Template: train a policy run on one GPU or a GPU list.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: train the same run on four GPUs if the upstream trainer supports it.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

The usual checkpoint directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. During evaluation, `ckpt_name` may be the short run name from training (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory.

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
| `policy_conda_env` | Conda env, `uv`, or a uv project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/MolmoACT2
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
| `policy_conda_env` | Conda env, `uv`, or a uv project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=joint`. |

```bash
cd XPolicyLab/policy/MolmoACT2
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
| `policy_conda_env` | Conda env, `uv`, or a uv project path for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_num` | Runtime or checkpoint option consumed by this adapter. |
| `inference_action_mode` | Runtime or checkpoint option consumed by this adapter. |
| `policy_uv_env_path` | Runtime or checkpoint option consumed by this adapter. |
| `device` | Runtime or checkpoint option consumed by this adapter. |
| `actions_per_chunk` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `CODEBASE_VER` | Optional override used by the local scripts or upstream runtime. |
| `COMMON_ARGS` | Optional override used by the local scripts or upstream runtime. |
| `CONDA_BASE` | Optional override used by the local scripts or upstream runtime. |
| `CONDA_PREFIX` | Optional override used by the local scripts or upstream runtime. |
| `CONTROL_MODE` | Optional override used by the local scripts or upstream runtime. |
| `DEPLOY_PROXY_HOST` | Optional override used by the local scripts or upstream runtime. |
| `DEPLOY_PROXY_PORT` | Optional override used by the local scripts or upstream runtime. |
| `GLOBAL_BATCH_SIZE` | Optional override used by the local scripts or upstream runtime. |
| `GPU_ARR` | Optional override used by the local scripts or upstream runtime. |
| `HF_DATASETS_CACHE` | Optional override used by the local scripts or upstream runtime. |
| `HTTPS_PROXY` | Optional override used by the local scripts or upstream runtime. |
| `HTTP_PROXY` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Training accepts a short run id such as `cotrain`; evaluation should use the full checkpoint directory name such as `RoboDojo-cotrain-arx_x5-joint-0`, or an explicit absolute/relative checkpoint path.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
