# H_RDT

**Contributor:** RoboDojo Team | **Paper:** H-RDT: Hybrid Robot Diffusion Transformer | **arXiv:** https://arxiv.org/abs/2507.23523 | **Original code:** https://github.com/embodiedfoundation/H-RDT

`H_RDT` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `data/` | Vendored upstream code, policy-specific assets, or helper scripts. |
| `H_RDT/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Name of the conda environment used by the policy runtime. |

```bash
cd XPolicyLab/policy/H_RDT
# Example: install dependencies for the H_RDT policy adapter.
bash install.sh
# Example: activate the environment used later as <policy_conda_env>.
conda activate <policy_env>  # e.g. h-rdt
```

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
| `pretrained_backbone_path` | Optional pretrained H-RDT backbone path; defaults to the vendored pretrain checkpoint path. |

```bash
cd XPolicyLab/policy/H_RDT
# Template: train a policy run on one GPU or a GPU list.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: train the same run on four GPUs if the upstream trainer supports it.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
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
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/H_RDT
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
cd XPolicyLab/policy/H_RDT
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
| `checkpoint_path` | Runtime or checkpoint option consumed by this adapter. |
| `config_path` | Runtime or checkpoint option consumed by this adapter. |
| `lang_embedding_path` | Runtime or checkpoint option consumed by this adapter. |
| `lang_embedding_dir` | Runtime or checkpoint option consumed by this adapter. |
| `stats_path` | Runtime or checkpoint option consumed by this adapter. |
| `device` | Runtime or checkpoint option consumed by this adapter. |
| `dtype` | Runtime or checkpoint option consumed by this adapter. |
| `input_color_order` | Runtime or checkpoint option consumed by this adapter. |
| `vision_backbone_id` | Runtime or checkpoint option consumed by this adapter. |
| `vision_image_size` | Runtime or checkpoint option consumed by this adapter. |
| `allow_dummy_lang_embedding` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `DEMO_ENV_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `HF_ENDPOINT` | Optional override used by the local scripts or upstream runtime. |
| `HF_HOME` | Optional override used by the local scripts or upstream runtime. |
| `HRDT_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `HUGGINGFACE_HUB_CACHE` | Optional override used by the local scripts or upstream runtime. |
| `H_RDT` | Optional override used by the local scripts or upstream runtime. |
| `IMREAD_COLOR` | Optional override used by the local scripts or upstream runtime. |
| `PYTHONWARNINGS` | Optional override used by the local scripts or upstream runtime. |
| `TASK_ENV` | Optional override used by the local scripts or upstream runtime. |
| `TRANSFORMERS_CACHE` | Optional override used by the local scripts or upstream runtime. |
| `WANDB_PROJECT` | Optional override used by the local scripts or upstream runtime. |
| `XPOLICY_HRDT_ACTION_TYPE` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
