# EventVLA

**Contributor:** RoboDojo Team | **Paper:** EventVLA technical report | **arXiv:** TBD | **Original code:** See vendored `source_eventvla/`.

`EventVLA` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

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
| `results/` | Vendored upstream code, policy-specific assets, or helper scripts. |
| `source_eventvla/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Name of the conda environment used by the policy runtime. |

```bash
cd XPolicyLab/policy/EventVLA
# Example: install dependencies for the EventVLA policy adapter.
bash install.sh
# Example: activate the environment used later as <policy_conda_env>.
conda activate <policy_env>  # e.g. eventvla
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
cd XPolicyLab/policy/EventVLA
# Template: convert all available demonstrations for one run.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: convert stack_bowls demos for arx_x5 joint control.
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data.
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Model Training

What it does: starts the EventVLA upstream training recipe and writes results under `results/Checkpoints/<RUN_ID>/`. The printed `RUN_ID` is the value to pass as eval `ckpt_name`.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `data_mix` | Upstream EventVLA data mix name, for example `robodojo`. |
| `memory_ablation_mode` | Memory ablation/profile name, for example `pure_image_keyframe_memory`. |
| `keyframe_memory_policy` | Keyframe memory policy. Supported values are `teacher` and `predict` aliases. |
| `extra_args` | Optional arguments forwarded to the upstream EventVLA train script. |
| `RUN_ID` | Optional environment override for the run directory; this becomes eval `ckpt_name`. |

```bash
cd XPolicyLab/policy/EventVLA
# Template: launch EventVLA training.
bash train.sh <data_mix> <memory_ablation_mode> <keyframe_memory_policy> [extra_args...]

# Example: train with teacher keyframe memory.
bash train.sh robodojo pure_image_keyframe_memory teacher

# Example: force a stable run id that can be reused as eval ckpt_name.
RUN_ID=RoboDojo-eventvla-arx_x5-joint-0   bash train.sh robodojo pure_image_keyframe_memory teacher
```

Evaluate with `ckpt_name=<RUN_ID>`. EventVLA stores checkpoints under `results/Checkpoints/<RUN_ID>/`, not the generic `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/` layout.

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
cd XPolicyLab/policy/EventVLA
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
cd XPolicyLab/policy/EventVLA
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
| `eventvla_root` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_path` | Runtime or checkpoint option consumed by this adapter. |
| `eventvla_server_host` | Runtime or checkpoint option consumed by this adapter. |
| `eventvla_server_port` | Runtime or checkpoint option consumed by this adapter. |
| `unnorm_key` | Runtime or checkpoint option consumed by this adapter. |
| `action_mode` | Runtime or checkpoint option consumed by this adapter. |
| `use_ddim` | Runtime or checkpoint option consumed by this adapter. |
| `num_ddim_steps` | Runtime or checkpoint option consumed by this adapter. |
| `image_size` | Runtime or checkpoint option consumed by this adapter. |
| `include_state` | Runtime or checkpoint option consumed by this adapter. |
| `temporal_absolute_indices` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `DATA_SUBDIR` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_TRAIN_SCRIPT` | Optional override used by the local scripts or upstream runtime. |
| `DOWNLOAD_PATH` | Optional override used by the local scripts or upstream runtime. |
| `DOWNLOAD_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_CKPT_PATH` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_DATA_MIX` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_DATA_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_MEMORY_ABLATION_MODE` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_ROOT` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_RUN_ROOT_DIR` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_SERVER_PID` | Optional override used by the local scripts or upstream runtime. |
| `EVENTVLA_SERVER_READY_TIMEOUT` | Optional override used by the local scripts or upstream runtime. |

## Notes

- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
