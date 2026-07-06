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

What it does: downloads the upstream EventVLA LeRobot dataset and links it as local training data. EventVLA does not convert per-task RoboDojo demos in this wrapper.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Accepted for the standard XPolicyLab interface and logged for traceability. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `expert_data_num` | Optional compatibility argument; EventVLA ignores it and uses the upstream dataset as a whole. |

```bash
cd XPolicyLab/policy/EventVLA
# Template: fetch and link the upstream training dataset.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example: prepare the upstream EventVLA training dataset for arx_x5 joint control.
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

## Model Training

What it does: starts the EventVLA upstream training recipe and writes results under `results/Checkpoints/<RUN_ID>/`. The printed `RUN_ID` is the value to pass as eval `ckpt_name`.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `data_mix` | Upstream EventVLA data mix name, for example `robodojo`. |
| `memory_ablation_mode` | Memory ablation/profile name, for example `pure_image_keyframe_memory`. |
| `keyframe_memory_policy` | Keyframe memory policy. Supported values are `teacher` and `predict` aliases. |
| `data_root_dir` | Optional first extra argument, used as the training data root unless `EVENTVLA_DATA_ROOT` is set. |
| `train_args` | Optional remaining arguments appended to `eventvla/training/train_eventvla.py`, for example `--trainer.max_train_steps 20000`. |
| `RUN_ID` | Optional environment override for the run directory; this becomes eval `ckpt_name`. |

```bash
cd XPolicyLab/policy/EventVLA
# Template: launch EventVLA training.
bash train.sh <data_mix> <memory_ablation_mode> <keyframe_memory_policy> [data_root_dir] [train_args...]

# Example: train with teacher keyframe memory.
bash train.sh robodojo pure_image_keyframe_memory teacher

# Example: force a stable run id that can be reused as eval ckpt_name.
RUN_ID=RoboDojo-eventvla-arx_x5-joint-0   bash train.sh robodojo pure_image_keyframe_memory teacher

# Example: override an upstream trainer option.
bash train.sh robodojo pure_image_keyframe_memory teacher --trainer.max_train_steps 20000
```

Evaluate with `ckpt_name=<RUN_ID>`. EventVLA stores checkpoints under `results/Checkpoints/<RUN_ID>/`, not the generic `checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/` layout.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client. Use `eval.sh` for a same-machine smoke test, or split server/client scripts for debugging and multi-machine evaluation.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | EventVLA run directory name, usually the `RUN_ID` printed by `train.sh`. |
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

# Example: evaluate a trained EventVLA run on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-eventvla-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

Parameters used by the split server/client flow:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | EventVLA run directory name, usually the `RUN_ID` printed by `train.sh`. |
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
  RoboDojo stack_bowls RoboDojo-eventvla-arx_x5-joint-0 arx_x5 joint 0 \
  0 <policy_conda_env> 5000 0.0.0.0

# Terminal 2 on the environment machine: connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>

# Example: connect to a policy server reachable at <policy_server_ip>:5000.
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls RoboDojo-eventvla-arx_x5-joint-0 arx_x5 joint 0 \
  0 <eval_env_conda_env> "ckpt_name=RoboDojo-eventvla-arx_x5-joint-0,action_type=joint" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks when the adapter supports it; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Important Parameters

Common parameter meanings used across the commands above:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | EventVLA run directory name, usually the `RUN_ID` printed by `train.sh`. |
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

Frequently used environment variables:

| Variable | Notes |
|---|---|
| `RUN_ID` | Overrides the training run directory name and eval `ckpt_name`. |
| `EVENTVLA_RUN_ROOT_DIR` | Overrides the wrapper training output root, default `policy/EventVLA/results/Checkpoints`. |
| `EVENTVLA_DATA_ROOT` | Overrides both `process_data.sh` download root and training data root. |
| `EVENTVLA_TRAIN_SCRIPT` | Overrides the upstream training entry script. |
| `EVENTVLA_CKPT_PATH` | Bypasses run-directory checkpoint lookup during eval. |
| `EVENTVLA_SERVER_READY_TIMEOUT` | Timeout, in seconds, while waiting for the upstream EventVLA server. |
| `BASE_VLM` | Overrides the base Qwen/VLM path used by the upstream training recipe. |
| `MAX_KEYFRAME_IMAGES` | Overrides the upstream keyframe memory image count. |
| `KEEP_RECENT_CHECKPOINTS` | Overrides how many step checkpoints the upstream trainer keeps. |

## Notes

- Keep the training `RUN_ID` stable and pass it as eval `ckpt_name`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
