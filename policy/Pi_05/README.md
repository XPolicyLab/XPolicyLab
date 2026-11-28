# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

### RLinf training environment

Run the following once from the **XPolicyLab root** on Linux with CUDA and `uv` installed. The example creates `.venv` for training, installs the bundled OpenPI dependencies, and selects RLinf’s PyTorch 2.11 CUDA build. For RoboTwin, also install the simulator dependencies in the training environment. [RLinf’s RoboTwin setup](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html)

```bash
git submodule update --init infra/RLinf/RLinf
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" GIT_LFS_SKIP_SMUDGE=1 uv sync --project policy/Pi_05/openpi --locked --inexact --group lerobot
python -m pip install -e . "numpy<2"
bash infra/RLinf/scripts/install_rl_runtime.sh --upgrade-torch
```

For subsequent training runs, activate `.venv` from the XPolicyLab root with `source .venv/bin/activate`.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The dataset uses the official keys — `observation.state`, `action`, `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)); the bundled script exists because conversion must run inside openpi's own pinned LeRobot environment, which sets the dataset version. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).

### RLinf PyTorch supervised fine-tuning (SFT)

From the XPolicyLab root, activate the RLinf training environment. Set `POLICY_MODEL_PATH` to your initial Pi0.5 PyTorch checkpoint (or pass `actor.model.model_path` on the command line). Set `PI05_NORM_STATS_PATH` to the normalization file for the training data. Adjust the example path if your file is stored elsewhere.

```bash
export PI05_NORM_STATS_PATH="$POLICY_MODEL_PATH/assets/robodojo_stack_bowls/norm_stats.json"
bash infra/RLinf/scripts/train_pi05_sft.sh RoboDojo stack_bowls arx_x5 joint 0 '<gpu-ids>'
```

Append `--dry-run` to inspect the configuration, or `--compute-norm-stats` to compute the dataset statistics on CPU and write them to a new `norm_stats.json` at `PI05_NORM_STATS_PATH`. SFT checkpoints go under `policy/Pi_05/checkpoints/pytorch_sft/`.

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

In `deploy.yml`, `train_config_name` selects the named OpenPI recipe used when training the checkpoint. For example, `pi05_base_aloha_full_sim_arx-x5_seed_0` is registered in the bundled OpenPI configuration. The `seed` argument on the training command sets the seed for the current run. Set `repo_id` to the directory containing `norm_stats.json` under the checkpoint’s `assets/`, and `checkpoint_num` to the training step to evaluate. [OpenPI training recipes](openpi/src/openpi/training/config.py)

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.

## RLinf v0.4 backend

To run PPO or GRPO on RoboDojo or RoboTwin, follow the [RLinf setup and launch guide](../../infra/RLinf/doc/GUIDE.md).

For standalone evaluation of RL weights, select `rl_backend: rlinf_v0_4` and supply model weights and matching normalization statistics.
