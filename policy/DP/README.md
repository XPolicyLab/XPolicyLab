# DP

**Contributor:** RoboDojo Team | **Paper:** Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | **arXiv:** https://arxiv.org/abs/2303.04137 | **Original code:** https://github.com/real-stanford/diffusion_policy

`DP` adapts Diffusion Policy (visuomotor policy learning via action diffusion) to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `diffusion_policy/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
conda create -n dp python=3.10 -y
conda activate dp
cd XPolicyLab/policy/DP
bash install.sh
```

## Data Processing

Reads raw demos from `data/<bench_name>/<ckpt_name>/<env_cfg_type>` and produces the zarr dataset `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>.zarr` consumed by `train.sh`. The only extra argument is the optional `[expert_data_num]` episode limit:

```bash
cd XPolicyLab/policy/DP
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: convert only the first 50 episodes from data/RoboDojo/stack_bowls/arx_x5
bash process_data.sh RoboDojo stack_bowls arx_x5 joint 50
```

## Training

```bash
cd XPolicyLab/policy/DP
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id such as 0,1,2,3 if the upstream trainer supports it)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. `train.sh` derives the action dimension from `env_cfg_type` (via `utils/get_action_dim.sh`) and expects the matching zarr dataset from `process_data.sh` under `data/`.

## Evaluation

```bash
cd XPolicyLab/policy/DP
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`.
