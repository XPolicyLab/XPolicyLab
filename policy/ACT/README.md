# ACT

**Contributor:** RoboDojo Team | **Paper:** Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware | **arXiv:** https://arxiv.org/abs/2304.13705 | **Original code:** https://github.com/tonyzhaozh/act

`ACT` adapts Action Chunking with Transformers (ACT) to XPolicyLab/RoboDojo. The upstream implementation is vendored at this directory level (`imitate_episodes.py`, `constants.py`, `utils.py`), with the DETR-based model code in `detr/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/ACT
bash install.sh
conda activate <policy_env>  # e.g. act
```

## Data Processing

`ckpt_name` selects the raw data directory under `data/<bench_name>/` and the generated training config, so use the same value when running `train.sh`:

```bash
cd XPolicyLab/policy/ACT
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

# Example
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

## Training

```bash
cd XPolicyLab/policy/ACT
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id such as 0,1,2,3 if the upstream trainer supports it)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. `train.sh` derives the action dimension from `env_cfg_type` (via `utils/get_action_dim.sh`) and exports it as `ACT_ACTION_DIM` before launching `imitate_episodes.py`.

## Evaluation

```bash
cd XPolicyLab/policy/ACT
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

`eval_batch` is `false` and must stay that way: ACT is stateful — temporal aggregation keeps the per-instance buffer `all_time_actions` and the step counter `t`, both cleared by `reset()` — so one policy-server instance can serve exactly one active environment. To run concurrent rollouts, start one policy server per rollout.

`update_obs_batch` and `get_action_batch` raise `NotImplementedError`, but that is not a safety net: the simulator's batch mode drives every worker through the single-env `update_obs` / `get_action` pair against the one shared server, so a multi-worker run degrades silently instead of failing.

## Configuration

`deploy.yml` keys to check before evaluation: `ckpt_setting`, `kl_weight`, `chunk_size`, `hidden_dim`, `dim_feedforward`, `temporal_agg`, `device`, `ckpt_dir`, `policy_class`, `num_epochs`, `position_embedding`, `lr_backbone`.

`ACT_ACTION_DIM` is the only adapter-specific environment variable; `train.sh` and `setup_eval_policy_server.sh` set it automatically from `env_cfg_type`, and the model code reads it for the state/action dimension.
