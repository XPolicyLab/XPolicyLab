# FastWAM

**Contributor:** RoboDojo Team | **Paper:** Fast-WAM: Do World Action Models Need Test-time Future Imagination? | **arXiv:** https://arxiv.org/abs/2603.16666 | **Original code:** https://github.com/yuantianyuan01/FastWAM

`FastWAM` adapts the Fast-WAM world-action model to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `FastWAM/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use; it covers setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, and multi-environment runtime notes.

```bash
cd XPolicyLab/policy/FastWAM
bash install.sh
conda activate <policy_env>  # e.g. fastwam
```

## Data Processing

No top-level `process_data.sh`. Training consumes a prepared LeRobot v2.1 dataset directly (see Training below); use the upstream FastWAM tooling under `FastWAM/` for data preparation.

## Training

```bash
cd XPolicyLab/policy/FastWAM
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

# Example: train a cotrain run on four GPUs (the upstream model is large; multi-GPU is recommended)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory. Training expects a prepared LeRobot v2.1 dataset under `data/<dataset_id>/lerobot/` with `data/<dataset_id>/dataset_stats.json`, a matching T5 text embedding cache under `FastWAM/data/text_embeds_cache/xpolicylab/<dataset_id>/`, and the ActionDiT backbone at `FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. `<dataset_id>` defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` (override with `FASTWAM_DATASET_ID`), and `train.sh` prints the upstream commands that generate the text cache and backbone when they are missing. The process count is inferred from a comma-separated `gpu_id` unless passed as the optional 7th argument `num_gpus`. The wrapper defaults to `FASTWAM_BATCH_SIZE=8`; you may still need to lower it or increase the GPU count depending on available memory. `train.sh` sets `DIFFSYNTH_MODEL_BASE_PATH` to `FastWAM/checkpoints` automatically.

## Evaluation

```bash
cd XPolicyLab/policy/FastWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `action_dim`, `checkpoint_path`, `dataset_stats_path`, `sim_cfg_name`, `sim_task`, `device`, `mixed_precision`, `action_horizon`, `replan_steps`, `num_inference_steps`, `sigma_shift`.

Optional policy-specific environment overrides used by the scripts: `FASTWAM_DATASET_ID`, `FASTWAM_BATCH_SIZE`, `FASTWAM_GRADIENT_ACCUMULATION_STEPS`, `FASTWAM_NUM_WORKERS`, `FASTWAM_NUM_EPOCHS`, `FASTWAM_CKPT_SETTING`, `FASTWAM_CKPT_ROOT`, `FASTWAM_CHECKPOINT_PATH`, `FASTWAM_DATASET_STATS_PATH`, `FASTWAM_ALLOW_DUMMY_POLICY`, `FASTWAM_SEQUENTIAL_AUX_OFFLOAD`, `FASTWAM_WS_REQUEST_TIMEOUT_S`, `FASTWAM_WS_PING_INTERVAL_S`, `FASTWAM_WS_PING_TIMEOUT_S`.

Set `FASTWAM_SEQUENTIAL_AUX_OFFLOAD=true` when policy inference and the simulator share a 24 GiB GPU. It is disabled by default. FastWAM client timeouts default to 600 seconds per request, a 60-second ping interval, and a 300-second ping timeout; override them with the corresponding `FASTWAM_WS_*` variables above.

## Notes

- Use the same `action_type` for training and evaluation. The reference FastWAM path follows XPolicyLab's `pack_robot_state` / `unpack_robot_state` helpers directly and does not add policy-local `ee` pose conversion.
