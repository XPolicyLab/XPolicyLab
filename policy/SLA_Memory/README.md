# SLA_Memory

**Contributor:** Qizhou Huang | **Paper:** SLA: Scaling Spatial-Language-Action Model via Continual Pre-training | **arXiv:** [2611.08373](https://arxiv.org/abs/2611.08373) | **Original code:** [SLA](https://github.com/OpenHelix-Team/Spatial-Forcing/tree/master/SLA)

`SLA_Memory` adapts SLA, a π0.5-based vision-language-action model, to XPolicyLab/RoboDojo with end-effector actions. Besides the three current camera views, the model sees nine history frames from the head camera: the episode's first frame and one frame per second over the last 8 seconds. The adapter covers RoboDojo simulation (`arx_x5`) and the RoboDojo real robots (`piper_x`, `arx_x5`, `piper`). The vendored implementation lives in `openpi/`, a trimmed fork of [openpi](https://github.com/Physical-Intelligence/openpi) that also vendors a patched lerobot in `openpi/packages/lerobot`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/SLA_Memory
bash install.sh
```

`install.sh` requires `uv` on `PATH`. It builds the policy environment in `openpi/.venv` from the lock file (`uv sync --frozen`) and installs XPolicyLab into it. There is no conda environment for the policy: pass `uv` as `<policy_uv_env>` to `eval.sh` (it uses `policy_uv_env_path` from `deploy.yml`) or the path of an `openpi` project.

## Data Processing

The training data is converted from the official HDF5 release, downloaded with XPolicyLab's `scripts/RoboDojo/download_robodojo_data.sh hdf5` (simulation) or `... real` (real robots) into `<XPolicyLab parent>/data/RoboDojo` and `<XPolicyLab parent>/data/RoboDojo_real`:

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: every simulation task, into one dataset
bash process_data.sh RoboDojo sim arx_x5 ee
# Example: every real-robot task for all three robots, one dataset per task and robot
bash process_data.sh RoboDojo_real real all ee
```

Output, LeRobot v2.1, in `Data/` inside the policy directory:

- `RoboDojo` → `Data/RoboDojo_sim_ee`, all tasks of `env_cfg_type` in one dataset.
- `RoboDojo_real` → `Data/RoboDojo_real_ee/RoboDojo_real_<task>_<robot>`. `env_cfg_type` may be one robot or `all`.

The keys differ from the official converters, which export joint angles. `observation.state` and `action` are 16-dim end-effector poses, `[x, y, z, qw, qx, qy, qz, gripper]` for the left arm, then the right arm. `action` is the HDF5 action. Images are `observation.images.cam_high`, `cam_left_wrist` and `cam_right_wrist` at 480×640. The frame rate comes from the HDF5 files: 25 fps in simulation and on `piper_x`, 30 fps on `arx_x5` and `piper`. The official end-effector export `RoboDojo_ee_lerobot_v30_video` is LeRobot v3.0, which the vendored lerobot cannot read, so it is not used.

`expert_data_num` caps the episodes converted per task. `raw_task_dirs` is a comma-separated task list (default: every task). `ckpt_name` is ignored, because all runs of a benchmark share these datasets. Converting the full data takes many hours. `SLA_MEMORY_CONVERT_JOBS` (default 4) sets how many real-robot datasets are converted in parallel.

## Training

Download the SLA base weights, which both configs start from, then train:

```bash
bash download_checkpoints.sh training

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [train.py args...]

# Example: simulation on one node with 8 GPUs
bash train.sh RoboDojo run1 arx_x5 ee 42 0,1,2,3,4,5,6,7
# Example: all three real robots on one node with 8 x 80 GB GPUs
bash train.sh RoboDojo_real run1 all ee 42 0,1,2,3,4,5,6,7 --batch-size 256 --gradient-accumulation-steps 2
```

`bench_name` selects the config: `train_robodojo_sim`, or `train_robodojo_real`. Both start from `checkpoints/SLA_base/params` and run 60k steps with EMA 0.999 and state dropout 0.25. The batch is 256 samples per step for simulation and 512 for real robots.

- **Checkpoints** go to `checkpoints/<train config>/<bench_name>-<ckpt_name>-<env_cfg_type>-ee-<seed>/<step>/`, each step holding `params/`, `assets/<train config>/norm_stats.json` and, for resuming, `train_state/`.
- **Norm stats**: on the first run `train.sh` computes them into `assets/<train config>/norm_stats.json`.
- **Extra arguments** after `gpu_id` go to `openpi/scripts/train.py`, for example `--resume`, `--overwrite` or `--num-train-steps`.


### Multi-node

Run the same `train.sh` command on every node, with the node count, the node's index and node 0's address in front; only `JAX_PROCESS_INDEX` differs between nodes. Compute the norm stats once beforehand, since `train.sh` does not compute them in a multi-node run.

```bash
openpi/.venv/bin/python openpi/scripts/compute_norm_stats.py --config-name train_robodojo_real

# Example: real robots on 4 nodes x 8 GPUs, 512 samples per step. On node 0:
JAX_PROCESS_COUNT=4 JAX_PROCESS_INDEX=0 JAX_COORDINATOR_ADDR=<node 0 IP> JAX_COORDINATOR_PORT=12355 \
  bash train.sh RoboDojo_real run1 all ee 42 0,1,2,3,4,5,6,7
```
Each process logs to `logs/<run>.process<index>.log`.

## Evaluation

Evaluation loads the released real-robot checkpoint `SLA_Memory_robodojo_real_ckpt` from [haofuly/SLA_benchmark_ckpts](https://huggingface.co/haofuly/SLA_benchmark_ckpts), or a checkpoint trained with `train.sh`. For the released one, `download_checkpoints.sh robodojo_real` fetches its weights and norm stats, without the optimizer state, into `checkpoints/SLA_Memory_robodojo_real_ckpt/`; set `train_config_name: train_robodojo_real` and `extra_frames_first: false` in `deploy.yml`:

```bash
bash download_checkpoints.sh robodojo_real

bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: the released checkpoint on piper_x, offline wiring check (no robot)
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo_real make_bread checkpoints/SLA_Memory_robodojo_real_ckpt piper_x ee 0 0 0 uv <eval_env_conda_env>
# Example: the simulation run trained above, on the official layout group 0
bash eval.sh RoboDojo stack_bowls checkpoints/train_robodojo_sim/RoboDojo-run1-arx_x5-ee-42 arx_x5 ee 0 0 0 uv <eval_env_conda_env>
```

`ckpt_name` is the run name used for training, which loads the latest step of `checkpoints/<train_config_name>/<bench_name>-<ckpt_name>-<env_cfg_type>-ee-<seed>/`. For real-robot checkpoints, set `train_config_name: train_robodojo_real` in `deploy.yml` and pass the run directory as `ckpt_name` (e.g. `checkpoints/train_robodojo_real/RoboDojo_real-run1-all-ee-42`), since one checkpoint serves all three robots while `env_cfg_type` names the robot being driven. The released checkpoint was trained with 256 samples per step and with the history frames after the current views, hence `extra_frames_first: false`.

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys beyond the standard set:

| Key | Default | Meaning |
| --- | --- | --- |
| `policy_uv_env_path` | `openpi` | uv project used when `<policy_uv_env>` is `uv`. |
| `train_config_name` | `train_robodojo_sim` | `train_robodojo_sim` or `train_robodojo_real`; must match the checkpoint. |
| `extra_frames_first` | `true` | Whether the history-frame tokens go ahead of the current views. Must match how the checkpoint was trained: `true` for checkpoints trained with this code, `false` for the released `SLA_Memory_robodojo_real_ckpt`. |
| `attention_impl` | `xla` | Attention kernel for inference: `xla` runs on any CUDA GPU; `xla_cudnn` needs a Hopper GPU. Checkpoints work with either. |

Environment variables: `XLA_PYTHON_CLIENT_MEM_FRACTION` (default 0.5 for the policy server, 0.95 for training) and `SLA_MEMORY_CONVERT_JOBS` (see Data Processing).
