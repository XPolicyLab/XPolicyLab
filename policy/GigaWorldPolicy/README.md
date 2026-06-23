# GigaWorldPolicy

This policy follows the XPolicyLab data contract. Raw trajectories are read from XPolicyLab HDF5 episodes and converted into LeRobot v2.1 before training.

## XPolicyLab Contract

- `process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>` prepares `data/<5-tuple>`.
- `train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>` writes checkpoints to `checkpoints/<6-tuple>`.
- `eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>` starts the XPolicyLab model server and env client.
- `model.py` implements `update_obs`, `update_obs_batch`, `get_action`, `get_action_batch`, and `reset` for `XPolicyLab/setup_policy_server.py`.

## Raw Data

Expected XPolicyLab input layout:

```text
data/<dataset_name>/<task_name>/<env_cfg_type>/
├── data/episode_*.hdf5
├── preview_video/
├── scene_layout/
├── seed.txt
└── traj_data/
```

For the current demo:

```text
data/XPolicyLab_demo/stack_bowls/arx_x5/data/episode_*.hdf5
```

`arx_x5` maps to `dual_x5`, so joint state/action is 14-D: left arm 6 + left gripper 1 + right arm 6 + right gripper 1.

## Converted Data

`process_data.sh` converts HDF5 episodes to LeRobot v2.1:

```text
policy/GigaWorldPolicy/data/<dataset>-<ckpt>-<env_cfg>-<num>-<action>/
├── meta/info.json
├── meta/tasks.jsonl
├── meta/episodes.jsonl
├── meta/episodes_stats.jsonl
├── meta/stats.json
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.cam_high/episode_000000.mp4
├── videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
├── videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
├── videos/chunk-000/observation.images.cam_third_view/episode_000000.mp4
└── norm_stats_delta.json
```

Default camera mapping:

- `cam_head` -> `observation.images.cam_high`
- `cam_left_wrist` -> `observation.images.cam_left_wrist`
- `cam_right_wrist` -> `observation.images.cam_right_wrist`
- `cam_third_view` -> `observation.images.cam_third_view`

Images are decoded from XPolicyLab HDF5 bytes or arrays as RGB and stored as RGB videos. Default stored resolution is `640x480`; override with `GIGAWORLD_IMAGE_WIDTH` and `GIGAWORLD_IMAGE_HEIGHT` if needed.

Example:

```bash
GIGAWORLD_PYTHON=/path/to/python   bash process_data.sh XPolicyLab_demo stack_bowls arx_x5 50 joint
```

For multi-task conversion, pass comma-separated task names via `GIGAWORLD_TASK_NAMES`; `ckpt_name` still controls the XPolicyLab 5-tuple output name:

```bash
GIGAWORLD_TASK_NAMES=stack_bowls,another_task   bash process_data.sh XPolicyLab_demo cotrain arx_x5 50 joint
```

If you already have a LeRobot v2.1 dataset, link it instead:

```bash
GIGAWORLD_SOURCE_DATA_DIR=/path/to/lerobot   bash process_data.sh XPolicyLab_demo stack_bowls arx_x5 50 joint
```

Optional helpers:

```bash
GIGAWORLD_COMPUTE_NORM=1 bash process_data.sh ...   # default enabled
GIGAWORLD_GENERATE_T5=1 bash process_data.sh ...    # optional, GPU-heavy
```

## Training

Default LeRobot data path is `${XPOLICYLAB_LEROBOT_DATA_ROOT:-${LEROBOT_DATA_ROOT:-<XPolicyLab>/data}}/<repo_id>`.
For `arx_x5`, the default repo id is `XPolicyLab_sim_arx-x5_v30`. Set `GIGAWORLD_DATA_DIR` to override the complete data path, or set `LEROBOT_DATASET_REPO_ID` to override only the repo id.

```bash
GIGAWORLD_PYTHON=/path/to/python   bash train.sh XPolicyLab_demo stack_bowls arx_x5 50 joint 0 0,1,2,3
```

Training seed is propagated as `XPolicyLab_seed + 1` for giga-train (`seed > 0`), and `PYTHONHASHSEED` uses the same resolved value.

Default training config is `configs.xpolicylab_gigaworld.config` with 14-D state/action and LeRobot v2.1 input. The converted dataset stores all four demo cameras; the default model config uses the first three GWP views unless you override `view_keys` in the config.

Useful overrides:

- `GIGAWORLD_DATA_DIR`: explicit converted LeRobot data path.
- `GIGAWORLD_NORM_PATH`: normalization stats JSON.
- `GIGAWORLD_PRETRAINED_PATH`: Wan2.2 Diffusers model path.
- `GIGAWORLD_MODEL_ACTION_DIM`, `GIGAWORLD_MODEL_STATE_DIM`, `GIGAWORLD_NUM_FRAMES`, `GIGAWORLD_ACTION_CHUNK`.
- `GIGAWORLD_DRY_RUN=1`: write the effective config without launching training.

## Evaluation

Environment setup is documented in [INSTALLATION.md](INSTALLATION.md). Run `bash install.sh` before first use.

`deploy.yml` defaults to the XPolicyLab demo dimensions: `model_action_dim=14`, `model_state_dim=14`, three XPolicyLab camera views, and RGB input observations.

For real checkpoint inference, set one of:

```yaml
checkpoint_path: /path/to/checkpoint-5000/model_ema.pt
# or use the standard XPolicyLab 6-tuple through eval.sh
checkpoint_num: checkpoint-5000
```

For server/client interface smoke tests, set `load_model: false` in `deploy.yml`; the wrapper will return zero actions with the configured action chunk.

Single-machine evaluation (`eval.sh` allocates a port, starts the policy server, waits until it is ready, and then starts the env client):

```bash
bash eval.sh XPolicyLab debug_task <ckpt_name> arx_x5 1 joint 0 0 0 gigaworld-policy gigaworld-policy
```

Split-machine evaluation (GPU host runs policy server, simulator host runs env client):

```bash
# GPU host
FREE_PORT=$(bash ../../utils/get_free_port.sh)
bash setup_eval_policy_server.sh XPolicyLab debug_task <ckpt_name> arx_x5 1 joint 0 0 gigaworld-policy "${FREE_PORT}" 0.0.0.0

# Simulator host
bash setup_eval_env_client.sh XPolicyLab debug_task <ckpt_name> arx_x5 joint 0 0 gigaworld-policy \
  "ckpt_name=<ckpt_name>,action_type=joint" <port> <policy_server_ip>
```

Set `POLICY_SERVER_HOST` / `POLICY_SERVER_IP` (or GigaWorldPolicy-specific equivalents) to control server binding and client connection when using `eval.sh`.
