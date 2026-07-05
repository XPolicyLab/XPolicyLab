![](./assets/logo.png)

<h1 align="center">XPolicyLab: A Unified Standard for Robot Policy Development and Evaluation</h1>

XPolicyLab is a unified platform for robot policy development, training, inference, evaluation, and reinforcement learning. It defines a common interface for data processing, model training, policy serving, and environment-side evaluation, with the goal of improving code readability, reproducibility, and ecosystem compatibility across robot learning projects.

The project integrates a broad set of frontier policies and is maintained with the global developer community. Contributions of new policies, benchmarks, and infrastructure components are welcome. XPolicyLab is currently designed to work closely with RoboDojo Benchmark and RoboTwin Benchmark.

arXiv (coming soon) | User Group (coming soon) | Tutorial (RoboDojo)

Co-Project Leads: Tianxing Chen, Tian Nian, Zijian Cai

# 🚀 1. Getting Started

## 1.1 Supported Scope

| Area | Supported models, projects, and benchmarks |
|---|---|
| WAM | FastWAM, Motus, GigaWorldPolicy, DreamZero, LingBot-VA, Wall-WM |
| VLA | GalaxeaVLA, GR00T_N17, H_RDT, InternVLA_A1, LDA-1B, , LingBot-VLA, MolmoACT2, OpenVLA_OFT, Pi0, Pi05, Pi0-Fast, RDT-1B, SmolVLA, Spirit_v15, TinyVLA, X_VLA, starVLA, A1, Abot_M0, Being_H05, GO1, Xiaomi_Robotics_0, Dexbotic_DM0, Mem_0, RISE, UniDex, Wall-OSS |
| Imitation learning | ACT, DP |
| Infrastructure | RLinf, starVLA |
| In progress | ... |

## 1.2 Environment and Data

Create a workspace, clone XPolicyLab, and download the demonstration data and environment configuration:

```bash
mkdir demo_env
cd demo_env

# Clone this repository.
git clone git@github.com:Luminis-Platform/XPolicyLab.git
cd XPolicyLab

# Download the complete demo dataset in the standard format.
# This also downloads env_cfg. Use it when training with XPolicyLab only;
# do not download it inside an existing simulator workspace.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface demo # huggingface

# Optional: download RoboDojo data in the standard HDF5 format.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface hdf5 # huggingface
bash scripts/RoboDojo/download_robodojo_data.sh modelscope hdf5  # modelscope

# Optional: download RoboDojo data in LeRobot v3.0 format.
# qpos denotes joint positions; ee data requires reconversion.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v3.0 # huggingface
bash scripts/RoboDojo/download_robodojo_data.sh modelscope lerobot_v3.0  # modelscope

# Optional: download RoboDojo data in LeRobot v2.1 format.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v2.1 # huggingface
bash scripts/RoboDojo/download_robodojo_data.sh modelscope lerobot_v2.1  # modelscope
```

XPolicyLab is usually mounted inside a larger experiment or simulation workspace. A typical layout is:

```text
demo_env/
├── data
│   ├── demo/         # scripts/download_demo_data.sh
│   ├── RoboDojo/     # scripts/RoboDojo/download_robodojo_data.sh modelscope hdf5
│   └── {bench_name}/
│       └── {task_name}/
│           └── {env_cfg}/
│               ├── data
│               ├── preview_video
│               ├── scene_layout
│               ├── seed.txt
│               └── traj_data
├── env_cfg           # scripts/download_demo_data.sh
└── XPolicyLab
```

# 📐 2. Standard Formats

## 2.1 Observation Format

<details>
<summary>Observation Data Format v1.0</summary>

XPolicyLab represents all pose values as `[x, y, z, qw, qx, qy, qz]`. The same convention is used for end-effector poses, TCP poses, mobile base poses, and trajectory datasets.

```text
Observation Data Format v1.0
├── data_format_version                        (Field)    string
├── instruction / instructions                 (Field)    string or list of strings
├── env_idx                                    (Field)    int
├── additional_info/                           (Group)
│   └── frequency                              (Field)    int
├── vision/                                    (Group)
│   ├── cam_head/                              (Group)
│   │   ├── color                              (Field)    (H, W, 3), RGB
│   │   ├── depth                              (Field)    (H, W) or (H, W, 1)
│   │   ├── approximate_depth                  (Field)    optional
│   │   ├── intrinsic_matrix                   (Field)    (3, 3)
│   │   ├── extrinsics_matrix                  (Field)    (4, 4)
│   │   └── shape                              (Field)    (2,) or (3,)
│   ├── cam_left_wrist/                        (Group, optional)
│   ├── cam_right_wrist/                       (Group, optional)
│   ├── cam_wrist/                             (Group, optional for single-arm robots)
│   └── cam_third_view/                        (Group, optional)
└── state/                                     (Group)
    ├── left_arm_joint_state                   (Field)    (DOF,), optional
    ├── left_ee_joint_state                    (Field)    (EEF_DOF,), optional
    ├── left_ee_pose                           (Field)    (7,), optional
    ├── left_tcp_pose                          (Field)    (7,), optional
    ├── left_delta_ee_pose                     (Field)    (7,), optional
    ├── right_arm_joint_state                  (Field)    (DOF,), optional
    ├── right_ee_joint_state                   (Field)    (EEF_DOF,), optional
    ├── right_ee_pose                          (Field)    (7,), optional
    ├── right_tcp_pose                         (Field)    (7,), optional
    ├── right_delta_ee_pose                    (Field)    (7,), optional
    ├── arm_joint_state                        (Field)    (DOF,), optional for single-arm robots
    ├── ee_joint_state                         (Field)    (EEF_DOF,), optional for single-arm robots
    ├── ee_pose                                (Field)    (7,), optional for single-arm robots
    ├── tcp_pose                               (Field)    (7,), optional for single-arm robots
    ├── delta_ee_pose                          (Field)    (7,), optional for single-arm robots
    └── mobile/                                (Group, optional)
        ├── base_pose                          (Field)    (7,)
        └── base_twist                         (Field)    (6,)
```

</details>

## 2.2 Trajectory Data Format

<details>
<summary>Trajectory Data Format v1.0</summary>

XPolicyLab represents all pose values as `[x, y, z, qw, qx, qy, qz]`. The same convention is used for end-effector poses, TCP poses, mobile base poses, and trajectory datasets.

```text
Trajectory Data Format v1.0
├── data_format_version                        (Dataset)  string, e.g. "v1.0"
├── instructions                               (Dataset)  JSON-serialized string list
├── subtasks                                   (Dataset)  JSON-serialized stage annotations
├── additional_info/                           (Group)
│   └── frequency                              (Dataset)  int, control / recording frequency in Hz
├── vision/                                    (Group)
│   ├── cam_head/                              (Group)
│   │   ├── colors                             (Dataset)  (T, H, W, 3) uint8 RGB image byte stream
│   │   ├── depths                             (Dataset)  (T, H, W) or (T, H, W, 1)
│   │   ├── approximate_depths                 (Dataset)  optional
│   │   ├── intrinsic_matrix                   (Dataset)  (3, 3) or (T, 3, 3)
│   │   ├── extrinsics_matrix                  (Dataset)  (4, 4) or (T, 4, 4)
│   │   └── shape                              (Dataset)  (2,) [H, W] or (3,) [H, W, C]
│   ├── cam_left_wrist/                        (Group, optional for dual-arm robots)
│   ├── cam_right_wrist/                       (Group, optional for dual-arm robots)
│   ├── cam_wrist/                             (Group, optional for single-arm robots)
│   └── cam_third_view/                        (Group, optional)
└── state/                                     (Group)
    ├── left_arm_joint_states                  (Dataset)  (T, DOF_L), optional
    ├── left_ee_joint_states                   (Dataset)  (T, EEF_DOF_L), optional
    ├── left_ee_poses                          (Dataset)  (T, 7), optional
    ├── left_tcp_poses                         (Dataset)  (T, 7), optional
    ├── left_delta_ee_poses                    (Dataset)  (T, 7), optional
    ├── right_arm_joint_states                 (Dataset)  (T, DOF_R), optional
    ├── right_ee_joint_states                  (Dataset)  (T, EEF_DOF_R), optional
    ├── right_ee_poses                         (Dataset)  (T, 7), optional
    ├── right_tcp_poses                        (Dataset)  (T, 7), optional
    ├── right_delta_ee_poses                   (Dataset)  (T, 7), optional
    ├── arm_joint_states                       (Dataset)  (T, DOF), optional for single-arm robots
    ├── ee_joint_states                        (Dataset)  (T, EEF_DOF), optional for single-arm robots
    ├── ee_poses                               (Dataset)  (T, 7), optional for single-arm robots
    ├── tcp_poses                              (Dataset)  (T, 7), optional for single-arm robots
    ├── delta_ee_poses                         (Dataset)  (T, 7), optional for single-arm robots
    └── mobile/                                (Group, optional)
        ├── base_poses                         (Dataset)  (T, 7)
        └── base_twists                        (Dataset)  (T, 6), [vx, vy, vz, wx, wy, wz]
```

</details>

# 🧩 3. Policy Structure

XPolicyLab uses a standard policy package layout so that different policies can share the same data, training, serving, and evaluation interfaces. The teaching example in `policy/demo_policy` contains the complete template:

| File | Purpose |
|---|---|
| `deploy.py` | Implements the deployment loop, including serial and batched interaction modes. |
| `deploy.yml` | Defines deployment parameters passed into `model.Model`, including default model configuration and checkpoint loading options. |
| `eval.sh` | Local one-command evaluation entry point. It allocates an available `policy_server_port`, starts both `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` on the same machine, and cleans up the policy server when evaluation exits. |
| `setup_eval_policy_server.sh` | Policy-side startup script. It runs the model server inside the policy environment and binds `policy_server_host:policy_server_port`; in remote deployment, this script is launched on the model/GPU machine, while in local evaluation it could be launched by `eval.sh`. |
| `setup_eval_env_client.sh` | Environment-side startup script. It runs the evaluation client inside the environment conda env and dispatches to debug or simulation runners according to `EVAL_ENV_TYPE` (default: `sim`); in remote deployment, this script is launched on the environment/simulator machine, while in local evaluation it could be launched by `eval.sh`. |
| `install.sh` | Installs the policy environment and editable XPolicyLab package. |
| `model.py` | Defines model loading, observation updates, action generation, and reset logic. |
| `process_data.sh` | Converts raw datasets into the policy-specific training format. |
| `train.sh` | Launches policy training. |

`policy/DP` provides a more complete and easy-to-follow reference implementation.

# 🛠️ 4. Integrating a Custom Policy

The recommended integration principle is script-level reproducibility: avoid hard-coded paths, expose key training and evaluation parameters, support multiple robot morphologies when possible, and keep policy-specific code minimal and documented.

You can first run the demo policy to validate the end-to-end interaction flow. Ensure that you either have a simulation environment available or have downloaded the demo data.

```bash
conda create -n demo python=3.10
conda activate demo
cd policy/demo_policy
bash install.sh
bash eval.sh RoboDojo stack_bowls demo_ckpt arx_x5 joint 0 0 0 demo demo
```

## 4.1 Create a Policy Template

From the XPolicyLab root directory:

```bash
bash scripts/create_policy.sh ${policy_name}
```

The script creates `policy/${policy_name}` with the standard template files and inline parameter comments. External source code can be placed in a dedicated subdirectory, such as `policy/starVLA/source_starvla`.

If the external source is cloned from another repository, remove its `.git` directory to avoid registering it as a Git submodule. We recommend committing the imported source snapshot before making adaptation changes, which makes subsequent diffs easier to inspect.

## 4.2 Standard Parameters and Naming

Entry scripts use fixed positional arity: **process_data** accepts 4 required arguments plus 1 optional (`expert_data_num`), **train** accepts 6, and **eval** accepts 10. Only `process_data.sh` may take `expert_data_num` as an optional trailing argument; train and eval locate artifacts by `ckpt_name` and the 4-tuple naming convention below.

These parameters should appear consistently in data processing, training, and evaluation scripts when applicable:

| Parameter | Requirement | Training usage | Evaluation usage |
|---|---:|---|---|
| `bench_name` | Required | Identifies the source dataset under `data/`, e.g. RoboDojo, RoboTwin, or RoboDojo with depth. | Identifies the dataset family used by the evaluation task and checkpoint naming. |
| `ckpt_name` | Required | Names the experiment and processed artifacts. It may equal `task_name` for single-task training. | Locates the checkpoint directory, often together with dataset, robot configuration, action type, and seed. |
| `task_name` | Required for evaluation | Optional for training. Multi-task or co-training pipelines can decide internally which raw tasks to consume. | Specifies the task executed by the environment client. |
| `env_cfg_type` | Required | Selects robot and environment configuration, including robot morphology. Demo and RoboDojo data use `arx_x5` by default. | Selects the environment configuration for rollout. |
| `action_type` | Required | Specifies the policy action representation, such as `joint` or `ee`. | Must match the deployment control mode. |
| `seed` | Required | Controls training randomness and enables multi-seed reporting. | Controls evaluation randomness. |
| `expert_data_num` | Optional | Optional trailing argument of `process_data.sh` only; caps episodes per task and defaults to all available episodes. To ablate data scale, use a distinct `ckpt_name` (e.g. `myrun_50ep`). | Not used. Evaluation scripts locate checkpoints by `ckpt_name`; pass the full run directory name as `ckpt_name`. |

Recommended artifact names:

| Artifact | Naming convention | Default location |
|---|---|---|
| Processed dataset | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` | `policy/<policy_name>/data/` or `policy/<policy_name>/processed_data/` |
| Training checkpoint | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>` | `policy/<policy_name>/checkpoints/` |
| Raw RoboDojo data | `data/${bench_name}/${task_name}/${env_cfg}` | Workspace-level `data/` directory |

This convention keeps datasets and checkpoints identifiable from their core experimental parameters, while still allowing `process_data.sh` to read one or more raw `task_name` directories.

Standard script contracts (positional args):

```bash
# Data processing (expert_data_num optional — omit to use all episodes)
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Training
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Evaluation (ckpt_name = full checkpoint directory name under checkpoints/)
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>
```

End-to-end example:

```bash
bash process_data.sh RoboDojo cotrain arx_x5 joint
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <env> RoboDojo
```

## 4.3 Implement `install.sh`

Each policy should install its original dependencies and install XPolicyLab in editable mode:

```bash
# Install policy-specific dependencies.
# ...

# Install XPolicyLab from the repository root.
cd ../../
pip install -e .
```

Editable installation is recommended because policy code often imports utilities from `XPolicyLab.utils`; installing from the root keeps these imports synchronized with the local repository during development.

## 4.4 Implement `process_data.sh`

`process_data.sh` converts official data under the workspace-level `data/` directory into the model-specific training format. RoboDojo LeRobot v2.1 and v3.0 datasets are also available for policies that can consume them directly. For an example, see `policy/DP/process_data.sh` and the corresponding Python processing code.

The following utilities are commonly used:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

| Utility | Description |
|---|---|
| `load_hdf5` | Loads an HDF5 trajectory file from an input path. |
| `decode_image_bit` | Decodes RGB image byte streams into NumPy arrays. It supports both single-frame byte streams and full trajectory byte streams. |
| `get_robot_action_dim_info` | Takes `env_cfg_type` as a string and returns `arm_dim` and `ee_dim` lists for robot-specific action dimensions. |

Image data is stored in RGB channel order and may be represented as multi-frame byte streams. Use `decode_image_bit` to recover correct image arrays. Avoid hard-coding action dimensions when the model architecture permits configurable dimensions.

<details>
<summary>Robot dimension example</summary>

`env_cfg/robot/_robot_info.json` stores morphology-specific dimensions indexed by `env_cfg_type`. A list of length 1 denotes a single-arm robot, while a list of length 2 denotes a dual-arm robot.

```json
{
    "x5": {
        "arm_dim": [6],
        "ee_dim": [1]
    },
    "arx_x5": {
        "arm_dim": [6, 6],
        "ee_dim": [1, 1]
    },
    "g1_inspire": {
        "arm_dim": [7, 7],
        "ee_dim": [12, 12]
    }
}
```

</details>

## 4.5 Implement `train.sh`

`train.sh` should expose the key experiment parameters listed above and pass `seed` through to the underlying training code. Some external policy repositories hard-code the seed; adapt them so that different seeds produce independent training runs and can be averaged in later evaluation.

## 4.6 Implement Evaluation and Deployment

Evaluation requires two components:

| Component | Responsibility |
|---|---|
| `model.py` | Loads the policy, maintains observation state, returns actions, and resets model-side state. |
| `deploy.py` | Runs the environment interaction loop and calls the policy server. |

XPolicyLab provides an offline debug environment through `debug_env_client.py`. It generates correctly shaped observations and validates returned actions, enabling rapid checks of parameter routing, model input/output dimensions, and server-client communication. Set `EVAL_ENV_TYPE=debug` for offline debugging; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation without editing `eval.sh`.

### 4.6.1 `model.Model` Interface

`model.py` should define a `Model` class that inherits from `ModelTemplate` and implements:

| Method | Contract |
|---|---|
| `__init__(model_cfg)` | Receives model configuration from `deploy.yml` and runtime overrides from `setup_eval_policy_server.sh`. |
| `update_obs(obs)` | Updates the model with a single environment observation. See [Observation Format](#21-observation-format). |
| `update_obs_batch(obs_list)` | Updates the model with batched observations. Each observation dictionary includes an `env_idx`. |
| `get_action()` | Returns one action dictionary. Action keys define the control mode, e.g. `left_arm_joint` for joint control or `left_ee_pose` for end-effector pose control. |
| `get_action_batch()` | Returns a list of action dictionaries for batched evaluation. |
| `reset()` | Resets model-side state. |

The returned action representation should match `action_type` and the observation state family. For example, a policy should not mix `left_arm_joint_state`-style joint control with `left_ee_pose`-style pose control in the same output. If implementing batched inference is difficult for a policy, `update_obs_batch` and `get_action_batch` can be implemented as simple loops over the single-environment methods. See `policy/DP/model.py` for a concrete implementation.

# 🔌 5. Deployment Workflow

During evaluation, the environment process and policy process communicate through a policy server. This design supports remote deployment and isolates simulator dependencies from policy dependencies.

| File | Role |
|---|---|
| `deploy.yml` | Defines model deployment parameters. Low-frequency options can be set as constants or `null` and overridden later with `--overrides`. `eval_batch` selects serial or batched inference. |
| `eval.sh` | Local orchestration script for same-machine evaluation. It selects an available `policy_server_port`, launches the policy server first, then launches the environment client, and finally terminates the server when evaluation exits. Use this script when the policy and environment can run on the same machine. |
| `setup_eval_policy_server.sh` | Policy-side startup script. It enters `policy_conda_env` or uses `policy_uv_env_path`, starts `setup_policy_server.py`, loads `model.Model`, and binds `policy_server_host:policy_server_port` so that environment clients can request actions. In remote deployment, run this script on the model/GPU machine. |
| `setup_eval_env_client.sh` | Environment-side startup script. It enters `eval_env_conda_env`, connects to the policy server through `policy_server_host:policy_server_port`, and calls `XPolicyLab/utils/setup_env_client.sh`, which dispatches to `run_debug_env_client.sh` or `run_sim_env_client.sh` according to `EVAL_ENV_TYPE` (default: `sim`). In remote deployment, run this script on the environment or simulator machine. |

Use `policy_gpu_id` for the model process and `env_gpu_id` for the simulation process. The conda-based policy environment can be modeled after `policy/DP`; the uv-based path can be modeled after `policy/PI_05`.

For remote deployment, run `setup_eval_policy_server.sh` on the GPU policy machine, then run `setup_eval_env_client.sh ... <policy_server_port> <policy_server_ip>` on the simulation machine. Both sides only need to agree on `policy_server_ip:policy_server_port`.

## 5.1 `deploy.py` Notes

Read `policy/demo_policy/deploy.py` for the reference control flow. The most important calls are:

| API | Purpose |
|---|---|
| `TASK_ENV.is_episode_end()` | Checks whether all evaluation episodes have finished. |
| `model_client.call(func_name, obs)` | Serializes the target `Model` method name and observation payload, sends them to the policy server, and returns the model-side result. |

After offline debugging, unset `EVAL_ENV_TYPE` or set `EVAL_ENV_TYPE=sim` for simulation. The environment client dispatches to the corresponding runner without changes to `eval.sh`, `setup_eval_policy_server.sh`, or `setup_eval_env_client.sh`.

| `EVAL_ENV_TYPE` | Mode |
|---|---|
| unset or `sim` | RoboDojo simulation (`run_sim_env_client.sh`) |
| `debug` | Offline shape/IO validation (`run_debug_env_client.sh`) |
| `real` | Not available in open-source release |

The legacy `deploy.yml` key `eval_env` is deprecated and ignored; `setup_env_client.sh` prints a warning when it is still present. `EVAL_ENV_TYPE` also selects the policy-server wire protocol when `deploy.yml` does not set `protocol` explicitly: `sim` (or unset) defaults to `legacy_tcp` to match the simulation eval client (`ModelClient`), while `debug` / `real` default to `robodojo_ws`. Set `protocol` in `deploy.yml` (or pass `--protocol` to `setup_policy_server.py`) to pin it. When the env client daemon runs sim trials (`--eval-env-type sim`), it shells out to `scripts/eval_policy.sh`; set `XPOLICYLAB_SIM_CONDA_ENV=<simulator-conda-env>` so the subprocess activates the simulator environment (`XPOLICYLAB_SIM_ROOT`, `XPOLICYLAB_SIM_DEVICE_ID`, and `XPOLICYLAB_SIM_SEED` override the benchmark root / GPU / seed), and note a trial only completes when `_result.json` reports `eval_time >= 1`.

## 5.2 Platform evaluation (env client daemon)

When RoboDojo drives trials from the control plane (x-policy-web), the environment machine should run the env client as a long-lived HTTP daemon instead of a one-shot `debug_env_client.py` process.

| Setting | Purpose |
|---|---|
| `deploy.yml` → `env_client_mode: daemon` | `setup_eval_env_client.sh` launches `python -m eval_station.servers.env_client_server` and listens on `0.0.0.0:19200`. |
| `deploy.yml` → `env_client_mode: run-once` (default) | Local `eval.sh` keeps the existing one-shot debug client path. |

Typical platform startup on the environment / robot station:

```bash
bash policy/<POLICY>/setup_eval_env_client.sh \
  <dataset> <task> <ckpt> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

With `env_client_mode: daemon` in `deploy.yml`, the process stays up and exposes:

| Endpoint | Role |
|---|---|
| `GET /v1/health` | Daemon liveness and baseline metadata (`policy_name`, `eval_env_type`). |
| `POST /sessions/{evaluation_id}/dispatch` | Cache platform dispatch payload for the session. |
| `POST /sessions/{evaluation_id}/trials/{trial_index}/start` | Run the selected trial (blocking until completion or stop). |
| `POST /sessions/{evaluation_id}/trials/{trial_index}/stop` | Request stop at the next `is_episode_end` check (typically one action step later). |

The control plane calls these endpoints on the eval station; finish webhooks and artifact upload remain on the RoboDojo publish path.

### 5.2.1 Real-robot (`EVAL_ENV_TYPE=real`)

Real-robot evaluation is not shipped in the open-source release. `EVAL_ENV_TYPE=real` prints a warning and still routes to the real env client, which requires daemon mode (`env_client_mode: daemon` in `deploy.yml`) plus `action_type` and `base_cfg` in `additional_info` (or `ROBODOJO_BASE_CFG`); without the internal `task_env.real_env_client` runtime the daemon fails at startup.

# 📚 6. Citation

```bibtex
Coming Soon
```

# 📬 7. Contact

Tianxing Chen: [chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)