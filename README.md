<div align="center">

<img src="assets/logo.png" alt="XPolicyLab" width="480"/>

<p><strong>XPolicyLab: A unified standard and infrastructure for robot policy development and deployment.</strong></p>

</div>

XPolicyLab provides shared standards and infrastructure for developing, serving, training, evaluating, and deploying robot policies. It keeps policy code, dependencies, checkpoints, and training recipes under `policy/<POLICY>/`, while exposing a common adapter contract for model serving and environment-side evaluation.

Use this README for repository-level concepts and integration steps. For model-specific setup, checkpoints, and training details, start from the corresponding policy README.

## 🚀 What XPolicyLab Enables

- **Environment isolation**: run the policy model in its own conda/uv environment while the simulator, benchmark, or robot client runs separately.
- **Remote deployment**: connect the policy server and environment client through websocket, either on one machine or across machines.
- **A common adapter contract**: use the same high-level lifecycle for installation, data conversion, training, serving, and evaluation.
- **A large policy zoo**: reuse adapters for VLA/WAM policies, imitation-learning baselines, and reference templates.
- **Benchmark and infra integration**: mount XPolicyLab into benchmark or simulator workspaces without coupling policy code to one environment.

## 🌐 Supported Benchmarks And Infrastructure

**Benchmarks**

- **RoboDojo**: supported for RoboDojo simulator-backed evaluation and RoboDojo-format data exports.
- **RoboTwin**: supported as a benchmark and data source through policy-specific adapters and conversion scripts.

**Infrastructure**

- **RLinf**: supported infrastructure target for policy development and deployment workflows.
- **StarVLA**: supported infrastructure and policy stack; see [policy/starVLA](policy/starVLA/README.md).

## 🧭 Integrated Policies

Top-level adapters live in `policy/`. Treat each policy README as the source of truth for that model's paper/repo link, environment, data format, training entrypoint, and checkpoint layout.

<details>
<summary>Policy catalog</summary>

**Foundation / VLA / WAM policies**

- [A1](policy/A1/README.md), [AHA_WAM](policy/AHA_WAM/README.md), [Abot_M0](policy/Abot_M0/README.md), [Being_H05](policy/Being_H05/README.md), [Dexbotic_DM0](policy/Dexbotic_DM0/README.md), [Dexora_1B](policy/Dexora_1B/README.md)
- [DreamZero](policy/DreamZero/README.md), [EventVLA](policy/EventVLA/README.md), [FastWAM](policy/FastWAM/README.md), [GO1](policy/GO1/README.md), [GR00T_N17](policy/GR00T_N17/README.md), [GalaxeaVLA](policy/GalaxeaVLA/README.md)
- [GigaWorldPolicy](policy/GigaWorldPolicy/README.md), [H_RDT](policy/H_RDT/README.md), [Hy_Embodied_05_VLA](policy/Hy_Embodied_05_VLA/README.md), [InternVLA_A1](policy/InternVLA_A1/README.md), [LDA_1B](policy/LDA_1B/README.md)
- [LingBot_VA](policy/LingBot_VA/README.md), [LingBot_VLA](policy/LingBot_VLA/README.md), [Mem_0](policy/Mem_0/README.md), [MolmoACT2](policy/MolmoACT2/README.md), [Motus](policy/Motus/README.md)
- [OpenVLA_OFT](policy/OpenVLA_OFT/README.md), [Pi_0](policy/Pi_0/README.md), [Pi_05](policy/Pi_05/README.md), [Pi_0_Fast](policy/Pi_0_Fast/README.md), [RDT_1B](policy/RDT_1B/README.md), [RISE](policy/RISE/README.md)
- [SmolVLA](policy/SmolVLA/README.md), [Spatial_Forcing](policy/Spatial_Forcing/README.md), [Spirit_v15](policy/Spirit_v15/README.md), [TinyVLA](policy/TinyVLA/README.md), [X_VLA](policy/X_VLA/README.md), [X_WAM](policy/X_WAM/README.md), [Xiaomi_Robotics_0](policy/Xiaomi_Robotics_0/README.md), [starVLA](policy/starVLA/README.md)

**Baselines and examples**

- [ACT](policy/ACT/README.md), [DP](policy/DP/README.md), [demo_policy](policy/demo_policy/README.md)

</details>

## 🧩 Framework Overview

XPolicyLab separates model-side dependencies from environment-side dependencies.

```text
Policy environment                         Evaluation / benchmark environment
------------------                         ----------------------------------
policy/<POLICY>/model.py     <---ws--->    env client / simulator / robot
policy server                              environment client
deploy.yml runtime config                  benchmark task and observation API
```

A typical adapter contains:

```text
policy/<POLICY>/
├── README.md                    # policy-specific guide
├── INSTALLATION.md              # optional detailed setup notes
├── install.sh                   # environment setup
├── process_data.sh              # optional data conversion
├── train.sh                     # optional training
├── eval.sh                      # same-machine evaluation
├── setup_eval_policy_server.sh  # policy-side server
├── setup_eval_env_client.sh     # environment-side client
├── deploy.yml                   # runtime config
├── deploy.py                    # evaluation loop
└── model.py                     # model adapter
```

`model.py` implements the model-facing API. `deploy.py` bridges environment observations to model-server calls. Use [policy/demo_policy](policy/demo_policy/README.md) as the minimal adapter reference.

`model.py` should define a `Model` class with this shape:

| Method | Contract |
| --- | --- |
| `__init__(model_cfg)` | Load model config, checkpoints, processors, and runtime overrides from `deploy.yml`. |
| `update_obs(obs)` | Update model state from one observation dictionary. |
| `update_obs_batch(obs_list)` | Update model state from a list of observation dictionaries. |
| `get_action()` | Return one action chunk as a list of action dictionaries. |
| `get_action_batch(env_idx_list=None)` | Return batched action chunks aligned with active environment indices. |
| `reset()` | Clear model-side state between evaluation episodes. |

The default policy-server protocol is websocket (`protocol: ws` in `deploy.yml`). Keep `legacy_tcp` only for old adapters that have not migrated yet.

## 🛠️ Model Integration Guide

The fastest way to add a model is to copy the reference adapter, keep the XPolicyLab boundary small, and debug the adapter before touching a real simulator.

1. **Learn the reference adapter**: read [policy/demo_policy](policy/demo_policy/README.md), especially `model.py`, `deploy.py`, `deploy.yml`, `eval.sh`, `setup_eval_policy_server.sh`, and `setup_eval_env_client.sh`.
2. **Understand the arguments**: keep `bench_name`, `task_name`, `ckpt_name`, `env_cfg_type`, `action_type`, and `seed` consistent across data, training, and eval.
3. **Create a skeleton**: run `bash scripts/create_policy.sh <POLICY_NAME>` and immediately fill in `policy/<POLICY_NAME>/README.md`.
4. **Implement `model.py` first**: load model resources in `__init__`, store observations in `update_obs`, translate observations to model-native inputs, return XPolicyLab action dictionaries from `get_action`, and reset state in `reset`.
5. **Keep deployment simple**: put runtime defaults in `deploy.yml`; keep `deploy.py` aligned with `demo_policy/deploy.py` unless the environment loop truly differs.
6. **Debug without a simulator**: run `EVAL_ENV_TYPE=debug` to check imports, server startup, observation serialization, action keys, action dimensions, and batch logic.
7. **Move to simulator or remote deployment**: after debug mode passes, use `EVAL_ENV_TYPE=sim` or split policy server and environment client across machines.

<details>
<summary>Agent Skill checklist for model integration</summary>

When using a coding agent, give it this checklist:

```text
Integrate <POLICY_NAME> into XPolicyLab.

Use policy/demo_policy as the reference.
1. Inspect the upstream model's inference API and dependencies.
2. Create or update policy/<POLICY_NAME>/README.md with install, checkpoint, train, and eval commands.
3. Implement install.sh and, if needed, process_data.sh and train.sh.
4. Implement model.py with Model.__init__, update_obs, get_action, reset, and batch methods.
5. Keep deploy.py aligned with policy/demo_policy/deploy.py.
6. Put runtime defaults in deploy.yml and use protocol: ws.
7. Run EVAL_ENV_TYPE=debug eval.sh and fix shape/action-key/server errors.
8. Summarize supported action_type, env_cfg_type, checkpoint layout, and remaining limitations.
```

A minimal Cursor Agent Skill can look like this:

```markdown
---
name: xpolicylab-model-integration
description: Guides agents through integrating a new robot policy into XPolicyLab. Use when adding or updating policy/<POLICY>/ adapters, model.py, deploy.py, deploy.yml, install scripts, training scripts, or debug-mode evaluation.
---

# XPolicyLab Model Integration

Follow the XPolicyLab README and policy/demo_policy reference adapter.

1. Read policy/demo_policy/model.py, deploy.py, deploy.yml, eval.sh, and README.md.
2. Inspect the target model's inference API, dependencies, checkpoint layout, and expected observations/actions.
3. Create or update policy/<POLICY>/ with scripts/create_policy.sh if needed.
4. Implement model.py first; keep upstream model code unchanged when possible.
5. Keep deploy.py aligned with demo_policy unless the environment loop truly differs.
6. Put runtime defaults in deploy.yml and prefer protocol: ws.
7. Run EVAL_ENV_TYPE=debug before simulator-backed evaluation.
8. Document exact install, train, eval, action_type, env_cfg_type, and checkpoint assumptions in policy/<POLICY>/README.md.
```

</details>

## ⚡ Quick Start

Clone XPolicyLab as a normal Python project when you are developing adapters, running offline checks, training from prepared data, or using your own environment client:

```bash
mkdir demo_env
cd demo_env
git clone git@github.com:Luminis-Platform/XPolicyLab.git
cd XPolicyLab
pip install -e .
```

You do not need a simulator installation to start model-side development. Download a small HuggingFace demo bundle and keep the data next to `XPolicyLab/`:

```bash
# From demo_env/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh huggingface demo
```

This creates:

```text
demo_env/
├── data/        # demo data, including a small 10-episode HuggingFace bundle
└── XPolicyLab/
```

You can also pull HDF5 or LeRobot exports for RoboDojo and other benchmark-backed experiments:

```bash
bash scripts/RoboDojo/download_robodojo_data.sh huggingface hdf5
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v3.0
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v2.1
```

With this setup, you can test data conversion, model loading, training scripts, and debug-mode evaluation before connecting to a simulator-backed benchmark.

```bash
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

For RoboDojo simulation, mount `XPolicyLab/` beside the simulator-side `env_cfg/`, `scripts/`, `src/eval_client/`, and `task/` directories.

## 🔄 Common Workflow

Most adapters expose the same top-level shape. Some policies add extra arguments, consume upstream-native datasets, or skip training support. Follow the policy README when it differs from this template.

```bash
cd policy/<POLICY>

# Install the policy runtime.
bash install.sh

# Optional: convert or prepare policy-specific data.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

# Optional: train.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Evaluate on one machine.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Common argument meanings:

| Argument | Meaning |
| --- | --- |
| `bench_name` | Dataset or benchmark family, for example `RoboDojo`, `RoboTwin`, or a custom suite. Use `RoboDojo` for RoboDojo data/eval. |
| `task_name` | Task or environment name used by the environment client. |
| `ckpt_name` | Checkpoint/run identifier, full run directory, or policy-specific checkpoint path. |
| `env_cfg_type` | Robot or environment configuration key, for example `arx_x5`. |
| `action_type` | Action representation, usually `joint` or `ee`. |
| `seed` | Training or evaluation seed. |
| `policy_gpu_id` / `env_gpu_id` | GPU assignment for the policy process and environment process. |
| `policy_env_or_uv_path` | Policy runtime environment name or uv environment path. |
| `eval_env_conda_env` | Environment-client runtime environment. |

## 🔌 Deployment Flow

During evaluation, the environment process and policy process communicate through a policy server. This isolates simulator/robot dependencies from model dependencies and supports remote deployment.

For same-machine evaluation, use `eval.sh`; it starts the policy server, starts the environment client, and tears the policy server down when evaluation exits.

For split-machine deployment, start the policy server first on the model/GPU machine:

```bash
bash policy/<POLICY>/setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <policy_server_port> 0.0.0.0
```

Then start the environment client on the benchmark, simulator, or robot-side machine:

```bash
bash policy/<POLICY>/setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

`EVAL_ENV_TYPE` selects the environment-side backend:

- unset or `sim`: simulator-backed evaluation when the corresponding environment integration is installed.
- `debug`: offline shape and IO check.
- `real`: real-robot client path, only available where the corresponding hardware integration is installed.

## 📐 Standard Data Formats

XPolicyLab standardizes the observation and trajectory dictionaries passed between adapters, converters, and environment clients. Individual policies may convert this standard format into their upstream-native format.

All pose values use `[x, y, z, qw, qx, qy, qz]`. Images are RGB unless a policy README states otherwise.

<details>
<summary>Observation Data Format v1.0</summary>

```text
Observation Data Format v1.0
├── data_format_version                        string, optional
├── instruction / instructions                 string or list[str]
├── env_idx                                    int, optional for batched eval
├── additional_info/
│   └── frequency                              int, optional
├── vision/
│   ├── cam_head/
│   │   ├── color                              (H, W, 3), RGB
│   │   ├── depth                              (H, W) or (H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_state                   (DOF,), optional
    ├── left_ee_joint_state                    (EEF_DOF,), optional
    ├── left_ee_pose                           (7,), optional
    ├── left_tcp_pose                          (7,), optional
    ├── left_delta_ee_pose                     (7,), optional
    ├── right_arm_joint_state                  (DOF,), optional
    ├── right_ee_joint_state                   (EEF_DOF,), optional
    ├── right_ee_pose                          (7,), optional
    ├── right_tcp_pose                         (7,), optional
    ├── right_delta_ee_pose                    (7,), optional
    ├── arm_joint_state                        (DOF,), optional for single-arm robots
    ├── ee_joint_state                         (EEF_DOF,), optional for single-arm robots
    ├── ee_pose                                (7,), optional for single-arm robots
    ├── tcp_pose                               (7,), optional for single-arm robots
    ├── delta_ee_pose                          (7,), optional for single-arm robots
    └── mobile/                                optional
        ├── base_pose                          (7,)
        └── base_twist                         (6,), [vx, vy, vz, wx, wy, wz]
```

</details>

<details>
<summary>Trajectory Data Format v1.0</summary>

```text
Trajectory Data Format v1.0
├── data_format_version                        string, e.g. "v1.0"
├── instructions                               JSON-serialized list[str]
├── subtasks                                   JSON-serialized annotations, optional
├── additional_info/
│   └── frequency                              int
├── vision/
│   ├── cam_head/
│   │   ├── colors                             (T, H, W, 3), uint8 RGB or encoded stream
│   │   ├── depths                             (T, H, W) or (T, H, W, 1), optional
│   │   ├── intrinsic_matrix                   (3, 3) or (T, 3, 3), optional
│   │   ├── extrinsics_matrix                  (4, 4) or (T, 4, 4), optional
│   │   └── shape                              (2,) or (3,), optional
│   ├── cam_left_wrist/                        optional
│   ├── cam_right_wrist/                       optional
│   ├── cam_wrist/                             optional for single-arm robots
│   └── cam_third_view/                        optional
└── state/
    ├── left_arm_joint_states                  (T, DOF), optional
    ├── left_ee_joint_states                   (T, EEF_DOF), optional
    ├── left_ee_poses                          (T, 7), optional
    ├── left_tcp_poses                         (T, 7), optional
    ├── left_delta_ee_poses                    (T, 7), optional
    ├── right_arm_joint_states                 (T, DOF), optional
    ├── right_ee_joint_states                  (T, EEF_DOF), optional
    ├── right_ee_poses                         (T, 7), optional
    ├── right_tcp_poses                        (T, 7), optional
    ├── right_delta_ee_poses                   (T, 7), optional
    ├── arm_joint_states                       (T, DOF), optional for single-arm robots
    ├── ee_joint_states                        (T, EEF_DOF), optional for single-arm robots
    ├── ee_poses                               (T, 7), optional for single-arm robots
    ├── tcp_poses                              (T, 7), optional for single-arm robots
    ├── delta_ee_poses                         (T, 7), optional for single-arm robots
    └── mobile/                                optional
        ├── base_poses                         (T, 7)
        └── base_twists                        (T, 6), [vx, vy, vz, wx, wy, wz]
```

</details>

Useful converter helpers:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

`decode_image_bit` handles encoded RGB image streams. `get_robot_action_dim_info(env_cfg_type)` returns robot-specific `arm_dim` and `ee_dim` lists, so adapters do not need to hard-code action dimensions.

## 💾 Data And Checkpoints

By convention, converted datasets and checkpoints often use:

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
```

Policies may also accept explicit checkpoint paths or upstream-native layouts. Check the policy README before assuming a checkpoint name. For quick data downloads, use the demo workspace flow in [Quick Start](#-quick-start).

## ✅ Checks

Run static checks from the XPolicyLab repo root. Run `eval.sh` from `policy/<POLICY>/`, same as [Common Workflow](#-common-workflow).

Static checks:

```bash
git diff --check
bash -n policy/<POLICY>/*.sh
python -m py_compile policy/<POLICY>/model.py policy/<POLICY>/deploy.py
```

Adapter wiring check (no simulator required):

```bash
pip install -e .
export EVAL_ENV_TYPE=debug
cd policy/<POLICY>
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 \
  <policy_env_or_uv_path> <eval_env_conda_env>
```

For a quick smoke test, use `policy/demo_policy` and placeholder env names such as `base`. See [Quick Start](#-quick-start) for full argument meanings.

## 📬 Contact

Tianxing Chen: [chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)
