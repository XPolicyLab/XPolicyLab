![](./assets/logo.png)

<h1 align="center">XPolicyLab</h1>
<p align="center"><b>A unified adapter layer for robot policy training, serving, and evaluation.</b></p>

XPolicyLab wraps different robot policies behind a shared script and server interface. Each policy keeps its own dependencies and upstream code in `policy/<POLICY>/`; XPolicyLab standardizes the outer workflow for data processing, training, model serving, and RoboDojo/RoboTwin-style evaluation.

**Project leads:** Tianxing Chen, Tian Nian, Zijian Cai

**Links:** arXiv coming soon | user group coming soon | RoboDojo tutorial coming soon

## Quick Start

```bash
mkdir demo_env
cd demo_env
git clone git@github.com:Luminis-Sim/XPolicyLab.git
cd XPolicyLab

# Small demo bundle for smoke tests.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface demo

# Optional: raw HDF5 demos for policies that run their own conversion.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface hdf5

# Optional: LeRobot datasets for policies that train directly from LeRobot.
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v3.0
bash scripts/RoboDojo/download_robodojo_data.sh huggingface lerobot_v2.1
```

Data is downloaded to `../data` relative to this repo:

```text
demo_env/
├── data/
├── env_cfg/
└── XPolicyLab/
```

## Run a Policy

Always start from the policy guide:

```bash
cd XPolicyLab/policy/<POLICY>
less README.md
```

Most policies follow this shape:

```bash
# Install policy dependencies.
bash install.sh

# Optional: prepare policy-specific training data.
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [extra_args...]

# Optional: train.
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [extra_args...]

# Evaluate on one machine.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Use `EVAL_ENV_TYPE=debug` for an offline wiring check. Leave it unset, or set `EVAL_ENV_TYPE=sim`, for RoboDojo simulation.

## Policies

Top-level policy adapters live in `policy/`. Each top-level policy README contains the paper/repo link, installation notes, data processing, training, deployment, and important parameters.

<details>
<summary>Show policy catalog</summary>

| Category | Policies |
| --- | --- |
| VLA / WAM / foundation policies | [`A1`](policy/A1/README.md), [`AHA_WAM`](policy/AHA_WAM/README.md), [`Abot_M0`](policy/Abot_M0/README.md), [`Being_H05`](policy/Being_H05/README.md), [`Dexbotic_DM0`](policy/Dexbotic_DM0/README.md), [`Dexora_1B`](policy/Dexora_1B/README.md), [`DreamZero`](policy/DreamZero/README.md), [`EventVLA`](policy/EventVLA/README.md), [`FastWAM`](policy/FastWAM/README.md), [`GO1`](policy/GO1/README.md), [`GR00T_N17`](policy/GR00T_N17/README.md), [`GalaxeaVLA`](policy/GalaxeaVLA/README.md), [`GigaWorldPolicy`](policy/GigaWorldPolicy/README.md), [`H_RDT`](policy/H_RDT/README.md), [`Hy_Embodied_05_VLA`](policy/Hy_Embodied_05_VLA/README.md), [`InternVLA_A1`](policy/InternVLA_A1/README.md), [`LDA_1B`](policy/LDA_1B/README.md), [`LingBot_VA`](policy/LingBot_VA/README.md), [`LingBot_VLA`](policy/LingBot_VLA/README.md), [`Mem_0`](policy/Mem_0/README.md), [`MolmoACT2`](policy/MolmoACT2/README.md), [`Motus`](policy/Motus/README.md), [`OpenVLA_OFT`](policy/OpenVLA_OFT/README.md), [`Pi_0`](policy/Pi_0/README.md), [`Pi_05`](policy/Pi_05/README.md), [`Pi_0_Fast`](policy/Pi_0_Fast/README.md), [`RDT_1B`](policy/RDT_1B/README.md), [`RISE`](policy/RISE/README.md), [`SmolVLA`](policy/SmolVLA/README.md), [`Spatial_Forcing`](policy/Spatial_Forcing/README.md), [`Spirit_v15`](policy/Spirit_v15/README.md), [`TinyVLA`](policy/TinyVLA/README.md), [`X_VLA`](policy/X_VLA/README.md), [`X_WAM`](policy/X_WAM/README.md), [`Xiaomi_Robotics_0`](policy/Xiaomi_Robotics_0/README.md), [`starVLA`](policy/starVLA/README.md) |
| Imitation learning baselines | [`ACT`](policy/ACT/README.md), [`DP`](policy/DP/README.md) |
| Reference adapter | [`demo_policy`](policy/demo_policy/README.md) |

</details>

## Adapter Layout

A normal adapter looks like this:

```text
policy/<POLICY>/
├── README.md                    # source of truth for this policy
├── INSTALLATION.md              # optional; only for extra setup details
├── install.sh                   # install policy env
├── process_data.sh              # optional data conversion
├── train.sh                     # optional training
├── eval.sh                      # same-machine eval
├── setup_eval_policy_server.sh  # policy-side server
├── setup_eval_env_client.sh     # env-side client
├── deploy.yml                   # runtime config
├── deploy.py                    # server loop
└── model.py                     # Model adapter
```

`INSTALLATION.md` is kept only when `install.sh` is not enough, for example when a policy needs external checkpoints, system packages, or multiple environments.

## Common Arguments

| Parameter | Meaning |
| --- | --- |
| `bench_name` | Dataset / benchmark family, usually `RoboDojo`. |
| `task_name` | Evaluation task, for example `stack_bowls`. |
| `ckpt_name` | Data/run/checkpoint identifier. |
| `env_cfg_type` | Robot/environment config, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint` or `ee`. |
| `seed` | Training or evaluation seed. |
| `expert_data_num` | Optional episode cap for data processing. |
| `gpu_id` | Training GPU id or comma-separated ids. |
| `policy_gpu_id` | GPU for the policy server. |
| `env_gpu_id` | GPU for the simulator/client. |
| `policy_conda_env` / `policy_uv_env_path` | Policy runtime environment. |
| `eval_env_conda_env` | Environment-client runtime. |
| `protocol` | Policy-server transport in `deploy.yml`; defaults to `ws`. |

Most converted datasets use:

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>
```

Most checkpoints use:

```text
<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>
```

Some policies intentionally differ. Follow the policy README when it disagrees with this table.

<details>
<summary>Known command exceptions</summary>

| Policy | Difference |
| --- | --- |
| `EventVLA` | `train.sh` takes `<data_mix> <memory_ablation_mode> <keyframe_memory_policy>` and prints `RUN_ID` for eval. |
| `Hy_Embodied_05_VLA` | `process_data.sh` computes `norm_stats.pkl` from `<manifest_csv> <hdf5_dir> <output_pkl>`. |
| `Mem_0` | Supports `M1` / `Mn`, `execution` / `planning` / `both`, and optional Mn `planning_gpu_ids`. |
| `RDT_1B` | `process_data.sh` accepts `source_path`, `--overwrite`, `--skip-encode`, and `--gpu`. |
| `RISE` | `train.sh` has `advantage`, `policy`, and `all` stages. |
| `GalaxeaVLA`, `Hy_Embodied_05_VLA`, `Pi_0`, `Pi_05`, `Pi_0_Fast`, `Spatial_Forcing` | Use uv-style policy env paths. |
| `Dexora_1B`, `Spatial_Forcing` | Evaluation adapters only; no top-level training wrapper currently. |
| `X_WAM` | No top-level `install.sh`; read `policy/X_WAM/INSTALLATION.md`. |

</details>

## Deployment

Same-machine evaluation uses `eval.sh`. For split machines, start the policy server first and then the environment client:

Every policy `deploy.yml` should include `protocol: ws`. Keep `legacy_tcp` only for local debugging of an adapter that has not migrated to the websocket server/client path.

```bash
# Policy machine
bash policy/<POLICY>/setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <policy_server_port> 0.0.0.0

# Environment machine
bash policy/<POLICY>/setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>
```

| `EVAL_ENV_TYPE` | Behavior |
| --- | --- |
| unset / `sim` | RoboDojo simulation. |
| `debug` | Offline shape/IO check via `debug_env_client.py`. |
| `real` | One-shot real-robot eval via `task_env/real_env_client.py` (internal hardware path, not shipped in the open-source release). |

## Data Format

XPolicyLab uses `[x, y, z, qw, qx, qy, qz]` for poses. Images are RGB. Detailed trees are rarely needed unless you are writing a new adapter.

<details>
<summary>Observation and trajectory trees</summary>

Observation keys usually include:

```text
instruction / instructions
vision/cam_head/color
vision/cam_left_wrist/color
vision/cam_right_wrist/color
state/<robot_state_keys>
additional_info/frequency
```

Trajectory keys usually include:

```text
instructions
subtasks
vision/cam_head/colors
vision/cam_left_wrist/colors
vision/cam_right_wrist/colors
state/<robot_state_sequence_keys>
additional_info/frequency
```

Use these helpers when writing converters:

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

</details>

## Adding a New Policy

```bash
cd XPolicyLab
bash scripts/create_policy.sh <POLICY_NAME>
```

Then edit `policy/<POLICY_NAME>/README.md` first, because it should describe the exact command flow users will run. Keep examples copy-pasteable. If your policy has non-standard arguments, document them there instead of forcing them into the common template.

## Checks Before Push

```bash
git diff --check
bash -n policy/<POLICY>/*.sh
python -m py_compile policy/<POLICY>/*.py
```

For adapter wiring:

```bash
export EVAL_ENV_TYPE=debug
bash policy/<POLICY>/eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

The default policy-server protocol is `ws`. Install XPolicyLab in the **policy** environment:

```bash
pip install -e .   # from the XPolicyLab root
```

Set `protocol: legacy_tcp` only for adapters that still need the older
length-prefixed TCP client.

## Citation

```bibtex
Coming Soon
```

## Contact

Tianxing Chen: [chentianxing2002@gmail.com](mailto:chentianxing2002@gmail.com)
