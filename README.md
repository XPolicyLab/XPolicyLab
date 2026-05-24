<h1 align="center">XPolicyLab: A Unified Platform for Policy Deployment</h1>

XPolicyLab 是一个统一的策略训练与评测平台，旨在通过一套代码同时支持 RoboDojo 仿真环境与真机评测。

## 快速开始

### 环境配置与数据拉取

首先，克隆项目并拉取演示数据及环境配置：

```bash
mkdir demo_env
cd demo_env
git clone git@github.com:Luminis-Platform/XPolicyLab.git

# 拉取演示数据及环境配置，数据集格式为`data/${dataset_name}/${task_name}/${env_cfg}`
bbash scripts/download_data.sh
```
将内容移到`XPolicyLab`同级目录下。下面是示例的目录结构。
```text
demo_env/
├── data
│   └── {dataset_name}
│       └── {task_name}
│            └── {env_cfg}
│                 ├── data
│                 ├── preview_video
│                 ├── scene_layout
│                 ├── seed.txt
│                 └── traj_data
├── env_cfg
└── XPolicyLab
```

接着，创建并激活 Conda 环境，安装项目依赖：

```bash
cd XPolicyLab
conda create -n XPolicyLab python=3.10 -y
conda activate XPolicyLab
pip install -e .
```

## 数据与观测格式

在 XPolicyLab 中，位姿（Pose）数据的格式统一为 `[x, y, z, qw, qx, qy, qz]`。以下是详细的观测与数据格式说明。

### 观测格式 (Observation Data Format v1.0)

```text
Observation Data Format v1.0
├── data_format_version                        (Field)    string
├── instruction / instructions                 (Field)    string or list of strings
├── env_idx                                    (Field)    int
├── additional_info/                           (Group)
│   └── frequency                              (Field)    int
├── vision/                                    (Group)
│   ├── cam_head/                              (Group)
│   │   ├── color                              (Field)    (H, W, 3), 3通道为BGR
│   │   ├── depth                              (Field)    (H, W) or (H, W, 1)
│   │   ├── approximate_depth                  (Field)    optional
│   │   ├── intrinsic_matrix                   (Field)    (3, 3)
│   │   ├── extrinsics_matrix                  (Field)    (4, 4)
│   │   └── shape                              (Field)    (2,) or (3,)
│   ├── cam_left_wrist/                        (Group, optional)
│   ├── cam_right_wrist/                       (Group, optional)
│   ├── cam_wrist/                             (Group, optional for single-arm)
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
    ├── arm_joint_state                        (Field)    (DOF,), optional for single-arm
    ├── ee_joint_state                         (Field)    (EEF_DOF,), optional for single-arm
    ├── ee_pose                                (Field)    (7,), optional for single-arm
    ├── tcp_pose                               (Field)    (7,), optional for single-arm
    ├── delta_ee_pose                          (Field)    (7,), optional for single-arm
    └── mobile/                                (Group, optional)
        ├── base_pose                          (Field)    (7,)
        └── base_twist                         (Field)    (6,)
```

### 轨迹数据格式 (Trajectory Data Format v1.0)

```text
Trajectory Data Format v1.0
├── data_format_version                        (Dataset)  string, e.g. "v1.0"
├── instructions                               (Dataset)  JSON-serialized string list, task-level instructions
├── subtasks                                   (Dataset)  JSON-serialized list of stage annotations
├── additional_info/                           (Group)
│   └── frequency                              (Dataset)  int, control / recording frequency in Hz
├── vision/                                    (Group)
│   ├── cam_head/                              (Group)
│   │   ├── colors                             (Dataset)  (T, H, W, 3) uint8 BGR images
│   │   ├── depths                             (Dataset)  (T, H, W) or (T, H, W, 1), depth images
│   │   ├── approximate_depths                 (Dataset)  optional, approximated / processed depth images
│   │   ├── intrinsic_matrix                   (Dataset)  (3, 3) or (T, 3, 3)
│   │   ├── extrinsics_matrix                  (Dataset)  (4, 4) or (T, 4, 4)
│   │   └── shape                              (Dataset)  (2,) [H, W] or (3,) [H, W, C]
│   ├── cam_left_wrist/                        (Group, optional for dual-arm)
│   ├── cam_right_wrist/                       (Group, optional for dual-arm)
│   ├── cam_wrist/                             (Group, optional for single-arm)
│   └── cam_third_view/                        (Group, optional)
└── state/                                     (Group)
    ├── left_arm_joint_states                  (Dataset)  (T, DOF_L), optional
    ├── left_ee_joint_states                   (Dataset)  (T, EEF_DOF_L), optional
    ├── left_ee_poses                          (Dataset)  (T, 7), optional, [x, y, z, qw, qx, qy, qz]
    ├── left_tcp_poses                         (Dataset)  (T, 7), optional
    ├── left_delta_ee_poses                    (Dataset)  (T, 7), optional
    ├── right_arm_joint_states                 (Dataset)  (T, DOF_R), optional
    ├── right_ee_joint_states                  (Dataset)  (T, EEF_DOF_R), optional
    ├── right_ee_poses                         (Dataset)  (T, 7), optional, [x, y, z, qw, qx, qy, qz]
    ├── right_tcp_poses                        (Dataset)  (T, 7), optional
    ├── right_delta_ee_poses                   (Dataset)  (T, 7), optional
    ├── arm_joint_states                       (Dataset)  (T, DOF), optional for single-arm
    ├── ee_joint_states                        (Dataset)  (T, EEF_DOF), optional for single-arm
    ├── ee_poses                               (Dataset)  (T, 7), optional for single-arm
    ├── tcp_poses                              (Dataset)  (T, 7), optional for single-arm
    ├── delta_ee_poses                         (Dataset)  (T, 7), optional for single-arm
    └── mobile/                                (Group, optional)
        ├── base_poses                         (Dataset)  (T, 7), [x, y, z, qw, qx, qy, qz]
        └── base_twists                        (Dataset)  (T, 6), [vx, vy, vz, wx, wy, wz]
```

## 代码结构概览

为了使不同策略（Policy）的实现尽可能统一，并便于社区扩展，我们设计了一个兼具规范性与灵活性的框架。

请查看 `policy/demo_policy` 文件夹，这是一个教学演示模块，包含了完整的结构。其文件内容如下：

```text
demo_policy
├── deploy.py                       # 部署模型的流程（含串行与并行两种层级方案）
├── deploy.yml                      # 部署参数配置，参数将传入 model.Model 中，辅助用户定义模型参数并加载模型
├── eval.sh                         # 评测入口：编排 server + client（同机一键启动）
├── setup_eval_policy_server.sh     # 在 policy 环境中启动模型服务端，绑定 policy_server_port
├── setup_eval_env_client.sh        # 在 eval_env 环境中启动环境客户端，按 deploy.yml 的 eval_env 选择 debug/sim/real
├── __init__.py
├── install.sh                      # 环境安装脚本
├── model.py                        # 模型类，定义了模型导入、观测更新、动作获取以及重置逻辑
├── process_data.sh                 # 数据处理脚本，将 RoboDojo 数据转换为模型训练所需的格式
└── train.sh                        # 训练启动脚本
```

您也可以参考 `policy/DP`，这是一个简单且实现较为完善的策略示例。

## 接入自定义策略 (Policy)

### 1. 创建策略

在 XPolicyLab 根目录下，运行以下指令以创建新的策略模板：

```bash
bash create_policy.sh ${policy_name}
```
`create_policy.sh`会在`XPolicyLab/policy`创建新policy的目录，可以查看内容文件的相关注释以了解参数。

可将外部项目源码放在 `XPolicyLab/policy/${policy_name}` 下的独立子目录中。例如 `XPolicyLab/policy/DP/diffusion_policy` 。

拉取外部源码后，请删除该源码目录中的 `.git`，避免被 Git 识别为 submodule 。完成后**先进行一次提交保留源码快照**，再进行适配修改，以方便后续修改内容的对照。

**常见参数说明：**

训练与评测对参数的要求不同：

| 参数 | 训练 (`process_data.sh` / `train.sh`) | 评测 (`eval.sh`) |
|---|---|---|
| `ckpt_name` | **必填**。实验与产物标识，决定 `data/`、`checkpoints/` 子目录名 | **必填**。用来定位 checkpoint 子目录 |
| `task_name` | **不必填**。单任务训练时可与 `ckpt_name` 相同；cotrain 等多任务场景由 `process_data.sh` / `train.sh` 自行决定读哪些原始任务 | **必填**。指定仿真器中要跑的任务，传给环境客户端 |

其余参数：

- `dataset_name`: 数据集名称，目的是在 `data` 目录下区分不同项目的数据集，例如 RoboTwin 和 RoboDojo。
- `env_cfg_type`: 采集或评测的环境配置（包含本体信息等）。在 `demo_env/env_cfg_type` 中可以查看示例。教程中提供了两个示范数据：`dual_franka_panda`（双臂夹爪）和 `g1_inspire`（人形灵巧手）。
- `expert_data_num`: 训练使用的轨迹数量，参与 checkpoint 6 元组命名，为 `train.sh` 必填参数。
- `action_type`: 模型使用的数据类型（如 `ee` 或 `joint`）。这会影响使用的数据内容以及模型输入输出的维度。
- `seed`: 随机种子，便于多种子复现与对比。

> **命名约定：**
> - **处理后数据集**（`process_data.sh` 输出）固定为 5 元组：
>   `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`，落在 `policy/<policy_name>/data/` 下。
> - **训练产物**（`train.sh` 输出，对应 DP 的 `ckpt_setting`）固定为 6 元组：
>   `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`，落在 `policy/<policy_name>/checkpoints/` 下。
>
> 原始 RoboDojo 数据仍按 `data/${dataset_name}/${task_name}/${env_cfg}` 组织；`process_data.sh` 从其中读取一个或多个 `task_name`，但输出目录统一用 `ckpt_name` 命名。

#### 完善 install.sh
策略创建后，在根据原项目进行环境配置的同时完善install.sh

需要注意：确保每个policy的`install.sh`要分别对policy目录以及XPolicyLab的项目目录进行`pip install -e . `，以支持我们部分函数的调用。

建议参考 `demo_env/XPolicyLab/policy/DP/install.sh`，如下所示。

```bash
pip install -e . #policy目录下pip install -e .
cd ../../
pip install -e . #项目目录下pip install -e .
```

### 2. 完善数据处理 (`process_data.sh`)

您需要从 `demo_env/data` 中读取数据，并将其转换为模型所需的格式。我们已提供部分参考参数，您可以根据需要进行修改。建议参考 `demo_env/XPolicyLab/policy/DP/process_data.sh` 及其对应的 Python 文件。

在数据处理过程中，您可能会用到以下工具函数（参考 `policy/DP/diffusion_policy/process_data.py`）：

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import get_robot_action_dim_info, decode_image_bit
```

- `load_hdf5`: 输入路径读取数据。
- `decode_image_bit`: 将数据中的字节流解析还原为 NumPy 数组（支持单帧图片字节流及整条轨迹的字节流输入）。
- `get_robot_action_dim_info`: 输入 `env_cfg` 的名称（注意是字符串，不是字典），返回包含 `arm_dim` 和 `ee_dim` 列表的字典。

训练与部署中的图像处理应保持一致。要求分辨率统一为`640x480`
可参考 `policy/DP/diffusion_policy/process_data.py` 会将图像 resize 为 `640x480`，对应 HWC shape 为 `(480, 640, 3)`。

特别注意 BGR/RGB 通道顺序，请喂给模型 RGB 图像。

**维度信息示例：**

`env_cfg/robot/_robot_info.json`中有不同机器人的维度信息，可同过传入的`{env_cfg_type}`进行索引。

当列表长度为 1 时代表单臂机器人，长度为 2 时代表双臂机器人。在编写模型架构、定义输入输出以及处理数据时，请尽量利用此信息，以兼容未来不同自由度的末端执行器和机械臂任务。

```json
{
    "x5": {
        "arm_dim": [6],
        "ee_dim": [1]
    },
    "g1_inspire": {
        "arm_dim": [7, 7],
        "ee_dim": [12, 12]
    }
}
```

转换后数据默认保存在 `XPolicyLab/policy/${policy_name}/data` 下。

`demo_policy/process_data.sh` 的输入参数（5 个，与 LDA_1B / DP 对齐）：

| 序号 | 参数 | 含义 |
|---|---|---|
| 1 | `dataset_name` | 数据集名称（如 `RoboDojo`） |
| 2 | `ckpt_name` | **必填**。处理后数据与训练产物的标识；单任务时可与源 `task_name` 同名，cotrain 时可设为 `cotrain` 等 |
| 3 | `env_cfg_type` | 环境配置 / 本体类型（如 `g1_inspire`、`arx_x5`） |
| 4 | `expert_data_num` | 训练使用的轨迹数量 |
| 5 | `action_type` | 动作类型，`ee` / `joint` 等 |

输出目录名约定为 `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`。具体读哪些原始 `task_name`，由各 policy 的 `process_data.sh` / Python 脚本自行决定。

### 3. 训练模型

完善 `train.sh` 脚本。我们提供了一些演示参数。

`train.sh` 中包含 `seed` 参数，后续会进行不同 `seed` 的训练及测评，部分 `policy` 的源代码可能会把 `seed` 写死，需要注意且进行适配。

训练权重默认保存在 `XPolicyLab/policy/${policy_name}/checkpoints` 下；子目录名采用上文“命名约定”中的 6 元组 `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`。

`demo_policy/train.sh` 的输入参数（7 个）：

| 序号 | 参数 | 含义 |
|---|---|---|
| 1 | `dataset_name` | 数据集名称，与 `process_data.sh` 保持一致 |
| 2 | `ckpt_name` | **必填**。checkpoint 标识，决定输出子目录名；需与 `process_data.sh` 保持一致 |
| 3 | `env_cfg_type` | 环境配置 / 本体类型 |
| 4 | `expert_data_num` | 训练使用的轨迹数量 |
| 5 | `action_type` | 动作类型 |
| 6 | `seed` | 随机种子 |
| 7 | `gpu_id` | 训练所用 GPU id（多卡可写 `0,1,2,3`，由各 policy 自行处理） |

> 训练阶段不需要传入 `task_name`。若某 policy 的 `train.sh` 仍保留 `task_name` 参数，通常仅为兼容旧接口，或供脚本内部读取原始数据时使用。

### 4. 评测与部署

要支持评测，需要完善两个核心部分：模型推理支持 (`model.py`) 和控制流程 (`deploy.py`)。

我们提供了一个离线调试方案，默认环境为 `debug_policy_env.py`。该调试器会提供尺寸正确的观测数据（Observation），并根据返回的动作（Action）进行检查和模拟交互。调试通过后，即可尝试在仿真环境中运行。这个离线环境可通过将 `deploy.yml` 的 `eval_env` 设为 `debug` 来启动；切换到 `sim` 或 `real` 不需要修改 `eval.sh`。具体实现可参考 `XPolicyLab/policy/demo_policy/eval.sh` 及其 Python 文件。

#### 完善 `model.Model`

在 `model.py` 中，`Model` 类继承自 `ModelTemplate`，需要实现以下方法：

1. `__init__`: 接收 `model_cfg`（来自 `deploy.yml` 及 `eval.sh` 的覆盖参数）。
2. `update_obs`: 更新环境观测。
3. `update_obs_batch`: 批量更新多个环境的观测窗口。`obs_list` 是一个包含多个字典的列表，每个字典代表一个环境的观测，并指定了该观测的 `env_idx`。
4. `get_action`: 获取动作。要求返回动作字典，字典的 Key 决定了控制方式（例如，双臂指定 `left_arm_joint` 则使用关节控制，指定 `left_ee_pose` 则使用末端位姿控制）。可通过 `action_type` 参数控制返回内容，Key 需与观测状态中的 Key 类型一致。
5. `get_action_batch`: 批量获取动作，要求返回动作字典的列表。
6. `reset`: 重置模型状态。

> 详细实现可直接参考 `demo_policy/model.py` 中的代码及注释。

对`update_obs`和`update_obs_batch`的实现。若已实现了`update_obs_batch`，可以参考`DP`的形式直接实现`update_obs`。如果`update_obs_batch`在某个`policy`较难实现，可直接使用`for`循环`update_obs`。


#### 配置 `deploy.yml` 与三件套评测脚本

- **`deploy.yml`**: 指定模型部署所需的参数。部分参数可定义为 `null`，随后在 `setup_eval_policy_server.sh` 中通过 `--overrides` 覆盖。`eval_env` 字段（`debug` / `sim` / `real`）决定客户端走哪个 runner，无需改 `eval.sh`。`eval_batch` 控制是否走批量推理路径。
- **`eval.sh`**: 编排入口。分配一个空闲 `policy_server_port`，然后顺序拉起 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh`，并负责退出时清理 server。
- **`setup_eval_policy_server.sh`**: 在 `policy_conda_env` 中启动 `setup_policy_server.py`，绑定 `policy_server_port`（与 `policy_server_host`，默认 `localhost`，便于跨机部署）。
- **`setup_eval_env_client.sh`**: 在 `eval_env_conda_env` 中调用 `XPolicyLab/utils/setup_env_client.sh`，根据 `deploy.yml` 的 `eval_env` 转发到 `run_debug_env_client.sh` / `run_sim_env_client.sh` / `run_real_policy_client.sh`。

部署分为模型进程和环境进程，分别使用 `policy_conda_env/policy_uv_env_path` 和 `eval_env_conda_env`，通过 `policy_server_port` 通信，从而隔离环境配置。`policy_conda_env` 的实现可参考 DP，`policy_uv_env_path` 的实现可参考 PI_05。分别用 `policy_gpu_id` 和 `env_gpu_id` 分配模型和仿真的 GPU 占用，可参考 DP/demo_policy 脚本中只在子脚本内 `CUDA_VISIBLE_DEVICES="${policy_gpu_id}"` 的写法，而不是全局 `export CUDA_VISIBLE_DEVICES`。

`demo_policy/eval.sh` 的输入参数（11 个，默认顺序如下）：

| 序号 | 参数 | 含义 |
|---|---|---|
| 1 | `dataset_name` | 数据集名称，与训练时一致 |
| 2 | `task_name` | **必填**。仿真器中要跑的任务名，传给环境客户端 |
| 3 | `ckpt_name` | **必填**。用来反查 checkpoint 子目录；可与 `task_name` 不同（例如 `ckpt_name=cotrain` 同时在多个 `task_name` 上评测） |
| 4 | `env_cfg_type` | 环境设置，在RoboDojo中为`arx_x5` |
| 5 | `expert_data_num` | 训练使用的轨迹数量|
| 6 | `action_type` | 动作类型 |
| 7 | `seed` | 随机种子 |
| 8 | `policy_gpu_id` | 模型推理服务端使用的 GPU id |
| 9 | `env_gpu_id` | 环境客户端（仿真）使用的 GPU id |
| 10 | `policy_conda_env` | 模型服务端激活的 conda 环境名（PI_05 这类基于 uv 的策略可在脚本内改用 `policy_uv_env_path`） |
| 11 | `eval_env_conda_env` | 环境客户端激活的 conda 环境名 |

`eval.sh` 默认按 `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>` 在 `policy/<policy_name>/checkpoints/` 下找训练产物。

```bash
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        policy_server_port="${policy_server_port}" \
        policy_server_host="${policy_server_host}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
    &
SERVER_PID=$!
```


> **跨机部署**：把 `setup_eval_policy_server.sh` 放在带 GPU 的机器上后台运行，再在仿真机调用 `setup_eval_env_client.sh ... <policy_server_port> <policy_server_ip>` 即可。两侧只需指向同一个 `policy_server_ip:policy_server_port`，不必同机。

#### 部署逻辑 (`deploy.py`)

建议阅读 `demo_policy.py` 中的实现与注释以理解逻辑。关键点如下：

1. `TASK_ENV.is_episode_end()`: 判断当前环境是否全部结束。
2. `model_client.call`: `func_name` 为字符串，传入观测数据。两者共同序列化后通过端口通信，调用模型侧对应的函数并传递参数。

### 5. 仿真部署与调试体验

当一切调试完毕后，把 `deploy.yml` 里的 `eval_env` 从 `debug` 改为 `sim`（或 `real`）即可在仿真/真机环境中部署，无需改动 `eval.sh`、`setup_eval_policy_server.sh`、`setup_eval_env_client.sh`。`setup_env_client.sh` 会根据该字段自动转发到对应的 runner（`run_sim_env_client.sh` / `run_real_policy_client.sh`）。

您可以通过以下流程体验调试器：

```bash
cd policy/demo_policy
# dataset_name task_name ckpt_name env_cfg_type expert_data_num action_type seed policy_gpu_id env_gpu_id policy_conda_env eval_env_conda_env
bash eval.sh RoboDojo handover_bottle_and_put_into_dustbin demo_ckpt g1_inspire 50 ee 0 0 0 XPolicyLab XPolicyLab
```
