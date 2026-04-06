<h1 align="center">XPolicyLab: A Unified Platform for Policy Deployment</h1>

XPolicyLab 是一个统一的策略训练与评测平台，旨在通过一套代码同时支持 RoboDojo 仿真环境与真机评测。

## 快速开始

### 环境配置与数据拉取

首先，克隆项目并拉取演示数据及环境配置：

```bash
mkdir demo_env
cd demo_env
git clone git@github.com:Luminis-Platform/XPolicyLab.git

# 拉取演示数据及环境配置
bash clone_demo_tutorial.sh
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
│   │   ├── color                              (Field)    (H, W, 3)
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
│   │   ├── colors                             (Dataset)  (T, H, W, 3) uint8 RGB images
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

为了使不同策略（Policy）的实现尽可能统一，并便于社区扩展，我们设计了一个兼具规范性与灵活性的框架。您可以参考 `policy/DP`，这是一个简单且实现较为完善的策略示例。

请查看 `policy/demo_policy` 文件夹，这是一个教学演示模块，包含了完整的结构。其文件内容如下：

```text
demo_policy
├── deploy.py       # 部署模型的流程（含串行与并行两种层级方案）
├── deploy.yml      # 部署参数配置，参数将传入 model.Model 中，辅助用户定义模型参数并加载模型
├── eval.sh         # 评测启动脚本
├── __init__.py
├── install.sh      # 环境安装脚本 
├── model.py        # 模型类，定义了模型导入、观测更新、动作获取以及重置逻辑
├── process_data.sh # 数据处理脚本，将 RoboDojo 数据转换为模型训练所需的格式
└── train.sh        # 训练启动脚本  
```

## 接入自定义策略 (Policy)

### 1. 创建策略

在 XPolicyLab 根目录下，运行以下指令以创建新的策略模板：

```bash
bash create_policy.sh ${policy_name}
```

**常见参数说明：**
- `task_name`: 任务名称。
- `env_cfg_type`: 采集或评测的环境配置（包含本体信息等）。在 `demo_env/env_cfg_type` 中可以查看示例。教程中提供了两个示范数据：`dual_franka_panda`（双臂夹爪）和 `g1_inspire`（人形灵巧手）。
- `expert_data_num`: 训练使用的轨迹数量。
- `action_type`: 模型使用的数据类型（如 `ee` 或 `joint`）。这会影响使用的数据内容以及模型输入输出的维度。

> **建议：** 模型的 Checkpoint 命名应包含上述所有参数，以便后续唯一指定加载。当然，也支持用户自定义命名。

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

**维度信息示例：**
当列表长度为 1 时代表单臂机器人，长度为 2 时代表双臂机器人。在编写模型架构、定义输入输出以及处理数据时，请尽量利用此信息，以兼容未来不同自由度的末端执行器和机械臂任务。

```json
{
    "x5": {
        "arm_dim": [6],
        "ee_dim": [1]
    },
    "aloha_agilex": {
        "arm_dim": [6, 6],
        "ee_dim": [1, 1]
    }
}
```

### 3. 训练模型

完善 `train.sh` 脚本。我们提供了一些演示参数，您可以根据实际需求进行调整。

### 4. 评测与部署

要支持评测，需要完善两个核心部分：模型推理支持 (`model.py`) 和控制流程 (`deploy.py`)。

我们提供了一个离线调试方案，默认环境为 `debug_policy_env.py`。该调试器会提供尺寸正确的观测数据（Observation），并根据返回的动作（Action）进行检查和模拟交互。调试通过后，即可尝试在仿真环境中运行。

#### 完善 `model.Model`

在 `model.py` 中，`Model` 类继承自 `ModelTemplate`，需要实现以下方法：

1. `__init__`: 接收 `model_cfg`（来自 `deploy.yml` 及 `eval.sh` 的覆盖参数）。
2. `update_obs`: 更新环境观测。
3. `update_obs_batch`: 批量更新多个环境的观测窗口。`obs_list` 是一个包含多个字典的列表，每个字典代表一个环境的观测，并指定了该观测的 `env_idx`。
4. `get_action`: 获取动作。要求返回动作字典，字典的 Key 决定了控制方式（例如，双臂指定 `left_arm_joint` 则使用关节控制，指定 `left_ee_pose` 则使用末端位姿控制）。可通过 `action_type` 参数控制返回内容，Key 需与观测状态中的 Key 类型一致。
5. `get_action_batch`: 批量获取动作，要求返回动作字典的列表。
6. `reset`: 重置模型状态。

> 详细实现可直接参考 `demo_policy/model.py` 中的代码及注释。

#### 配置 `deploy.yml` 与 `eval.sh`

- **`deploy.yml`**: 指定模型部署所需的参数。部分参数可定义为 `null`，随后在 `eval.sh` 中进行覆盖。
- **`eval.sh`**: 部署时分为模型进程和环境进程，分别使用 `policy_conda_env` 和 `eval_env_conda_env`。两者通过 `FREE_PORT` 进行通信，从而隔离环境配置。

您只需修改脚本开头的参数定义，并在启动 Server 部分添加 `overrides` 参数以覆盖 `deploy.yml` 中的配置：

```bash
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
    &
SERVER_PID=$!
```

#### 部署逻辑 (`deploy.py`)

建议阅读 `demo_policy.py` 中的实现与注释以理解逻辑。关键点如下：

1. `TASK_ENV.is_episode_end()`: 判断当前环境是否全部结束。
2. `model_client.call`: `func_name` 为字符串，传入观测数据。两者共同序列化后通过端口通信，调用模型侧对应的函数并传递参数。

### 5. 仿真部署与调试体验

当一切调试完毕后，将 `eval.sh` 最后一行的 `run_debug_policy_client` 替换为 `run_policy_client`，即可真正在仿真环境中进行部署。

您可以通过以下流程体验调试器：

```bash
cd policy/demo_policy
bash eval.sh align_blocks dual_x5 50 ee 0 0 XPolicyLab XPolicyLab
```
