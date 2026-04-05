<h1 align="center">XPolicyLab: A unified platform for policy deployment</h1>

这是一个统一的策略训练与评测platform，可以以一套代码同时支持RoboDojo仿真与真机评测。


# 数据结构

```
mkdir demo_env
cd demo_env
git clone git@github.com:Luminis-Platform/XPolicyLab.git
 # 拉取我们的数据格式以及env_cfg
# 环境安装
cd XPolicyLab
conda create -n XPolicyLab python=3.10 -y
conda activate XPolicyLab
pip install -e .
```

# 了解我们的数据与观测格式

数据格式（数据与观测），其中pose是[x, y, z, qw, qx, qy, qz]

<details>
<summary>观测格式</summary>
```
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

</details>

<details>
<summary>数据格式</summary>
```
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
</details>

# 了解我们的代码结构

> 为了使得我们不同policy的实现尽可能趋同，使得整个community更容易扩展，我们设置了一个相对统一且具备一定灵活性的框架，step by step地去推进接入，也可以参考一下`policy/DP`，这是一个简单且实现比较完善的policy


请移步到`policy/demo_policy`文件夹，这是一个教学演示文件夹，包含了整个结构，以下是文件内容：
```
demo_policy
├── deploy.py       # 部署模型的流程（含串行与并行两种层级方案）
├── deploy.yml      # 部署传入参数，其中可以写一些参数，最后会传入model.Model中，辅助用户去定义模型的参数，从而load模型
├── eval.sh         # 评测启动脚本
├── __init__.py
├── install.sh      # 环境安装脚本 
├── model.py        # 模型类，定义了模型导入，观测更新，动作获取，以及重置
├── process_data.sh # 数据处理脚本，将RoboDojo数据转换为模型训练需要的格式
└── train.sh        # 训练启动脚本  
```

# 开始接入你的policy

首先在XPolicyLab根目录下，运行以下指令以实现创建策略
```
bash create_policy.sh ${policy_name}
```

了解我们的数据结构

了解几个常见参数:
1. task_name: 任务名
2. env_cfg: 代表采集/评测的环境配置，包括本体信息等，在`demo_env/env_cfg`中可以看到一些demo，在tutorial中使用的是`franka_`
3. expert_data_num: 训练使用多少条轨迹
4. action_type: 模型使用的数据类型，比如`ee`或者`joint`，这会影响到使用数据中什么数据，以及模型输入输出的维度

> 建议模型的checkpoint命名上考虑到以上全部参数，方便后续唯一指定load，当然也支持用户自己定义

## 完善`process_data.sh`
从数据中`demo_env/data`中读数据然后转换成你的模型需要的数据格式，目前已经给你写了部分参数，只是提供参考，是可以改的，可以看看`demo_env/XPolicyLab/policy/DP/process_data.sh`以及对应的python文件来参考示范。

以下是你可能会用到的函数，可以参考`policy/DP/diffusion_policy/process_data.py`，其中`load_hdf5`输入path读数据，`get_robot_action_dim_info`输入你的env_cfg的名字（注意，不是dict），可以获得一个字典，`decode_image_bit`可以将数据中字节流解析还原成np array（同时支持单帧图片字节流以及整个轨迹的字节流输入）。

```
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import get_robot_action_dim_info, decode_image_bit
```


`get_robot_action_dim_info`会从类似以下的字典中返回对应的内容，包含`arm_dim`列表以及`ee_dim`列表，当两者长度都是1的时候，代表单臂机器人，当两者长度都是2的时候，代表双臂机器人，写模型架构、模型输入输出定义，数据处理尽可能用上这个信息，未来可能会有单臂的任务，以及已经有不同自由度的末端执行器以及机械臂了。

```
"x5": {
    "arm_dim": [6],
    "ee_dim": [1]
},
"aloha_agilex": {
    "arm_dim": [6, 6],
    "ee_dim": [1, 1]
},
```

## 处理完数据后，支持训练

完善`train.sh`，给了一些demo参数，可以按照你自己的需求来改动

## 训练完毕后，支持评测

支持评测需要完善两个东西，一个是模型本身的推理支持`model.py`，一个是控制流程`deploy.py`。
我们此处实现了一个离线调试方案，默认环境用的是`debug_policy_env.py`，这个debuger会给你提供尺寸正确的obs，并根据你返回的action进行检查以及虚假交互，如果正确运行完毕后证明你调整通了第一步，接下来就可以尝试上仿真了。

### model.Model

可以看到`model.py`中的`Model`中继承自`ModelTemplate`，需要支持以下几个函数：

1. __init__: 传入model_cfg，来自`deploy.yml`以及eval.sh中的覆盖
2. update_obs：更新环境
3. update_obs_batch: 按照batch更新多个环境的观测窗口，obs_list是一个list，每个内容是一个dict，是一个环境的观测，其中也制定了该obs的`env_idx`
4. get_action: 获取动作，要求返回动作字典
5. get_action_batch: 获取动作list，要求返回动作字典list
6. reset:重置模型

可以直接进去看`demo_policy/model.py`的实现，都加了注释

### deploy.yml以及eval.sh

`deploy.yml`指定了模型部署需要的参数，其中可以给部分定义成null，然后在`eval.sh`中进行override

`eval.sh`: 部署时分两个进程，一个是模型进程，一个是环境进程，分别使用policy_conda_env以及eval_env_conda_env，然后通过FREE_PORT进行通信，这样可以隔离两个进程的环境配置。
这里只需要修改一开始的参数定义，然后在启动server部分添加overrides参数，这个会覆盖`deploy.yml`中的参数。

```
PYTHONWARNINGS=ignore::UserWarning \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        task_name="${task_name}" \
        env_cfg="${env_cfg}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
    &
SERVER_PID=$!
```

对于deploy.py而言，可以看`demo_policy.py`中的实现以及注释，理解逻辑，其中重要的是：

1. TASK_ENV.is_episode_end()：告诉你当前环境是否全部结束
2. model_client.call的func_name是str，obs传入obs，两者共同序列化并通过port进行通信，调用模型侧对应的函数并传入参数


当一切调试完毕后，可以将`eval.sh`中最后一行的`run_debug_policy_client`换成`run_policy_client`