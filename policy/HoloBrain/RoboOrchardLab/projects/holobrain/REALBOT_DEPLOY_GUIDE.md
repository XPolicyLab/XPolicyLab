# HoloBrain-0 真实机器人部署指南

本指南介绍如何将训练好的 HoloBrain 模型部署到物理机器人上，包括硬件搭建（含相机标定）以及在真实机器人上启动推理。

> [!NOTE]
> 本指南只关注**部署**。如果你还需要先录制数据、打包数据集或训练模型，请参阅 **[真实机器人流水线指南](REALBOT_PIPELINE_GUIDE.md)**。

## 前置条件

开始之前，请确认你已经完成以下准备：

- **已训练并导出 HoloBrain 模型**：参见主 README 中的 [导出模型](README.md#5-导出模型处理器和-pipeline)
- **已安装 Real Robot App**：安装说明见 [HoloBrain Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain)

## 部署概览

部署流程分为三个阶段：

| 阶段 | 章节 | 你需要做什么 |
| --- | --- | --- |
| 1 | [硬件搭建](#1-硬件搭建) | 组装机械臂、相机、支架，并完成相机标定 |
| 2 | [启动推理服务](#2-启动推理服务) | 在 GPU 机器上启动模型服务 |
| 3 | [配置并启动机器人应用](#3-配置并启动真实机器人应用) | 配置机器人应用并启动自主运行 |

> [!NOTE]
> 本指南涉及两个仓库：
> - **RoboOrchardLab**（本仓库）：模型训练、导出和推理服务（`projects/holobrain/`）
> - **[Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain)**：机器人控制、相机和启动配置（`projects/HoloBrain/`）
>
> 为了避免混淆，下文所有文件路径都会标注其所属仓库。

## 1. 硬件搭建

开始前，请准备好以下硬件组件。

| 组件 | 规格 | 数量 |
| --- | --- | --- |
| 机械臂 | Agilex Piper | 2 |
| 相机 | Intel RealSense D435i | 3 |

### 1.1 Grasp Anything 任务的硬件搭建

下表展示了我们在 Grasp Anything 实验中使用的参考配置。如果你的配置不同，需要自行进行相机标定（见下方 [相机标定](#12-相机标定)）。

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| 机械臂间距 | 60 cm | 两个 Piper 机械臂 base-link 中心之间的距离 |
| 机械臂安装高度 | 0 cm（桌面高度） | 两个机械臂都安装在桌面上 |
| 中间环境相机高度 | 50 cm | 中间环境相机光心高出桌面的高度 |
| 腕部相机支架 | 见 [bracket files](./assets/) | 用于将 D435i 安装到每个 Piper 腕部的 3D 打印支架 |

> [!NOTE]
> 所有高度值均相对于桌面测量。你也可以直接从 Agilex 购买现成相机支架：[link](https://item.taobao.com/item.htm?id=974751867024&mi_id=0000hTzpIhXxYygNl_eTQu9bu2vBIAj8rpzN26HxSygPPCo&spm=a21xtw.29178619.0.0&xxc=shop)。

### 1.2 相机标定

准确的相机外参，即每个相机相对于机器人的位置和姿态，对于模型将视觉观测正确映射到机器人动作至关重要。

如果你的硬件**严格按照上述配置搭建**，可以直接使用我们在 [HuggingFace processor](https://huggingface.co/HorizonRobotics/HoloBrain_v0.0_GD/blob/main/real_world_agilex_grasp_anything_processor.json) 中提供的相机外参，不需要额外标定。

如果你使用了**自定义硬件配置**（例如相机位置、机械臂间距或安装角度不同），则需要进行手眼标定。请参考：

👉 [Hand-Eye Calibration Tool](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain/handeye_calib)

## 2. 启动推理服务

硬件搭建并完成相机标定后，第一步是启动 HoloBrain 推理服务。该服务会加载你训练好的模型，并通过网络提供动作预测；真实机器人应用（第 3 节）会连接到它以获取实时命令。

```bash
cd projects/holobrain
# RoboOrchardLab · projects/holobrain/scripts/inference_server.py
python3 scripts/inference_server.py \
    --model_dir "/your/model_dir" \
    --port 2000 \
    --server_name holobrain \
    --num_joints 7 \
    --valid_action_step 64
```

> [!TIP]
> `model_dir` 应指向导出的模型目录（见主 README 中的 [导出模型](README.md#5-导出模型处理器和-pipeline)）。你也可以直接使用 HuggingFace 模型路径，例如 `hf://HorizonRobotics/HoloBrain_v0.0_Qwen`。

## 3. 配置并启动真实机器人应用

真实机器人应用负责相机采集、机械臂控制，以及与推理服务通信。它位于单独的仓库中，完整安装说明见 [HoloBrain Real Robot App](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain)。

启动前，你需要根据自己的硬件配置应用。下面的小节逐项说明配置内容。

### 3.1 配置机械臂 CAN ID

每个 Piper 机械臂通过 CAN（Controller Area Network）总线通信。你需要识别正确的 CAN 端口，并将其映射到左/右机械臂。

1. 发现所有可用 CAN 端口：

   ```bash
   # Real Robot App · projects/HoloBrain/teleop/find-all-can-port.sh
   bash teleop/find-all-can-port.sh
   ```

2. 根据输出，更新 `teleop/templates/rename-can.sh`（Real Robot App）中的 CAN 端口映射，使其匹配你的机械臂。

### 3.2 配置相机 ID

每个 Intel RealSense 相机都有唯一序列号。编辑 `launch/templates/launch.yaml`（Real Robot App），在 `environment` 部分更新相机序列号，使其匹配你的三个 D435i 相机（左、右、中）。

> [!TIP]
> 可以运行 `rs-enumerate-devices` 查看相机序列号。该命令行工具包含在 [Intel RealSense SDK (librealsense)](https://github.com/IntelRealSense/librealsense) 中。

### 3.3 添加推理 Tmux Session

将以下 tmux session 添加到 `launch/templates/launch.yaml`（Real Robot App），使推理客户端自动启动：

```yaml
  - window_name: inference
    layout: tiled
    shell_command_before:
      - cd $DOCKER_ROBO_ORCHARD_PATH/projects/holobrain
    panes:
      - shell_command:
        - CMD="bash inference/launch_async_infer.sh"
        - history -s "$CMD"
        - eval "$CMD"
```

### 3.4 配置推理设置

HoloBrain 支持两种推理模式。请选择适合你需求的模式：

| 模式 | 行为 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **Sync** | 机器人等待每次预测完成后再执行 | 配置更简单 | 较慢，动作之间可能出现停顿 |
| **Async** | 机器人在计算下一次预测时继续执行当前动作 | 运动更平滑、更快 | 需要额外配置 RTC 插件 |

> [!TIP]
> **真实部署推荐使用 Async 模式**，它能带来更平滑的运动和更好的任务表现。初始测试或调试时可以使用 Sync 模式。

#### 同步推理

同步模式下，机器人会等待每次预测完成后再执行下一段动作。

- 修改 `inference/gen_sync_config.py`（Real Robot App），填入你的推理服务地址和模型设置。关键字段包括 `server_url`（推理服务地址）和 `infer_frequency`（预测频率，单位 Hz）。

#### 异步推理

异步模式下，机器人会在下一次预测计算过程中继续执行当前动作，因此运动更平滑、更快。

- 修改 `inference/gen_async_config.py`（Real Robot App），填入你的推理服务地址和模型设置。关键字段包括 `server_url`（推理服务地址）、`infer_frequency` 和 `delay_horizon`。
- 将 RTC（Real-Time Correction）插件添加到导出模型的配置文件 `model.config.json` 中（RoboOrchardLab，位于模型导出目录内，例如 `model_export_path/model/model.config.json`）：
    ```json
    "async_inference_plugin": {
        "type": "robo_orchard_lab.models.rtc_plugin.rtc_plugin:RTCInferencePlugin"
    },
    ```

> [!NOTE]
> RTC 插件只在推理时需要，训练期间不会使用。

### 3.5 启动推理

全部配置完成后，使用启动脚本启动 Real Robot App：

```bash
# Real Robot App · projects/HoloBrain/launch/start.sh
bash launch/start.sh
```

然后：

1. 在应用 UI 中打开**控制面板**。
2. 将 **Control Mode** 设置为 `Auto`。
3. 在 **Inference Control** 下点击 `Start`，开始自主运行。

> [!TIP]
> 如果点击 Start 后机器人没有移动，请确认推理服务（第 2 节）正在运行，并且机器人应用所在机器可以访问该服务。
