# HoloBrain-0 真实机器人流水线指南

本指南以 **Grasp Anything** 任务为贯穿示例，带你完成完整的真实机器人流程。完成后，你将获得录制（或下载）的示教数据、可用于训练的数据集、训练好的 HoloBrain 模型，以及部署到物理机器人的准备工作。

**什么是 Grasp Anything？** 双臂 Agilex Piper 机器人从桌面抓取任意物体，并将其放入篮子中。该配置使用三个 Intel RealSense 相机（左、右、中）观察工作空间。在代码库中，这个任务名为 `place_objects_to_basket`。

## 前置条件

- **安装：** 按照 [主 README](README.md#-quick-start) 安装项目。
  ```bash
  cd /path/to/robo_orchard_lab
  make version
  pip install ".[holobrain_0]"
  ```
- **硬件（用于录制和部署）：** 双 Agilex Piper 机械臂 + 三个 Intel RealSense 相机。完整硬件细节见 [真实机器人部署指南](REALBOT_DEPLOY_GUIDE.md)。
- **工作目录：** 下方所有命令默认你位于 `projects/holobrain` 目录：
  ```bash
  cd projects/holobrain
  ```

---

## 流水线概览

流水线由五个顺序阶段组成。每个阶段依赖上一个阶段的输出：

| 阶段 | 你需要做什么 | 你会得到什么 |
| :--- | :--- | :--- |
| **1.&nbsp;数据录制** | 遥操作机器人示范任务，同时系统记录所有相机和关节数据。 | 原始 `.mcap` 录制文件，每条示教一份。见下方 [什么是 `.mcap` 文件？](#什么是-mcap-文件)。 |
| **2.&nbsp;数据打包** | 将原始录制转换为训练可用、支持高效随机访问的数据集。 | 结构化数据集：分片 `.arrow` 文件，以及训练流水线所需元数据。见 [预期输出结构](#预期输出结构)。 |
| **3.&nbsp;数据检查** | 可视化验证打包数据是否正确，训练输入是否合理。 | 可检查的回放视频（`.mp4`）和重建后的 `.mcap` 文件，可在 [Foxglove](https://foxglove.dev/) 中查看。 |
| **4.&nbsp;模型训练** | 使用配置文件在打包数据集上训练 HoloBrain 模型。 | 训练 checkpoint，包括模型权重（`.safetensors`）和模型配置（`.config.json`），以及定义数据前/后处理流程的 processor 配置（`*_processor.json`）。可用于导出。 |
| **5.&nbsp;部署** | 导出训练好的模型，并启动连接物理机器人的推理服务。 | 正在运行的推理服务，向机器人发送实时动作命令。 |

### 应该从哪里开始？

并非所有人都需要从零开始。可根据你已有的材料选择入口：

```text
什么都没有？
  └─➔ 从步骤 1（数据录制）开始

已有原始 .mcap 文件（例如从 HuggingFace 下载）？
  └─➔ 从步骤 2（数据打包）开始

已有打包好的 .arrow 数据集？
  └─➔ 从步骤 4（模型训练）开始

已有训练 checkpoint（或导出模型）？
  └─➔ 从步骤 5（部署）开始
```

### 捷径：从 HuggingFace 下载数据

如果你不想自己录制数据，可以从 HuggingFace 下载预录制数据集。下面的下载命令使用 `hf` CLI，可通过 `pip install -U "huggingface_hub[cli]"` 安装。

#### 选项 A：下载原始 `.mcap` 文件 → 跳到步骤 2（数据打包）

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "raw_data/place_objects_to_basket/*.mcap" \
    --local-dir ./data
```

下载后，目录结构应如下：
```text
data/raw_data/place_objects_to_basket/
  data0.mcap
  data1.mcap
  ...
```

接着继续执行**步骤 2（数据打包）**。

#### 选项 B：下载打包好的 `.arrow` 数据集 → 跳到步骤 4（模型训练）

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "arrow_dataset/place_objects_to_basket/*" \
    --local-dir ./data
```

下载后，目录结构应如下：
```text
data/arrow_dataset/place_objects_to_basket/
  part-00000/
    data-00000-of-00070.arrow
    ...
    dataset_info.json
    meta_db.duckdb
    state.json
```

然后直接跳到**步骤 4（模型训练）**。

你也可以浏览 [Real-World Dataset collection](https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/tree/main)，查看其他可用任务。

---

## 1. 数据录制

本步骤生成训练所需的**原始轨迹数据**。如果你已经从 HuggingFace 下载数据，可以完全跳过本步骤。

### 什么是 `.mcap` 文件？

`.mcap` 文件是 ROS 消息包，用于将同步的传感器流和机器人状态记录到单个文件中。对于 Grasp Anything 任务，每次录制包含：

- **相机数据：** 三个 RealSense 相机（左、右、中）的 RGB 和深度图像，以及它们的标定参数。
- **机器人数据：** 两个 Piper 机械臂的关节状态和动作命令。

### 可视化 `.mcap` 文件

你可以使用 [Foxglove](https://foxglove.dev/) 检查 `.mcap` 文件。若要使用本项目的预配置视图，请从 HuggingFace 下载示例布局：[Example Foxglove Layout](https://huggingface.co/datasets/HorizonRobotics/Real-World-Dataset/blob/main/visualization/arrow_foxglove_layout.json)

在 Foxglove 中，进入 **Layout -> Import layout**，加载下载的 `.json` 文件，然后打开你的 `.mcap` 文件。

### 录制配置

一次典型录制会包含：

1. 安装三个相机以观察工作空间。
2. 通过 CAN 总线连接两个 Piper 机械臂。
3. 使用遥操作示范抓取和放置任务，同时系统记录所有数据流。

详细录制说明请参考 [数据采集指南](https://github.com/HorizonRobotics/RoboOrchard/tree/master/projects/HoloBrain)。

### 什么时候可以跳过本步骤？

- 你已经从 HuggingFace 或其他来源下载了 `.mcap` 文件。
- 你已经有之前录制的原始数据。

原始数据布局示例：
```text
data/raw_data/
    data0.mcap
    data1.mcap
    ...
```

---

## 2. 数据打包

本步骤会将原始 `.mcap` 录制转换为训练流水线期望的 **Arrow 列式格式**。训练 dataloader 需要标准化、可索引的格式以高效随机访问数据；而原始 `.mcap` 文件是顺序数据流，训练时无法高效采样，因此需要转换。

### URDF：它是什么？在哪里获取？

打包器需要一个 URDF（Unified Robot Description Format）文件，用来描述机器人的运动学链，包括关节限制、连杆长度等。对于 Grasp Anything 的双臂 Piper 配置，请使用：

```text
./urdf/piper_description_dualarm.urdf
```

如果本地没有该 URDF 文件，可以从 HuggingFace 下载：

```bash
hf download HorizonRobotics/Real-World-Dataset \
    --repo-type dataset \
    --include "urdf/piper_description_dualarm.urdf" \
    --local-dir .
```

> [!NOTE]
> 如果你使用不同机器人，需要提供自己的 URDF 文件。

### 打包命令（Grasp Anything）

```bash
python3 -m robo_orchard_lab.dataset.horizon_manipulation.packer.mcap_arrow_packer \
    --input_path "./data/raw_data/place_objects_to_basket/*.mcap" \
    --output_path "./data/arrow_dataset/place_objects_to_basket" \
    --urdf_path "./urdf/piper_description_dualarm.urdf"
```

> [!TIP]
> 如果输出目录已经存在，可添加 `--force_overwrite` 覆盖。

打包过程中，工具会：

- 直接从 `.mcap` 文件中提取 ROS 消息
- 同步多模态传感器流，同时丢弃静态帧
- 将轨迹转换为标准 RO 数据集格式
- 为训练流水线编译必要元数据（保存为 `.duckdb`）

### 预期输出结构

```text
data/arrow_dataset/
  place_objects_to_basket/
    data-00000-of-00070.arrow
    data-00001-of-00070.arrow
    ...
    state.json
    dataset_info.json
    meta_db.duckdb
```

每个目录对应一次录制 session。分片 `.arrow` 文件包含所有 episode。`state.json` 文件用于让数据加载器发现有效数据集，这对故障排查很重要（见 FAQ）。

---

## 3. 数据检查

本步骤用于验证打包数据集的正确性和质量。对于新数据集或修改过数据流水线后的情况，**强烈建议执行**。如果是已知正常数据的重复运行，可以直接跳到步骤 4。

### 3.1 打包结果检查

**目的：** 确认 `.arrow` 打包过程准确保留了所有传感器流、动作和时间戳，没有数据丢失。

**输出：** 重建后的 `.mcap` 文件，可在 [Foxglove](https://foxglove.dev/) 等工具中可视化检查。

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 data_convert_mcap.py --config ${CONFIG}
```

输出 `.mcap` 文件默认保存到 `./workspace/`。可使用 `--workspace <path>` 修改输出目录。

**在 Foxglove 中需要关注：**

- **时间戳对齐：** 相机帧和关节状态应同步。较大间隔或抖动说明打包可能有问题。
- **缺失数据流：** 三个相机（左、右、中）都应有连续图像数据。缺少某路数据通常意味着配置中的相机 topic 名与录制不匹配。
- **轨迹连续性：** 关节状态曲线应平滑。突然跳变可能表示存在损坏数据点。

### 3.2 训练数据检查

**目的：** 验证 `.arrow` 数据集能被训练流水线正确加载，并确认所有数据变换（例如 resize、归一化和噪声增强）表现符合预期。

**输出：** 可视化视频（`.mp4`），展示模型作为输入张量实际“看到”的内容。

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 scripts/data_visualize.py --config ${CONFIG}
```

输出视频默认保存到 `./workspace/`。可使用 `--workspace <path>` 修改输出目录。

**检查输出视频时需要关注：**

- **图像正确性：** 相机视角应清晰展示工作空间，而不是黑屏或损坏帧。
- **增强是否合理：** 随机裁剪和噪声应看起来自然；严重畸变通常表示 transform 参数配置错误。
- **动作叠加：** 如果进行了可视化，预测动作应大致跟随示教轨迹。

---

## 4. 模型训练

本步骤使用打包好的 `.arrow` 数据集训练 HoloBrain 模型。训练由两个协同工作的 Python 配置文件控制。

### 4.1 配置结构

| 配置类型 | 作用 | 示例 |
| :--- | :--- | :--- |
| **Dataset Config** | 定义数据位置、相机名称、URDF 和预处理流水线。 | [`config_agilex_ro_dataset.py`](configs/config_agilex_ro_dataset.py) |
| **Training Config** | 定义模型架构、超参数，以及要使用哪些数据集。 | [`config_holobrain_qwen_common.py`](configs/config_holobrain_qwen_common.py) |

训练配置会导入并引用数据集配置，因此通常两个文件都需要编辑。

### 4.2 创建你的数据集配置

以 [`config_agilex_ro_dataset.py`](configs/config_agilex_ro_dataset.py) 作为模板。关键部分是 `dataset_config` 字典。对于 Grasp Anything，其中的值已经填写好：

```python
dataset_config = dict(
    grasp_anything_ro=dict(
        data_paths=[
            "./data/arrow_dataset/place_objects_to_basket/part*",
        ],
        urdf="./urdf/piper_description_dualarm.urdf",
        cam_names=["left", "right", "middle"],
    ),
)
```

你需要根据自己的任务定制三个字段：

- `data_paths`：打包后的 `.arrow` 数据集目录路径。加载器会在每个目录中查找 `state.json`，以发现有效数据集。
- `urdf`：机器人 URDF 文件路径。
- `cam_names`：相机名称，需要与录制时使用的 topic 匹配。

### 4.3 在训练配置中注册数据集

打开训练配置（例如 [`config_holobrain_qwen_common.py`](configs/config_holobrain_qwen_common.py)），将你的数据集标识符添加到 `training_datasets` 和 `deploy_datasets`：

```python
config.update(
    training_datasets=[
        "grasp_anything_ro",  # must match the key in dataset_config
    ],
    deploy_datasets=[
        "grasp_anything_ro",
    ],
)
```

### 4.4 关键超参数

训练配置中包含多个超参数。对初学者最重要的是：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `batch_size` | `16` | 每个训练 step 的样本数。如果 GPU 显存不足，请减小。 |
| `pred_steps` | `64` | 模型在每个时间步预测的未来动作步数。 |
| `lr` | `1e-4` | 学习率。VLM backbone 会自动使用 `lr * 0.1`。 |
| `max_step` | `100000` | 总训练迭代次数。 |
| `save_step_freq` | `5000` | 每 N 步保存一次 checkpoint。 |
| `step_log_freq` | `100` | 每 N 步向控制台打印训练指标（loss、学习率等）。 |
| `num_workers` | `16` | Dataloader worker 数量。如果遇到 CPU/内存限制请减小。对于 Grasp Anything 任务，`4` 通常已经足够。 |

> [!TIP]
> 当 `batch_size=16` 时，训练约需要 16 GB GPU 显存。如果遇到 OOM，请将 `batch_size` 降到 `8` 或 `4`。

### 4.5 预训练 Checkpoint

默认训练配置会加载一个预训练 HoloBrain checkpoint 进行微调：

```python
checkpoint="hf://model/HorizonRobotics/HoloBrain_v0.0_Qwen/pretrain/model.safetensors"
```

首次训练时会自动下载。相比从零训练，从该 checkpoint 开始能显著加快收敛。

### 4.6 启动训练

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py

# Single-GPU training
python3 scripts/train.py --config ${CONFIG}

# Multi-GPU / multi-machine training (example: 2 machines x 8 GPUs)
accelerate launch  \
    --num_machines 2 \
    --num-processes 16  \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7  \
    --machine_rank ${current_rank} \
    --main_process_ip ${main_process_ip} \
    --main_process_port 1227 \
    scripts/train.py \
    --workspace ./workspace \
    --config ${CONFIG}
```

Checkpoint 会保存到 `./workspace` 目录（或 `--workspace` 指定的路径）。

**训练输出：**

- **Checkpoints：** `{workspace}/checkpoints/`（默认 `./workspace/checkpoints/`）。只保留最新的 3 个。每个 checkpoint 目录包含：
  - `model.safetensors`：训练后的模型权重。
  - `model.config.json`：模型架构和推理配置。
- **Processor 配置：** `{workspace}/*_processor.json`。定义推理的数据前处理和后处理流水线（相机外参、图像变换、坐标系转换、机器人运动学等）。
- **TensorBoard 日志：** `{workspace}/logs/`。使用 `tensorboard --logdir ./workspace/logs` 查看。

---

## 5. 部署

训练完成后，需要先将 checkpoint **导出**为部署就绪目录，然后再启动推理。导出步骤会生成推理服务所需的三个产物：

| 产物 | 说明 |
| :--- | :--- |
| `model/model.safetensors` | 训练后的模型权重。 |
| `model/model.config.json` | 模型架构和推理配置。 |
| `*_processor.json` | 数据 processor 配置，定义推理时的前处理和后处理流水线，包括相机外参、图像变换、坐标系转换和机器人运动学。 |

### 5.1 导出命令

```bash
cd projects/holobrain
CONFIG=configs/config_holobrain_qwen_common.py
python3 scripts/export.py --config ${CONFIG} --workspace ./workspace
```

导出的目录（`./workspace`）会包含 `model/` 子目录和上面列出的 processor JSON 文件。

### 5.2 部署到机器人

部署包含：

1. **硬件搭建：** Agilex Piper 双臂 + Intel RealSense 相机，通过 CAN 总线和 USB 连接。
2. **相机标定：** 使用提供的外参，或对自定义配置执行手眼标定。
3. **推理服务：** 启动 HoloBrain 模型服务，通过网络提供预测。
4. **机器人应用：** 将 [ROS2 deploy node](https://github.com/HorizonRobotics/RoboOrchard/tree/master/ros2_package/robo_orchard_deploy_ros2) 连接到推理服务并执行预测动作。

完整流程，包括硬件搭建、手眼标定、CAN/相机 ID 配置、同步/异步推理模式，见 **[真实机器人部署指南](REALBOT_DEPLOY_GUIDE.md)**。

---

## 故障排查 / FAQ

**Q: 我的 arrow 数据集有 0 个 episode。**

数据加载器通过在 `data_paths` 指定目录中查找 `state.json` 来发现 episode。请检查：

1. 你的 `data_paths`（例如 `./data/arrow_dataset/place_objects_to_basket`）是否指向已存在目录。
2. 每个目录中是否包含 `state.json` 文件。

**Q: 可视化显示黑图。**

这通常意味着数据集配置中的相机 topic 名（`cam_names`）与录制时使用的 topic 不匹配。请在 Foxglove 中检查原始 `.mcap` 文件，确认相机名称后更新数据集配置中的 `cam_names`。

**Q: 训练 loss 不下降。**

常见原因：

1. **数据集路径无法解析：** 训练脚本可能静默加载了 0 个 episode。请先运行 `python3 scripts/data_visualize.py` 确认数据能正确加载。
2. **URDF 错误：** 如果 URDF 与录制数据的机器人不匹配，运动学变换会产生无效结果。请确认使用了正确的 URDF 文件。
3. **数据损坏：** 运行数据检查步骤（步骤 3），可视化检查实际训练输入。
