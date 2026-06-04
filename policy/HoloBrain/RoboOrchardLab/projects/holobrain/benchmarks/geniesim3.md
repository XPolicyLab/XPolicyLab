# GenieSim 3.0 Benchmark 评测指南

本指南介绍如何使用 HoloBrain policy 运行 [GenieSim 3.0](https://agibot-world.com/challenge2026/reasoning2action) benchmark，流程包括数据准备、微调、模型导出，以及启动用于闭环评测的 WebSocket 推理服务。

> [!NOTE]
> 除非特别说明，下方所有命令都假设你位于**仓库根目录**（`/path/to/robo_orchard_lab`）。

---

## 前置条件

| 项目 | 说明 |
|------|------------|
| **HoloBrain 环境** | 按照主 README 中的 [安装](../README.md#1-安装) 完成环境配置。 |
| **GenieSim 3.0 数据集** | 从官方 [Reasoning2Action Quick Start](https://agibot-world.com/challenge2026/reasoning2action/quick-start) 的 `01 Dataset` 部分下载微调数据集。解压后的预期目录结构见 Hugging Face 的 [Dataset Structure](https://huggingface.co/datasets/agibot-world/AgiBotWorldChallenge-2026#dataset-structure)。 |
| **G2 omnipicker URDF** | 从 GenieSim 仓库下载：[`G2_omnipicker.urdf`](https://github.com/AgibotTech/genie_sim/blob/v3.0.3/source/geniesim/app/robot_cfg/G2_omnipicker/G2_omnipicker.urdf)。将其放置或软链接到 `projects/holobrain/urdf/G2_omnipicker.urdf`。 |
| **GenieSim 3.0 仿真服务** | 闭环评测（步骤 5）需要该服务。安装和 Docker 镜像说明见官方 [仿真评测指南](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_352-run-icra-tasks)。 |

---

## 1. 准备数据

将原始 GenieSim 3.0 微调数据打包为 RoboOrchard Arrow 分片。请确认 `PATH` 中可用 `ffmpeg` 和 `ffprobe`。

### 快速开始（单任务、单分片）

```bash
python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py \
    --dataset_name AgiBotWorldChallenge-2026 \
    --input_dir /path/to/GenieSim3.0-Dataset/Reasoning2Action-Sim/dataset_without_depth \
    --urdf_path /path/to/G2_omnipicker.urdf \
    --robot_name G2_omnipicker \
    --task_name hold_pot \
    --output_dir projects/holobrain/data/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/hold_pot \
    --writer_batch_size 500 \
    --num_jobs 8 \
    --job_idx 0 \
    --force_overwrite
```

对每个任务，需要运行 `[0, num_jobs)` 范围内的所有 `job_idx`（例如 `hold_pot` 的 `num_jobs=8`，需要运行 8 次，即 `--job_idx 0` 到 `--job_idx 7`）。每个 job 会写入一个零填充编号的分片目录，例如 `hold_pot/part-00000-of-00008`。

<details>
<summary><b>任务参考（无深度 split）- 点击展开</b></summary>

`num_jobs` 是建议将每个任务数据拆分成的分片数量，用于并行打包。较大的任务 episode 更多，也更适合使用更多分片。对每个任务，都需要以 `--job_idx` 从 `0` 到 `num_jobs - 1` 各运行一次打包器。

| 任务名 | 推荐 `num_jobs` |
|-----------|----------:|
| `clean_the_desktop_addition` | 4 |
| `clean_the_desktop_part_1` | 2 |
| `clean_the_desktop_part_2` | 2 |
| `hold_pot` | 8 |
| `open_door` | 8 |
| `place_block_into_box` | 11 |
| `pour_workpiece` | 8 |
| `scoop_popcorn` | 4 |
| `scoop_popcorn_part_2` | 4 |
| `sorting_packages_part_1` | 3 |
| `sorting_packages_part_2` | 3 |
| `sorting_packages_part_3` | 3 |
| `stock_and_straighten_shelf` | 4 |
| `stock_and_straighten_shelf_part_2` | 4 |
| `take_wrong_item_shelf` | 9 |

</details>

打包器会自动检测深度视频；当输入为 `dataset_without_depth` 时，只会写入 RGB 特征。运行 `python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py --help` 可查看完整选项列表。

> **自定义输出路径？** 训练前请相应更新 `configs/config_agibot_geniesim_dataset.py` 中的 `data_paths`。
>
> **夹爪单位：** GenieSim 3.0 的原始观测中，夹爪值使用执行器原始范围，而夹爪动作已经归一化。数据集加载器会将观测中的夹爪值除以 `gripper_divisor`，但保持动作夹爪值不变，这样采样后 `joint_state` 与 `master_joint_state` 会保持在同一归一化训练范围内。

### 验证打包数据

打包完成后，可视化结果以粗略检查图像、关节状态和动作标签：

```bash
cd projects/holobrain

python3 scripts/data_visualize.py \
    --config configs/config_holobrain_qwen_common.py \
    --dataset_names agibot_geniesim3_challenge \
    --kwargs '{"interval": 3, "ee_indices": [7,15], "fps": 30}'
```

---

## 2. 训练

首先确认 `configs/config_holobrain_qwen_common.py` 使用 GenieSim 3.0 数据集。在配置覆盖的末尾添加或确认以下 `config.update` 代码块：

```python
config.update(
    training_datasets=[
        "agibot_geniesim3_challenge",
    ],
    deploy_datasets=["agibot_geniesim3_challenge"],
)
```

这会告诉训练流水线，通过 `configs/config_agibot_geniesim_dataset.py` 中定义的 `agibot_geniesim3_challenge` 条目，加载步骤 1 打包出的 Arrow 分片。

同时确认同一配置文件中的预训练 checkpoint 路径（`vlm_pretrain` 和 `checkpoint`）指向有效的本地路径。

### 单 GPU

```bash
cd projects/holobrain

python3 scripts/train.py \
    --config configs/config_holobrain_qwen_common.py
```

### 多 GPU / 多机器（示例：2 台机器 x 8 张 GPU）

```bash
cd projects/holobrain

accelerate launch \
    --num_machines 2 \
    --num-processes 16 \
    --multi-gpu \
    --gpu-ids 0,1,2,3,4,5,6,7 \
    --machine_rank ${CURRENT_RANK} \
    --main_process_ip ${MAIN_PROCESS_IP} \
    --main_process_port 1227 \
    scripts/train.py \
    --workspace ./workspace \
    --config configs/config_holobrain_qwen_common.py
```

更多训练选项（`--workspace`、`--eval_only`、`--kwargs` 等）见 [主 README - 运行训练](../README.md#3-运行训练)。

---

## 3. 导出模型

导出会将训练好的 checkpoint、processor 配置和 pipeline 定义打包成一个自包含产物，可直接用于推理。

```bash
cd projects/holobrain

python3 scripts/export.py \
    --config configs/config_holobrain_qwen_common.py \
    --workspace ./model_export_path
```

导出的 `./model_export_path` 目录会在下一步作为 `MODEL_DIR` 使用。

---

## 4. 启动推理服务

启动 GenieSim 3.0 WebSocket policy 服务。该服务会等待 GenieSim 仿真客户端连接。

```bash
cd projects/holobrain

python3 scripts/geniesim3_inference_server.py \
    --model_dir ./model_export_path \
    --inference_prefix agibot_geniesim3_challenge \
    --host 0.0.0.0 \
    --port 8999
```

启动时，服务会打印一个或多个 `ws://` URL；请将合适的 URL 作为 `--infer-host` 传给 GenieSim 3.0 benchmark 客户端。

<details>
<summary><b>服务选项 - 点击展开</b></summary>

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--model_dir` | `./model` | 导出模型目录（来自步骤 3）。 |
| `--inference_prefix` | `agibot_geniesim3_challenge` | 已保存推理 pipeline 配置文件的前缀。 |
| `--model_prefix` | `model` | `model_dir` 中导出模型文件的前缀。 |
| `--load_weights` | `true` | 是否加载模型权重。 |
| `--load_impl` | `native` | 模型加载后端（`native` 或 `accelerate`）。 |
| `--host` | `0.0.0.0` | WebSocket 绑定地址。 |
| `--port` | `8999` | WebSocket 端口。 |
| `--valid_action_step` | `32` | 每次推理调用发送给仿真的动作步数。 |
| `--sampling_ratio` | `1.0` | 模型输出截断前的重采样比例。`1` 保留原始值；大于 1 的整数使用 stride sampling；其他正数使用线性插值。 |
| `--gripper_limit` | `1.0` | 原始 payload 夹爪观测除以 `120.0` 后应用的缩放。保持 `1.0` 可对应归一化训练尺度。 |
| `--use_depth` | `false` | 仅当导出模型和 benchmark payload 都包含深度数据时才设置为 `true`。 |

</details>

部署 policy 会使用配置中的任务名指令映射，作为支持的 GenieSim 任务的标准 prompt 来源。只有当传入 payload 的 `task_name` 没有已配置的默认指令时，才会使用 payload 中的 `prompt`。

---

## 5. 运行闭环评测

推理服务（步骤 4）运行后，启动 GenieSim 3.0 仿真环境，并将其连接到服务的 WebSocket URL。仿真客户端会驱动闭环流程：每一步向推理服务发送观测，并接收动作块。

关于仿真评测环境搭建和运行的详细说明，请参考官方文档：

- **[GenieSim 3.0 仿真评测指南 - 运行 ICRA 任务](https://agibot-world.com/sim-evaluation/docs/#/v3?id=_352-run-icra-tasks)**：涵盖环境搭建、Docker 镜像、任务配置，以及如何将 benchmark 客户端指向你的推理服务（`--infer-host ws://<your_ip>:8999`）。

> [!TIP]
> 推理服务（步骤 4）启动时会打印可用的 `ws://` URL。启动 GenieSim benchmark 客户端时，请使用其中一个 URL 作为 `--infer-host` 参数。

---

## 快速参考

最小端到端运行流程（假设前置条件已准备好）：

```bash
# 0. Pack data (from repo root, repeat for each task and each job_idx in [0, num_jobs))
#    e.g. for hold_pot with num_jobs=8, run 8 times with --job_idx 0..7
python3 robo_orchard_lab/dataset/agibot_geniesim/packer/arrow_pack_geniesim3.py \
    --dataset_name AgiBotWorldChallenge-2026 \
    --input_dir /path/to/dataset_without_depth \
    --urdf_path /path/to/G2_omnipicker.urdf \
    --task_name hold_pot \
    --output_dir projects/holobrain/data/arrow_dataset/AgiBotWorldChallenge-2026/Reasoning2Action-Sim/hold_pot \
    --num_jobs 8 --job_idx ${JOB_IDX} --force_overwrite

# 1. Train (single GPU)
cd projects/holobrain
python3 scripts/train.py --config configs/config_holobrain_qwen_common.py

# 2. Export
python3 scripts/export.py \
    --config configs/config_holobrain_qwen_common.py \
    --workspace ./model_export_path

# 3. Serve
python3 scripts/geniesim3_inference_server.py \
    --model_dir ./model_export_path \
    --inference_prefix agibot_geniesim3_challenge \
    --port 8999
```
