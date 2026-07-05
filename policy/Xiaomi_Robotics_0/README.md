# Xiaomi_Robotics_0

Xiaomi-Robotics-0（XR-0）是小米开源的 4.7B 参数 Vision-Language-Action 模型。本目录是其在 XPolicyLab 下的接入层：将 RoboDojo 原始 HDF5 轨迹转换为 XR-0 JSON + 三视角视频格式，再启动 Hydra + DeepSpeed 微调。

上游训练代码位于：

```text
policy/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0
```

环境安装见 [INSTALLATION.md](INSTALLATION.md)。

## 整体流程

```text
RoboDojo HDF5 原始数据
        │  process_data.sh
        ▼
XR-0 JSON + mp4 + action_stats.json
        │  train.sh
        ▼
checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
```

## 数据格式

`process_data.sh` 调用 `scripts/transform_xr0_json_format.py`，输出目录结构如下：

```text
data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/
├── json/
│   ├── episode_000000.json
│   └── ...
├── videos/
│   ├── episode_000000_ego.mp4
│   ├── episode_000000_wrist_left.mp4
│   ├── episode_000000_wrist_right.mp4
│   └── ...
└── action_stats.json
```

每条轨迹包含三视角图像（head / 左腕 / 右腕）、双臂 proprio/action，以及 XR-0 使用的 32 维相对动作（action chunk 长度 30）。详细字段说明见 `xiaomi_robotics_0/xr0/docs/data_format.md`。

## 数据处理

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

`expert_data_num` 为可选尾参，留空表示使用全部 episode。数据量 ablation 请使用不同 `ckpt_name`（如 `sweep_blocks_50ep`）搭配相应 `expert_data_num`。

### 单任务示例

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Xiaomi_Robotics_0

# 使用全部 episode
bash process_data.sh RoboDojo sweep_blocks arx_x5 ee
# 只用前 50 条 episode
bash process_data.sh RoboDojo sweep_blocks arx_x5 ee 50
```

当 `bench_name=RoboDojo` 时，需设置 `XR0_RAW_DATA_ROOT` 指向 RoboDojo HDF5 根目录：

```bash
export XR0_RAW_DATA_ROOT=/path/to/RoboDojo
# 单任务: ${XR0_RAW_DATA_ROOT}/sim_cloud/<ckpt_name>/arx_x5/
```

### Co-train（35 任务联合训练）

`sim_cloud` 下共 35 个任务，每个任务 100 条 episode。使用 `ckpt_name=cotrain`；如设置可选尾参 `expert_data_num`，每个任务最多转换该数量：

```bash
bash process_data.sh RoboDojo cotrain arx_x5 ee
```

输出目录：

```text
data/RoboDojo-cotrain-arx_x5-ee/
```

`process_data.sh` 还会根据 `action_stats.json` 自动生成 Hydra 数据配置：

```text
xiaomi_robotics_0/xr0/configs/data/RoboDojo-cotrain-arx_x5-ee.yaml
```

## 训练

先运行 `process_data.sh`，再运行：

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

### 单任务示例

```bash
bash train.sh RoboDojo sweep_blocks arx_x5 ee 0 0
```

### Co-train 多卡示例

```bash
bash train.sh RoboDojo cotrain arx_x5 ee 0 0,1,2,3,4,5,6,7
```

训练产物保存在（XR-0 实际写入 `train_runs/<同名>/project_xr0/<同名>/`，`train.sh` 会自动把标准路径软链过去）：

```text
checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
```

例如：

```text
checkpoints/RoboDojo-cotrain-arx_x5-ee-0/
```

该目录名即 eval 时传入的 `ckpt_name`。

## 常用环境变量

| 变量 | 含义 |
|---|---|
| `XR0_RAW_DATA_ROOT` | RoboDojo 原始数据根目录 |
| `XR0_CONVERTED_DATA_ROOT` | 转换后数据输出目录 |
| `XR0_PRETRAINED_PATH` | 预训练 `.pt` 路径 |
| `XR0_DATA_CONFIG_NAME` | 生成的 Hydra data config 名称 |
| `XR0_CONVERT_WORKERS` | 数据转换并行进程数（默认 8） |
| `XR0_MAX_STEPS` | 训练步数（默认 30000） |
| `XR0_SAVE_INTERVAL` | checkpoint 保存间隔（默认 5000） |
| `XR0_ASYNC_TRAIN` | 是否启用异步训练（默认 `false`） |
| `RESOURCE_GPU` | 每节点 GPU 数，默认由 `gpu_id` 推断 |

## 评测（Debug / Sim / Real）

通过 `EVAL_ENV_TYPE` 控制评测环境：未设置或 `sim` 跑仿真，`debug` 离线校验；`real` 在开源版不可用。无需修改 `eval.sh`。

### 1. 链接训练权重

推荐使用 symlink，避免复制大体积 checkpoint：

```bash
cd policy/Xiaomi_Robotics_0

bash scripts/link_checkpoint.sh RoboDojo-cotrain-arx_x5-ee-0 /path/to/finetuned_ckpt
```

链接名即 eval 时传入的 `ckpt_name`。本地 `train.sh` 训练完成后无需再手动链接（脚本已自动创建标准路径软链）。

也可手动（`source` 为含 `config.py` 与 `last.ckpt/` 的目录）：

```bash
mkdir -p checkpoints
ln -sfn /path/to/finetuned_ckpt checkpoints/RoboDojo-cotrain-arx_x5-ee-0
```

### 2. 启动评测

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

`ckpt_name` 即 `checkpoints/` 下的完整 run 目录名。Debug 本地测试（policy server 与 env client 均使用 `mibot`，0 号卡）：

```bash
bash eval.sh RoboDojo sweep_blocks RoboDojo-cotrain-arx_x5-ee-0 arx_x5 ee 0 0 0 mibot mibot
```

`deploy.yml` 可配置项：

| 字段 | 含义 |
|---|---|
| `EVAL_ENV_TYPE` | unset/`sim` / `debug` / `real` (open-source blocks `real`) |
| `eval_batch` | 是否走 batch 推理 |
| `checkpoint_tag` | 加载 `last.ckpt` 或 `epoch=0-step=30000.ckpt` 等 |
| `model_dir` | 相对 policy 根目录的 checkpoint 路径；`null` 时用 `checkpoints/<ckpt_name>/` |
| `action_length` | 动作 chunk 长度（默认 30） |
| `vlm_processor_path` | HuggingFace 仓库 id（默认 `XiaomiRobotics/Xiaomi-Robotics-0-Pretrain`，启动时自动下载）；离线可改为 `xr0/` 下相对路径 |
| `default_prompt` | 观测无 instruction 时的默认语言指令 |

部署时 checkpoint 路径相对 policy 根目录解析；VLM processor 默认从 HuggingFace 下载，无需本地 `hf_pretrain` 软链。

## 与 XPolicyLab 的参数约定

| 参数 | 训练侧含义 |
|---|---|
| `bench_name` | 数据集名称，如 `RoboDojo` |
| `ckpt_name` | 实验标识；单任务可与源 `task_name` 相同，`cotrain` 表示 35 任务联合 |
| `env_cfg_type` | 本体配置，RoboDojo 双臂为 `arx_x5` |
| `expert_data_num` | `process_data.sh` 的可选尾参；留空 = 全部 episode，设置时每任务最多取该数量。train.sh 不接收该参数 |
| `action_type` | XPolicyLab 命名参数；XR-0 转换固定使用相对 EE + joint 组成 32 维动作 |
| `seed` | 随机种子，参与 checkpoint 命名 |
| `gpu_id` | 传给 `CUDA_VISIBLE_DEVICES` |

命名约定：

- **处理后数据**（4 元组）：`data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`
- **训练产物**（4 元组 + seed）：`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`

数据量 ablation：使用不同 `ckpt_name`（如 `cotrain_50ep`）+ 可选 `expert_data_num`。

## 项目结构

```text
Xiaomi_Robotics_0/
├── INSTALLATION.md
├── README.md
├── install.sh
├── process_data.sh
├── train.sh
├── eval.sh
├── deploy.yml
├── deploy.py
├── model.py
├── setup_eval_policy_server.sh
├── setup_eval_env_client.sh
├── scripts/
│   ├── generate_data_config.py
│   └── link_checkpoint.sh
├── checkpoints/
└── xiaomi_robotics_0/
    └── xr0/                     # 上游 XR-0 训练代码
        ├── configs/
        ├── mibot/
        ├── scripts/train.sh
        └── tools/
```

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-100-ee-0 arx_x5 ee 0 <policy_gpu> <env_gpu> mibot XPolicyLab
```

## 参考文档

- [XPolicyLab README](../../README.md)：平台总览、数据格式与评测约定
- [xr0/README.md](xiaomi_robotics_0/xr0/README.md)：XR-0 架构、异步训练与部署
- [xr0/docs/data_format.md](xiaomi_robotics_0/xr0/docs/data_format.md)：JSON 标注字段说明
