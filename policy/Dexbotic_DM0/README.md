# Dexbotic_DM0

Dexbotic DM0 在 XPolicyLab 下的接入目录。DM0 使用 **Dexdata** 格式（jsonl + video），Flow Matching 一次生成 50 步动作 chunk。

上游源码位于 `dexbotic/`。安装步骤见 `INSTALLATION.md`。

## 项目结构

```text
Dexbotic_DM0
├── INSTALLATION.md
├── install.sh
├── process_data.sh              # RoboDojo HDF5 -> Dexdata
├── train.sh                     # torchrun 训练入口
├── README.md
├── scripts/
│   ├── transform_dm0_dexdata_format.py
│   └── generate_data_source.py
├── data/                        # 转换后的 Dexdata（gitignore）
├── checkpoints/                 # 训练输出（gitignore）
└── dexbotic/                    # Dexbotic 上游源码
    ├── dexbotic/
    │   └── data/data_source/    # process_data 自动生成 robodojo_*.py
    ├── playground/benchmarks/robodojo/robodojo_dm0.py
    └── checkpoints/DM0-base/    # 预训练权重（需下载）
```

## 数据处理

```bash
cd policy/Dexbotic_DM0
bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

示例（35 任务 co-train，共 3500 条）：

```bash
export DM0_CONVERT_WORKERS=16   # 并行 worker 数，默认 8
bash process_data.sh RoboDojo cotrain arx_x5 3500 ee
```

示例（单任务）：

```bash
bash process_data.sh RoboDojo sweep_blocks arx_x5 100 ee
```

输出目录：

```text
data/RoboDojo-cotrain-arx_x5-3500-ee/
├── episode_000000.jsonl
├── video/
│   ├── episode_000000_head.mp4
│   ├── episode_000000_left_wrist.mp4
│   └── episode_000000_right_wrist.mp4
└── index_cache.json            # 首次训练时 Dexbotic 自动生成
```

同时会在 `dexbotic/dexbotic/data/data_source/` 下生成 `robodojo_<5-tuple>.py`。

## 训练

```bash
cd policy/Dexbotic_DM0
conda activate DM0
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例（8 卡，global batch = 256）：

```bash
bash train.sh RoboDojo cotrain arx_x5 3500 ee 0 0,1,2,3,4,5,6,7
```

### Batch 配置

`train.sh` 以 `DM0_GLOBAL_BATCH_SIZE` 为唯一目标，自动推导梯度累积：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DM0_GLOBAL_BATCH_SIZE` | `256` | 全局 batch |
| `DM0_BATCH_SIZE` | `4` | 每卡 micro batch |
| `DM0_GRAD_ACCUM` | 自动 | `global / (batch × num_gpus)` |

8 卡默认：`4 × 8 × 8 = 256`。启动时会打印 `per_device_batch`、`grad_accum`、`global_batch`。

其他常用变量：`DM0_MAX_STEPS`（默认 60000）、`DM0_SAVE_STEPS`（默认 2000）、`DM0_BASE_MODEL`。

Checkpoint 输出：

```text
checkpoints/RoboDojo-cotrain-arx_x5-3500-ee-0/
```

## Dexdata 字段说明

| 字段 | 说明 |
|------|------|
| `images_1/2/3` | head / left_wrist / right_wrist 三视角 video |
| `state` | 32 维双臂 proprio |
| `prompt` | 语言指令 |
| `is_robot` | 必须为 `true` |

训练时 Dexbotic 在线执行 `AddAction` → `PadState/PadAction(32D)` → `AddTrajectory(50)` → `DeltaAction`，rank0 自动计算 norm stats。`non_delta_mask=[6, 20]` 对应左右 gripper。

## 完整流程

1. 按 `INSTALLATION.md` 创建 `DM0` 环境并 `bash install.sh`
2. 下载权重：`hf download Dexmal/DM0-base --local-dir dexbotic/checkpoints/DM0-base`
3. `bash process_data.sh ...`
4. `bash train.sh ...`

## 推理（可选）

```bash
export DM0_OUTPUT_DIR=checkpoints/<run_name>
cd dexbotic
CUDA_VISIBLE_DEVICES=0 python playground/benchmarks/robodojo/robodojo_dm0.py --task inference
```

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-ee-0 arx_x5 3500 ee 0 <policy_gpu> <env_gpu> DM0 XPolicyLab
```
