# MolmoAct2

[MolmoAct2](https://allenai.org/blog/molmoact2) 是 Ai2 发布的具身动作推理 VLA 系列，在 Molmo2-ER 视觉语言骨干上接入 **flow-matching 连续 action expert**，支持 closed-loop 机器人控制。

本目录将 MolmoAct2 接入 XPolicyLab，统一 RoboDojo 数据处理、训练与仿真评测流程。上游源码位于 `molmoact2/`（含 FastAPI 推理 server 与 LeRobot 子模块）。

## 与 MolmoAct v1 的区别

| 项目 | MolmoAct v1 | MolmoAct2 |
| --- | --- | --- |
| 数据格式 | Action Reasoning（depth token + trace + 离散 action） | **LeRobot v3.0**（连续 action） |
| 训练框架 | OLMo / `robot-finetune` | **LeRobot** `policy.type=molmoact2` |
| Action | 离散 autoregressive token | **flow-matching 连续 action** |

**v1 的 `process_data.sh` 输出不能直接给 MolmoAct2 使用。** 请从 RoboDojo 原始数据重新转换为 LeRobot v3.0。

## 目录结构

```text
MolmoACT2/
├── INSTALLATION.md          # 环境安装
├── README.md                # 本文件
├── install.sh               # 一键 clone 上游 + 安装 venv（molmoact2/ 不提交 Git）
├── train.sh                 # LeRobot 微调入口
├── eval.sh                  # XPolicyLab 评测编排
├── model.py / deploy.py     # XPolicyLab 推理封装
└── molmoact2/               # 上游 MolmoAct2 源码（install.sh 本地 clone，已 .gitignore）
    ├── examples/            # DROID / YAM FastAPI 推理 server
    └── lerobot/             # LeRobot molmoact2-policy 子模块（训练用）
```

当前已就绪：**官方推理 server**（`molmoact2/examples/`）、**LeRobot 训练**与 **XPolicyLab eval.sh**（`model.py` + `deploy.yml`）。

## 环境安装

见 [`INSTALLATION.md`](./INSTALLATION.md)。

**要点：**
- `molmoact2/` 不在 Git 仓库中，首次使用运行 `bash install.sh`
- **RoboDojo 训练与 XPolicyLab 评测共用 `molmoact2/lerobot/.venv`**
- `molmoact2/.venv` 仅用于上游 FastAPI 官方 server（可选）

## 1. 数据处理

### 目标格式

MolmoAct2 要求 **LeRobot v3.0** 数据集，包含：

- `meta/info.json`（`codebase_version: v3.0`）
- `observation.images.*`、`observation.state`、`action` 等标准字段
- 建议补充 **quantile 统计**（训练默认使用 quantile 归一化）

图像要求与 XPolicyLab 一致：**640×480 RGB**（RoboDojo 观测与 `model.py` 均按 RGB，不做 BGR→RGB），训练与部署保持一致。

### 命名约定（XPolicyLab 5 元组）

处理后数据目录：

```text
policy/MolmoACT2/data/<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

### 调用方式（规划）

`process_data.sh` 遵循 XPolicyLab 统一 5 参数：

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

示例：

```bash
bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint
```

### 转换流程说明

1. 从 `demo_env/data/<bench_name>/<task_name>/<env_cfg>/` 读取 RoboDojo HDF5 轨迹（见 XPolicyLab 根目录 README）。
2. 转为 LeRobot v3.0 布局（可参考 `policy/Abot_M0/abot_m0/data_process/any4lerobot/ds_version_convert/v21_to_v30/`）。
3. 映射观测字段，例如单臂 ARX-X5：
   - `cam_head/color` → `observation.images.image`
   - `cam_wrist/color` → `observation.images.wrist_image`（若有）
   - `arm_joint_state` + `ee_joint_state` → `observation.state` / `action`
4. 生成 quantile stats（LeRobot 工具）：

```bash
cd molmoact2/lerobot
uv run python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id=<your_lerobot_repo_id>
```

> `ckpt_name` 为实验标识，可与源 `task_name` 相同；多任务 cotrain 时可设为 `cotrain` 等，具体读哪些 `task_name` 由 `process_data.sh` 内部决定。

## 2. 训练

训练通过 LeRobot `lerobot_train`，策略类型 `molmoact2`。

### 命名约定（XPolicyLab 6 元组）

```text
policy/MolmoACT2/checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

### 调用方式

`train.sh` 遵循 XPolicyLab 统一 7 参数：

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

**RoboDojo 双臂 co-train**（默认数据集由 `MOLMOACT2_DATASET_ROOT` / `MOLMOACT2_DATASET_REPO_ID` 指定，见 `train.sh`）：

```bash
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0
```

多卡示例：

```bash
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3
```

指定其他 LeRobot 数据集路径：

```bash
MOLMOACT2_DATASET_ROOT=/path/to/your/lerobot/dataset \
MOLMOACT2_DATASET_REPO_ID=your_repo_id \
bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0
```

训练产物：

```text
policy/MolmoACT2/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
```

常用环境变量：`MOLMOACT2_CHECKPOINT_PATH`（默认 `allenai/MolmoAct2`）、`MOLMOACT2_BATCH_SIZE`、`MOLMOACT2_STEPS`、`MOLMOACT2_TRAIN_ACTION_EXPERT_ONLY=0`（全量微调）、`MOLMOACT2_LOCAL_CACHE_ROOT`（多机训练时每台机器独立的 HF datasets 缓存，默认 `/tmp/molmoact2-cache-$(hostname)`）。

多机并发训练同一 NFS 数据集时，`train.sh` 会自动把 `HF_DATASETS_CACHE` 指到本机目录，避免 pyarrow mmap 缓存在共享存储上抢锁（避免 NFS 上 parquet mmap 抢锁）。首次启动会在本机重建 parquet cache，需预留约等于 parquet 体积的本地磁盘空间。

### 直接调用 LeRobot（高级）

在 submodule 初始化并完成 `INSTALLATION.md` 第 3 步（`lerobot/.venv`）后，可直接使用：

```bash
cd molmoact2/lerobot
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

accelerate launch \
  --num_processes=1 \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=<lerobot_repo_id> \
  --dataset.root=/path/to/lerobot/data/<lerobot_repo_id> \
  --dataset.video_backend=pyav \
  --policy.type=molmoact2 \
  --policy.checkpoint_path=allenai/MolmoAct2 \
  --policy.device=cuda \
  --policy.action_mode=continuous \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.setup_type="single arx x5 robotic arm in robodojo" \
  --policy.control_mode="absolute joint pose" \
  --policy.image_keys='["observation.images.image","observation.images.wrist_image"]' \
  --policy.model_dtype=bfloat16 \
  --policy.num_flow_timesteps=8 \
  --policy.gradient_checkpointing=true \
  --policy.train_action_expert_only=true \
  --output_dir=../checkpoints/robodojo_run \
  --steps=10000 \
  --batch_size=8
```

常用策略选项：

| 参数 | 说明 |
| --- | --- |
| `policy.checkpoint_path` | 官方 HF 权重（如 `allenai/MolmoAct2`） |
| `policy.path` | LeRobot 已保存的 checkpoint 目录 |
| `policy.action_mode` | `continuous` / `discrete` / `both` |
| `policy.train_action_expert_only` | 小数据集建议先只训 action expert |
| `policy.enable_lora_vlm` | 对 VLM 开 LoRA，action expert 仍全量训练 |
| `policy.setup_type` / `policy.control_mode` | 写入 prompt 的本体与控制模式描述 |
| `policy.image_keys` | 与 LeRobot 数据集图像字段一致，**顺序敏感** |

小数据集（<200 demos）建议：`batch_size=16~32`，`train_action_expert_only=true` 或 `enable_lora_vlm=true`。

完整参数说明：[LeRobot MolmoAct2 文档](https://github.com/allenai/lerobot/blob/molmoact2-policy/docs/source/molmoact2.mdx)。

## 3. 评测与部署

XPolicyLab 评测采用 **policy server + env client** 分离架构（见根目录 README §4）。

### 3.1 官方 FastAPI Server（当前可用）

适用于 DROID / YAM 等官方预微调 checkpoint：

```bash
cd molmoact2
uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8000 --dtype bfloat16
```

| Server | Checkpoint | 默认端口 | State | 相机 |
| --- | --- | --- | --- | --- |
| `examples/droid/host_server_droid.py` | `allenai/MolmoAct2-DROID` | 8000 | 8-D joint+gripper | external, wrist |
| `examples/yam/host_server_yam.py` | `allenai/MolmoAct2-BimanualYAM` | 8202 | 14-D 双臂 | top, left, right |

请求/响应为 `json_numpy` 编码，详见 `molmoact2/CLAUDE.md`。

### 3.2 XPolicyLab 统一评测

`eval.sh` 遵循 XPolicyLab 统一 11 参数。policy 侧使用 `molmoact2/lerobot/.venv`（与训练相同），在 `deploy.yml` 中通过 `policy_uv_env_path: molmoact2/lerobot` 指定；`eval.sh` 第 10 参数传 `uv` 即可。

```bash
bash eval.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> \
  <expert_data_num> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

示例（debug 模式，双臂 cotrain checkpoint）：

```bash
cd policy/MolmoACT2
bash eval.sh RoboDojo debug_task cotrain arx_x5 3500 joint 0 0 0 uv XPolicyLab
```

`deploy.yml` 中 `eval_env` 设为 `debug` 可离线调试观测/动作格式；通过后改为 `sim` 跑 RoboDojo 仿真。

### RoboDojo 观测映射（接入 `model.py` 时需实现）

XPolicyLab 观测 → MolmoAct2 输入的大致对应关系（单臂 ARX-X5）：

| XPolicyLab | MolmoAct2 / LeRobot |
| --- | --- |
| `vision/cam_head/color` (RGB) | `observation.images.image` (RGB, 640×480) |
| `vision/cam_wrist/color` | `observation.images.wrist_image` |
| `state/arm_joint_state` + `state/ee_joint_state` | `observation.state` |
| `instruction` | prompt 中的 task 文本 |

`model.py` 需实现 `ModelTemplate` 的 `update_obs` / `get_action` / `reset` 等接口；推理侧图像为 **RGB 直通**（`decode_image_bit`，无通道反转），并注意 **action 归一化/反归一化**（使用 checkpoint 内 `norm_stats.json` 或数据集 quantile stats）。

## 4. 参数速查

### 训练 / 数据处理

| 参数 | 训练 | 评测 | 含义 |
| --- | --- | --- | --- |
| `bench_name` | ✓ | ✓ | 数据集名称，如 `RoboDojo` |
| `ckpt_name` | ✓ | ✓ | 实验标识；训练时决定 data/checkpoint 子目录 |
| `task_name` | — | ✓ | 仿真任务名，传给环境客户端 |
| `env_cfg_type` | ✓ | ✓ | 本体配置，如 `arx_x5` |
| `expert_data_num` | ✓ | ✓ | 训练轨迹数量 |
| `action_type` | ✓ | ✓ | `joint` / `ee` 等 |
| `seed` | ✓ | ✓ | 随机种子 |

### 推荐 checkpoint

| 场景 | 起点 checkpoint |
| --- | --- |
| RoboDojo 新本体微调 | `allenai/MolmoAct2` |
| LIBERO 仿真 | `allenai/MolmoAct2-LIBERO` |
| Franka DROID 风格 | `allenai/MolmoAct2-DROID` |
| 双臂 YAM | `allenai/MolmoAct2-BimanualYAM` |

## 5. 参考链接

- 上游 README：`molmoact2/README.md`
- 开发说明：`molmoact2/CLAUDE.md`
- LeRobot 训练文档：[molmoact2.mdx](https://github.com/allenai/lerobot/blob/molmoact2-policy/docs/source/molmoact2.mdx)
- 论文：[arXiv:2605.02881](https://arxiv.org/abs/2605.02881)
- 官方模型：[HF MolmoAct2 Models](https://huggingface.co/collections/allenai/molmoact2-models-69f81e05242e2499606b1be6)
- 官方数据集：[HF MolmoAct2 Datasets](https://huggingface.co/collections/allenai/molmoact2-datasets-69f81e316ec3daafe3f9555c)
