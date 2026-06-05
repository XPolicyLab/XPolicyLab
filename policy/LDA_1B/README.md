# LDA_1B on XPolicyLab

LDA-1B（QwenMMDiT + DINOv3）在 XPolicyLab 上的适配。产物命名遵循 [XPolicyLab README §4.2](../../README.md)：

| 产物 | 命名 | 默认路径 |
|---|---|---|
| 处理后数据集 | `<dataset_name>-<ckpt_name>-<env_cfg_type>-<action_type>` | `policy/LDA_1B/data/` |
| 训练 checkpoint | `<dataset_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>` | `policy/LDA_1B/checkpoints/` |

旧版含 `expert_data_num` 的目录名（如 `RoboDojo-test_data-arx_x5-3-joint`）及 `cotrain_dataset` 仍可通过 `_artifact_paths.sh` 自动回退解析。

---

## 1. 安装

环境与权重下载见 [INSTALLATION.md](INSTALLATION.md)。

---

## 2. 数据转换

HDF5 → LeRobot v2.1；相机 RGB `(240,320,3)`；state/action 为双臂 joint + gripper。

### 2.1 单任务

```bash
conda activate LDA_1B
bash process_data.sh RoboDojo test_data arx_x5 100 joint
# 输出: data/RoboDojo-test_data-arx_x5-joint/
```

### 2.2 多任务 cotrain

```bash
bash process_data_batch.sh RoboDojo cotrain arx_x5 100 joint
# 输出: data/cotrain/
```

---

## 3. 训练

```bash
# dataset_name ckpt_name env_cfg_type expert_data_num action_type seed gpu_id
bash train.sh RoboDojo test_data arx_x5 100 joint 0 0
bash train.sh RoboDojo cotrain arx_x5 100 joint 0 0
```

- mixture：`xpolicylab`（`XPOLICYLAB_DATASET_ID` 由 `train.sh` 注入）
- 默认全量微调：`freeze_modules: ''`，`tune_vision_encoder: true`
- 建议从 `checkpoints/LDA-pretrain/LDA-pretrain.pt` 初始化

常用环境变量：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `LDA_DATA_ROOT` | `<policy>/data` | LeRobot 数据根 |
| `LDA_CKPT_ROOT` | `<policy>/checkpoints` | 训练输出根 |
| `LDA_PRETRAINED_CHECKPOINT` | `<policy>/checkpoints/LDA-pretrain/LDA-pretrain.pt` | 预训练起点 |
| `LDA_EXPERT_DATA_NUM` | — | eval 时解析旧版含 `expert_data_num` 的 ckpt 目录 |

训练产物：`<policy>/checkpoints/<dataset>-<ckpt_name>-<env>-<action>-<seed>/checkpoints/steps_*_pytorch_model.pt`

---

## 4. 部署与评测

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 LDA_1B XPolicyLab
# dataset task ckpt env action seed policy_gpu env_gpu policy_conda eval_conda
```

`deploy.yml` 中 `eval_env`：`debug` → `sim` → `real`。

旧 checkpoint 布局可通过环境变量回退：

```bash
export LDA_EXPERT_DATA_NUM=100          # 旧 6 元组 ckpt 目录
export LDA_CHECKPOINT_PATH=.../steps_*_pytorch_model.pt
```

---

## 5. 策略包结构

| 文件 | 用途 |
|---|---|
| `model.py` | 推理适配（RGB 224 letterbox、obs_horizon 缓冲、q99 反归一化） |
| `LDA-1B/xpolicylab_adapter/` | HDF5→LeRobot、产物路径、action dim |
| `LDA-1B/lda/config/training/xpolicylab_arx_x5_LDA.yaml` | arx_x5 训练配置 |
| `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` | 评测 server/client 拆分 |

目录结构：

```
XPolicyLab/policy/LDA_1B/
├── model.py / deploy.py / deploy.yml
├── install.sh / INSTALLATION.md / README.md
├── process_data.sh / process_data_batch.sh / train.sh / eval.sh
├── setup_eval_policy_server.sh / setup_eval_env_client.sh
├── data/          # process_data 产物
├── checkpoints/   # 预训练权重 + train.sh 输出
└── LDA-1B/        # 上游源码与 xpolicylab_adapter
```
