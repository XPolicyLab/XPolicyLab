# LDA_1B on XPolicyLab

LDA-1B（QwenMMDiT + DINOv3）在 XPolicyLab 上的适配。产物命名遵循 [XPolicyLab README §4.2](../../README.md)：

| 产物 | 命名 | 默认路径 |
|---|---|---|
| 处理后数据集 | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>` | `policy/LDA_1B/data/` |
| 训练 checkpoint | `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>` | `policy/LDA_1B/checkpoints/` |

---

## 1. 安装

环境与权重下载见 [INSTALLATION.md](INSTALLATION.md)。

---

## 2. 数据转换

HDF5 → LeRobot v2.1；相机 RGB `(240,320,3)`；state/action 为双臂 joint + gripper。

### 2.1 单任务

```bash
conda activate LDA_1B
bash process_data.sh RoboDojo test_data arx_x5 joint
# 可选尾参 expert_data_num 限制每 task episode 数
bash process_data.sh RoboDojo test_data arx_x5 joint 100
# 输出: data/RoboDojo-test_data-arx_x5-joint/
```

### 2.2 多任务 cotrain

```bash
bash process_data_batch.sh RoboDojo cotrain arx_x5 joint
# 输出: data/RoboDojo-cotrain-arx_x5-joint/  (dataset_id=cotrain 时)
```

---

## 3. 训练

```bash
# bench_name ckpt_name env_cfg_type action_type seed gpu_id
bash train.sh RoboDojo test_data arx_x5 joint 0 0
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
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
| `LDA_CHECKPOINT_PATH` | — | eval 时覆盖 checkpoint `.pt` 路径 |

训练产物：`<policy>/checkpoints/<bench>-<ckpt>-<env>-<action>-<seed>/checkpoints/steps_*_pytorch_model.pt`

---

## 4. 部署与评测

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 0 LDA_1B XPolicyLab
# dataset task ckpt env action seed policy_gpu env_gpu policy_conda eval_conda
```

Use the `EVAL_ENV_TYPE` environment variable: `debug` → `sim` → `real`。

Checkpoint 由 `setup_eval_policy_server.sh` 按 4+seed 元组 `ckpt_run_id` 精确解析；可用 `LDA_CHECKPOINT_PATH` 覆盖。

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
