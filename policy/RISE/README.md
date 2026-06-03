# RISE on XPolicyLab

[RISE](https://opendrivelab.com/rise/)（*Self-Improving Robot Policy with Compositional World Model*）在 XPolicyLab 中的集成，**仅包装离线流程**：HDF5/LeRobot 数据准备、value 打标、advantage-conditioned policy 训练，以及经 XPolicyLab policy server 的仿真/真机评测。

上游完整三件套（dynamics model、online RL、Piper 真机部署）仍保留在 vendored 源码树中供查阅，但 **XPolicyLab 不提供对应入口脚本**；详见 [§7 与上游 RISE 的关系](#7-与上游-rise-的关系)。上游总览见 [`RISE/README.md`](RISE/README.md) 与 [`RISE/docs/`](RISE/docs/)。

---

## 1. 安装

```bash
cd policy/RISE   # 在仓库根目录下执行
bash install.sh RISE
conda activate RISE
```

`install.sh` 不会自动下载大文件。按下面路径准备好权重和数据后，直接运行 `train.sh` / `eval.sh` 即可，**无需额外 export**。

### 1.1 Pi0.5 预训练权重（训练必需）

`train.sh` 默认读取 `policy/RISE/weights/pi05_base_pytorch/`（需含 `model.safetensors` 或 `model.pt`）。

#### 方式 A：链接已有 PyTorch 权重（推荐）

若已有转换好的 PyTorch 权重目录，只需 symlink 一次：

```bash
cd policy/RISE
bash setup_weights.sh <path/to/pi05_base_pytorch>
```

#### 方式 B：从 JAX `pi05_base` 自行转换

RISE 训练走 PyTorch 路径，需先把 OpenPI 的 JAX checkpoint 转成 PyTorch。转换脚本位于 vendored 副本 `RISE/policy_and_value/policy_offline_and_value/examples/convert_jax_model_to_pytorch.py`，**须**使用与 `pi05_base` 对应的配置名（本副本中为 `Pi05_base_convert`）。

```bash
cd policy/RISE
conda activate RISE

OFFLINE_DIR="RISE/policy_and_value/policy_offline_and_value"
cd "${OFFLINE_DIR}"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

# 1) 下载 JAX pi05_base，或设 JAX_CKPT 为本地目录（须含 params/）
JAX_CKPT=$(python -c "from openpi_value.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base'))")
# JAX_CKPT=<path/to/pi05_base>

# 2) 输出到 policy/RISE/weights/pi05_base_pytorch
python examples/convert_jax_model_to_pytorch.py \
  --config_name Pi05_base_convert \
  --checkpoint_dir "${JAX_CKPT}" \
  --output_path ../../../weights/pi05_base_pytorch \
  --precision bfloat16
```

说明：

- `--checkpoint_dir` 指向 **含 `params/` 的 JAX 根目录**（例如 `.../pi05_base`），不是 `.../pi05_base/params`。
- 转换产物为 `weights/pi05_base_pytorch/model.safetensors` + `config.json`（若有 JAX `assets/` 会一并复制）。
- 转换需 GPU 与足够内存；若已有方式 A 的权重，无需重复转换。

### 1.2 LeRobot 数据集（训练必需，一次性）

链接已有 LeRobot 数据集并计算 norm stats（`*_w_adv` 需与 raw 位于同一父目录）：

```bash
cd policy/RISE
bash process_lerobot.sh <path/to/lerobot_dataset> [link_name]
```

完成后会有：

- `data/<link_name>` → raw 数据集 symlink
- 同目录下的 `..._w_adv` 由 labeling 生成；若已存在则可直接训练 policy

`train.sh` 在找不到按 6 元组命名的本地数据时，会尝试 `data/` 下已存在的默认 symlink（可用 `RISE_RAW_DATASET` 覆盖）。

---

## 2. 数据准备

RISE 离线训练使用 **LeRobot v2.1**，分辨率 `(240, 320, 3)`，state/action 为 **14 维 joint**（与 `Policy_offline_release` 一致）。

相机键因数据来源而异（`train.sh` / `config.py` 通过 `RISE_LEROBOT_LAYOUT` 或数据集 `meta/info.json` 自动选择）：

| 来源 | LeRobot 图像键 | 说明 |
|------|----------------|------|
| `process_data.sh`（HDF5 转换） | `observation.images.top_head` 等 RISE 原生名 | 与上游 AgileX 布局一致 |
| `process_lerobot.sh`（链接已有 LeRobot） | `observation.images.cam_high` 等 | `RISE_LEROBOT_LAYOUT=robodojo`（或由 `meta/info.json` 推断） |

**注意**：当前 XPolicyLab 推理仅支持 `action_type=joint`。

### 2.1 从 XPolicyLab HDF5 转换

与主仓库一致：HDF5 与 `env_cfg` 放在 **仓库根目录的上一级**（`process_data.sh` / `eval.sh` 中的 `ROOT_DIR` = `policy/RISE` 向上三级）。布局见仓库根目录 `README.md`。

```bash
cd policy/RISE
bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type>
```

- **输入**：`${ROOT_DIR}/data/<dataset_name>/<task_name>/<env_cfg_type>/data/episode_*.hdf5`
- **输出**：`policy/RISE/data/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>-lerobot/`
- 脚本末尾会自动计算 raw 数据集的 `norm_stats.json`（写入 `RISE/policy_and_value/policy_offline_and_value/data/norms/<asset_id>/`）

---

## 3. 训练（离线 policy）

RISE 离线流程分两阶段：**advantage**（value 模型 + 数据打标）与 **policy**（advantage-conditioned policy）。`train.sh` 遵循 XPolicyLab 标准位置参数：

```bash
cd policy/RISE
bash train.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id> [stage] [extra args]
```

### 3.1 Stage 说明

| Stage | 作用 |
|-------|------|
| `advantage` | 计算 raw norm → 训练 value 模型 → 用 value 给数据打 advantage 标签，生成 `*_w_adv` 数据集 |
| `policy` | 在已有 `*_w_adv` 数据集上训练 `Policy_offline_release` |
| `all` | 依次执行 `advantage` → `policy` |

### 3.2 示例

```bash
# 仅准备 advantage 数据集（norm + value + label）
bash train.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> 100 joint 0 0 advantage

# 在已有 *_w_adv 上训练 policy
bash train.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> 100 joint 0 0 policy

# 完整离线流程
bash train.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> 100 joint 0 0 all
```

### 3.3 默认路径

| 资源 | 默认路径 |
|------|----------|
| Pi0.5 预训练 | `weights/pi05_base_pytorch/`（`setup_weights.sh` 或 §1.1 方式 B） |
| Raw 数据集 | `data/<link_name>`（`process_lerobot.sh` 创建的 symlink）；无 6 元组数据时 `train.sh` 会 fallback 到 `RISE_RAW_DATASET` 或默认 link |
| Advantage 数据集 | `<raw 路径>_w_adv` |
| 训练 checkpoint | `checkpoints/<6-tuple>/` |

- **GPU**：`gpu_id` 支持 `0` 或 `0,1,2,3`（逗号分隔时自动推断 `ngpus_per_node`）
- **Python**：先 `conda activate RISE`，脚本会使用当前环境
- **默认 prompt**：环境变量 `RISE_DEFAULT_PROMPT`（`train.sh` 会设置；可按任务覆盖）

---

## 4. 部署与评测

`deploy.yml` 的 `eval_env` 控制客户端模式：`debug` / `sim` / `real`；`eval_batch` 控制是否 batch 推理。切换模式无需改 `eval.sh`。仿真/真机依赖 `<ROOT_DIR>/data/` 与 `<ROOT_DIR>/env_cfg/`（与主 README 目录约定相同）。

```bash
cd policy/RISE
bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

### 4.1 Checkpoint 解析

`setup_eval_policy_server.sh` 按优先级查找 checkpoint：

1. 环境变量 `RISE_CHECKPOINT_PATH`
2. `checkpoints/<6-tuple>/`（含 `model.safetensors` / `model.pt` / `params/`）
3. `checkpoints/<6-tuple>/Policy_offline_release/Policy_offline_release/<step>/`（取最新 step，或 `RISE_CHECKPOINT_STEP` 指定）

其他可覆盖项：

| 变量 | 作用 |
|------|------|
| `RISE_CONFIG_NAME` | 训练配置名（默认 `Policy_offline_release`） |
| `RISE_DEFAULT_PROMPT` | 推理默认语言指令 |
| `RISE_ASSET_ID` | norm stats 的 asset id（默认可从 checkpoint `assets/` 自动推断） |
| `RISE_MODEL_ACTION_DIM` | 覆盖模型 action 维度 |
| `RISE_MODEL_CLIENT_TIMEOUT` | 客户端 socket 超时（默认 600s） |

### 4.2 跨机部署

在 GPU 机后台运行 `setup_eval_policy_server.sh`，仿真/真机侧用同一 `policy_server_ip:policy_server_port` 连接即可。

---

## 5. 关键设计

### 5.1 `model.py` 推理链

- 加载上游 `openpi_value` 的 `Policy_offline_release` checkpoint
- **仅支持 `action_type=joint`**；ee 模式会在初始化时报错
- obs → policy 的映射：
  - 相机：`cam_head` → `top_head`，`cam_left_wrist` → `hand_left`，`cam_right_wrist` → `hand_right`
  - 图像 resize 到 `(240, 320, 3)` RGB（与训练数据一致；上游内部再 `resize_with_pad` 到 224×224）
  - state：`pack_robot_state(..., source_type="obs")` → 14 维 float32
  - prompt：来自 obs 的 `instruction` / `instructions`
- action：`policy.infer` 输出经 `unpack_robot_state` 还原为 XPolicyLab 动作字典

### 5.2 目录结构（XPolicyLab 层）

```text
policy/RISE/                      # 包装层（train.sh、model.py、eval.sh 等）
├── train.sh, process_data.sh, process_lerobot.sh, setup_weights.sh, ...
├── model.py, deploy.py, deploy.yml
├── data/, weights/, checkpoints/
└── RISE/                         # vendored 上游 + XPolicyLab 新增文件
    ├── process_data.py           # 新增（官方 RISE 无此文件）
    ├── deploy/sitecustomize.py   # 新增（官方 RISE 无此文件）
    ├── assets/norm_stats.json    # 可选，供 train.sh 复用
    └── policy_and_value/policy_offline_and_value/
```

---

## 6. 注意事项

1. **action_type**：当前集成仅实现 joint 推理；LeRobot `robodojo` 布局见 `RISE_LEROBOT_LAYOUT`。
2. **预训练权重**：按 §1.1 放到 `weights/pi05_base_pytorch/`。
3. **Advantage 数据集**：直接跑 `policy` stage 前，需已有 `<raw>_w_adv` 及对应 norm stats。
4. **上游能力**：dynamics / online RL / Piper 部署请直接按 `RISE/docs/` 操作；保留 §7.3 所列新增文件：
   - [安装](RISE/docs/installation.md)
   - [离线 policy & value](RISE/docs/offline_learning.md)
   - [Dynamics model](RISE/docs/dynamics_model.md)
   - [Online training](RISE/docs/online_training.md)
   - [Piper 部署](RISE/docs/deploy.md)


---

## 7. 与上游 RISE 的关系

对照官方仓库：[OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE)。本集成将上游副本放在 `policy/RISE/RISE/`（下文简称 **`RISE/`**）。

### 7.1 维护约定

| 位置 | 说明 |
|------|------|
| `policy/RISE/*.sh`、`model.py` 等 | 包装层，随 XPolicyLab 集成维护 |
| `RISE/process_data.py`、`RISE/deploy/sitecustomize.py` | 相对 [OpenDriveLab/RISE](https://github.com/OpenDriveLab/RISE) **新增**，升级上游后需合并回 vendored 树 |
| `RISE/policy_and_value/…` 等官方文件 | 勿手改；升级时替换官方对应路径，保留上述新增文件 |
| 运行时 `data/norms`、`wandb`、`checkpoints/` | 训练产物，非源码 |

### 7.2 范围裁剪

| 上游模块 | 上游用途 | XPolicyLab 状态 |
|----------|----------|-----------------|
| `policy_and_value/policy_offline_and_value` | 离线 value + advantage-conditioned policy | **已包装**：`train.sh`、`process_*`、`model.py`、`eval.sh` |
| `policy_and_value/policy_online` | 想象 rollout 在线 RL | 保留于 `RISE/`，无 XPolicyLab 入口 |
| `dynamics/dynamics_model` | 组合式 world model | 保留于 `RISE/`，无 XPolicyLab 入口 |
| `deploy/`（Piper 等） | 真机 Piper 部署 | 按上游文档使用，未接入 XPolicyLab client |

### 7.3 文件对照（相对 OpenDriveLab/RISE 官方仓库）

**包装层**（`policy/RISE/`，官方 RISE 仓库根下无对应路径）：

| 文件 | 作用 |
|------|------|
| `train.sh` | 离线训练总入口：6 元组参数、`advantage`/`policy`/`all` stage，设置 `RISE_*` 环境变量 |
| `process_data.sh` | 调用 `RISE/process_data.py` 并计算 norm |
| `process_lerobot.sh` | symlink 已有 LeRobot + `Compute_norm` |
| `setup_weights.sh` | 链接 `weights/pi05_base_pytorch/` |
| `model.py` / `deploy.py` / `deploy.yml` | policy server 与评测 |
| `setup_eval_*.sh` / `eval.sh` | 一键 server + client |

**vendored 树内新增**（官方仓库中不存在，勿在升级时误删）：

| 文件 | 作用 |
|------|------|
| `RISE/process_data.py` | HDF5 → LeRobot（`process_data.sh` 调用） |
| `RISE/deploy/sitecustomize.py` | 评测客户端 socket 超时（`setup_eval_env_client.sh` 经 `PYTHONPATH` 加载） |
| `RISE/assets/norm_stats.json`（可选） | `train.sh` 中 `PRECOMPUTED_RAW_NORM` 的回退源 |

vendored 副本内对 `openpi_value` 的配置等与官方 `main` 可能不同（如 `Pi05_base_convert`、`RISE_XPOLICYLAB_DATASET`）；由 `train.sh` 注入环境变量，一般无需再改。

### 7.4 评测适配

- **Checkpoint**：`setup_eval_policy_server.sh`
- **观测**：`model.py`
- **超时**：`RISE/deploy/sitecustomize.py`（`RISE_MODEL_CLIENT_TIMEOUT`）

### 7.5 建议工作流（离线 policy）

1. `bash install.sh RISE && conda activate RISE`
2. `bash setup_weights.sh <path/to/pi05_base_pytorch>`（或 §1.1 方式 B 转换）
3. `bash process_lerobot.sh <path/to/lerobot_dataset> [link_name]`，或 `bash process_data.sh ...`
4. `bash train.sh ... all`（或分 `advantage` / `policy`）
5. `bash eval.sh ...`（在 `deploy.yml` 中设置 `eval_env`：`debug` / `sim` / `real`）

---

## 8. 引用

```bibtex
@article{rise2026,
  title={RISE: Self-Improving Robot Policy with Compositional World Model},
  author={Yang, Jiazhi and Lin, Kunyang and Li, Jinwei and Zhang, Wencong and Lin, Tianwei and Wu, Longyan and Su, Zhizhong and Zhao, Hao and Zhang, Ya-Qin and Chen, Li and Luo, Ping and Yue, Xiangyu and Li, Hongyang},
  journal={arXiv preprint arXiv:2602.11075},
  year={2026}
}
```
