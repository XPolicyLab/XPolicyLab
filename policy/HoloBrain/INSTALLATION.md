# HoloBrain 安装与全流程指南

---

## 前置条件

- 主机 CUDA toolkit ≥ 12.1（flash-attn wheel 必须匹配该版本）
- `g++` / `build-essential`（pytorch3d 需要源码编译）
- `git`、`wget`、`conda`（miniconda / anaconda）

---

## 1. 安装环境

```bash
cd XPolicyLab/policy/HoloBrain
bash install.sh holobrain
```

1. 创建 conda env `holobrain`（python=3.10）—— 已存在则跳过。
2. 从 `cu128` 源安装 `torch==2.8.0` + `torchvision==0.23.0` —— 已可导入 torch 则跳过。可通过环境变量覆盖：
   ```bash
   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
   TORCH_VERSION=2.6.0 TORCHVISION_VERSION=0.21.0 \
   bash install.sh holobrain
   ```
3. 以 editable 模式安装 `RoboOrchardLab[holobrain_0]`。（若 pytorch3d / flash-attn 在此步构建失败，会自动 fallback 到 core + 安全 extras —— 由步骤 4/5 处理。）
4. 从源码编译 `pytorch3d==0.7.8`（约 10 分钟）。
5. 打印 flash-attn wheel 的 URL —— 请**手动**安装匹配你的 torch + CUDA 组合的 wheel。例如 torch 2.8 / cu12 / py310 / cxx11ABI=False：
   ```bash
   pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
   ```
   到 https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.1 选对应的 wheel。
6. 以 editable 模式将 XPolicyLab 装入同一 env。

最终会打印已安装 / 缺失组件的汇总。

---

## 2. 数据处理

将 XPolicyLab HDF5 数据（`data/<dataset_name>/<task_name>/<env_cfg_type>/data/*.hdf5`）转换为 HoloBrain 的 `robotwin_packer_input/`，并自动打包为 LMDB。

### 通用命令

```bash
cd XPolicyLab/policy/HoloBrain
bash process_data.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type>
```
### 示例

```bash
bash process_data.sh test_data arx_x5 dual_x5 3 joint
```

### 重要约束

- **`action_type` 只能填 `joint`** —— 当前转换流程不支持 ee-pose action。
- **`env_cfg_type` 必须在 `env_cfg/robot/_robot_info.json` 里有条目**，或者在 `env_cfg/<env_cfg_type>.yml` / `env_cfg/robot/<env_cfg_type>.yml` 里存在；否则脚本会报 `Could not resolve env_cfg_type`。
- 输入目录结构必须是 `<repo_root>/data/<dataset_name>/<task_name>/<env_cfg_type>/data/*.hdf5`。如果你的数据集是扁平结构（例如 `test_data/arx_x5/data/*.hdf5`，没有 `<env_cfg_type>` 这一级），需要手动建一层软链接或调整数据布局，否则会报 `HDF5 input not found`。

### 输出

输出到 `data/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>/`：

- `robotwin_packer_input/<task_name>/demo_clean/data/episode*.hdf5` + `seed.txt`
- `xpolicylab_npz/episode_*.npz`
- `manifest.json`、`dataset_statistics.json`
- `lmdb/`（始终生成；后续训练直接读这里）

---

## 3. 可视化检查（强烈建议）

在训练之前**先跑一次可视化**，确认 FK 红点是否正确投影到机械臂关节上。如果红点偏离臂体或全部不可见，先去 [README.md](README.md) 的"快速排查清单"对照排查（典型坑：相机外参约定、URDF 基座 + `T_base2world` 双重变换、关节/link 命名）。

### 通用命令

```bash
cd XPolicyLab/policy/HoloBrain/RoboOrchardLab/projects/holobrain
export POLICY_DIR="$(cd ../../.. && pwd)"
export XPOLICY_HOLOBRAIN_LMDB="${POLICY_DIR}/data/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>/lmdb"
export XPOLICY_HOLOBRAIN_DATASETS=robotwin2_0

python scripts/data_visualize.py \
    --config configs/config_holobrain_qwen_common.py \
    --dataset_names robotwin2_0 \
    --workspace ./workspace_data_check \
    --max_episode 1
```

参数说明：

| 参数                      | 含义                                                            |
| ------------------------- | --------------------------------------------------------------- |
| `--config`                | 训练 config，与训练时保持一致                                   |
| `--dataset_names`         | 要渲染的 dataset key（默认为 `robotwin2_0`）                    |
| `--workspace`             | mp4 输出目录                                                    |
| `--vis_validation`        | 加上则用 validation dataset transforms（默认走 training 流程）  |
| `--max_episode`           | 最多渲染几个 episode；不加则全跑                                |
| `--episode_interval`      | 隔几个 episode 渲一个（默认 1）                                 |
| `--manual`                | 交互模式：手动输入要渲染的 episode 索引                         |

### 示例

```bash
cd XPolicyLab/policy/HoloBrain/RoboOrchardLab/projects/holobrain
export POLICY_DIR="$(cd ../../.. && pwd)"
export XPOLICY_HOLOBRAIN_LMDB="${POLICY_DIR}/data/test_data-arx_x5-dual_x5-3-joint/lmdb"
export XPOLICY_HOLOBRAIN_DATASETS=robotwin2_0
python scripts/data_visualize.py \
    --config configs/config_holobrain_qwen_common.py \
    --dataset_names robotwin2_0 \
    --workspace ./workspace_data_check \
    --max_episode 1
```

输出 mp4 在 `./workspace_data_check/`。每帧从左到右是四个相机视角（默认 `front / left / right / head`），上半部为 RGB + 红绿蓝坐标轴（标识每个关节的位姿），下半部为深度伪彩。

### 如何判断红点是否对齐

- 在腕部相机（`left_camera` / `right_camera`）里，应能看到从机械臂底部一路延伸到夹爪的关节轴序列。
- 在头部相机（`front_camera` / `head_camera`）里，应能看到两个臂的 EE 坐标轴出现在夹爪所在的桌面位置。
- 如果红点全部消失 → 多半是相机外参约定错了（OpenGL vs OpenCV）。
- 如果红点位置整体偏移一个固定量 → 多半是 URDF 基座位置 与 `T_base2world` 没对上。

详细排查见 [README.md](README.md)。

---

## 4. 训练

### 通用命令

```bash
cd XPolicyLab/policy/HoloBrain
bash train.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <gpu_id> <seed> [config_path]
```

参数说明：

| 参数              | 含义                                                                |
| ----------------- | ------------------------------------------------------------------- |
| `dataset_name` ~ `action_type` | 与 `process_data.sh` 完全一致 —— 用于定位 LMDB 路径    |
| `gpu_id`          | 物理 GPU 编号，逗号分隔，如 `0` 或 `4,5,6,7`                        |
| `seed`            | 训练随机种子                                                        |
| `config_path`     | 可选，默认 `configs/config_holobrain_qwen_common.py`                |

`train.sh` 会自动设置：

- `XPOLICY_HOLOBRAIN_LMDB` → `data/<...>/lmdb`（让 `robotwin2_0.paths` 在 dataset config 里指向它）
- `XPOLICY_HOLOBRAIN_DATASETS=robotwin2_0`（跳过没有数据的 `robotwin2_0_ur5_wsg`）
- `CUDA_VISIBLE_DEVICES=$gpu_id`
- `--workspace ./workspace/<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>-seed<seed>`（每次训练根据参数+seed 独立目录，互不污染）

`DualArmKinematics` 使用的双臂 URDF 在 `RoboOrchardLab/projects/holobrain/urdf/arx5/arx5_description_isaac.urdf`，joint id 通过 `config_robotwin_dataset.py` 显式配置（`left=[10-15]`、`right=[18-23]`）。

### 示例

```bash
bash train.sh test_data arx_x5 dual_x5 3 joint 0 0
```
---

## 5. 导出模型

### 通用命令

```bash
cd XPolicyLab/policy/HoloBrain
bash export.sh <workspace_path>
```

参数说明：

| 参数              | 含义                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------ |
| `workspace_path`  | 训练时的 workspace 目录（带 `checkpoints/`），即 `./workspace/<run_key>`                   |

### 示例

```bash
bash export.sh ./workspace/test_data-arx_x5-dual_x5-3-joint-seed0
```

`export.sh` 做的事：

1. 把 `RoboOrchardLab/projects/holobrain/urdf/` 暂存到 `workspace/urdf/`（`export.py` 会把它嵌入 `workspace/model/urdf/`，推理时 `DualArmKinematics` 需要它）。
2. 设置 `XPOLICY_HOLOBRAIN_DATASETS=robotwin2_0`，让导出的 inference config 只包含训练用的那一个。
3. 执行 `python3 scripts/export.py --config <config> --workspace <ws>`。

输出目录结构：

```
workspace/
├── configs/                            # 完整 config 快照
└── model/
    ├── model.config.json
    ├── model.safetensors                # 或分片文件
    ├── robotwin2_0.config.json          # 每个 dataset 的 inference cfg
    ├── robotwin2_0_processor.json       # processor cfg
    └── urdf/arx5/arx5_description_isaac.urdf
```

之后修改 `deploy.yml`：

```yaml
model_dir: ./workspace/test_data-arx_x5-dual_x5-3-joint-seed0/model
inference_prefix: robotwin2_0
```

---

## 6. Debug 验证（端到端冒烟测试）

注意：`eval.sh` 用的 `env_cfg_type` 是 `arx_x5` 而不是 `dual_x5` —— XPolicyLab 的 `get_robot_action_dim_info` 通过顶层 `env_cfg/<...>.yml` 解析 robot；`dual_x5` 只在 `process_data.sh` 阶段有效（走 HoloBrain 自己的 robot-info 查找）。

### 通用命令

```bash
cd XPolicyLab/policy/HoloBrain
bash eval.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env>
```

参数说明：

| 参数                          | 含义                                                |
| ----------------------------- | --------------------------------------------------- |
| 前 7 个                       | 与 `train.sh` 一致（但 `env_cfg_type` 用 `arx_x5`） |
| `policy_conda_env`            | 策略推理用的 conda env（一般填 `holobrain`）        |
| `eval_env_conda_env`          | 评估环境用的 conda env（一般填 `XPolicyLab`）       |

### 示例

```bash
bash eval.sh test_data arx_x5 arx_x5 3 joint 0 0 holobrain XPolicyLab
```

`eval.sh` 做的事：

1. 在 `holobrain` env 里启动 `setup_policy_server.py`，它会加载 `XPolicyLab.policy.HoloBrain.model.Model` 并通过 `ModelServer` 在空闲端口提供服务。
2. 执行 `setup_env_client.sh`，根据 `deploy.yml` 里的 `eval_env` 分发：
   - `debug` → `debug_env_client.py`（mock 480×640 RGB+depth observation）
   - `sim`   → `run_sim_env_client.sh`
   - `real`  → `run_real_policy_client.sh`
3. 退出时清理 server。

`model.py` 会把 observation 缩放到 240×320 RGB，并通过 `pack_robot_state` 把关节状态打包成扁平向量 `[left_arm(6), left_ee(1), right_arm(6), right_ee(1)]`（dual_x5 共 14 维）。

---

## 常见问题排查

**flash-attn 导入报错 `libcusparseLt.so.0`**
wheel 与主机 CUDA 不匹配。到 https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.1 选对应 cuXX/torchYY 的 wheel 重装。

**pytorch3d 编译失败（`g++: not found` 或 CUDA 版本不一致）**
`apt install build-essential`；确认 `nvcc --version` 与 torch 的 CUDA build 一致。也可以装预编译 wheel：https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

**可视化视频里红点位置不对**
看 [README.md](README.md) 的"快速排查清单"。典型原因：相机外参约定、URDF 基座 + `T_base2world` 没对齐、关节 id 错位。

**推理失败：`config file not found: inference.config.json`**
`deploy.yml` 的 `inference_prefix` 写错了。改成训练时使用的 dataset 名（arx_x5 流程用 `robotwin2_0`）。
