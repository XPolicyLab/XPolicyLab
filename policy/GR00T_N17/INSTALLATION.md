# GR00T_N17 环境配置

NVIDIA Isaac GR00T N1.7 在 XPolicyLab 中的接入。上游源码位于 [gr00t_n17/](gr00t_n17/)，使用 `uv`（Python 3.10，建议 CUDA 12.8 dGPU）。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 系统依赖

`ffmpeg` 用于读取视频；从 HuggingFace 拉模型时建议安装 `git-lfs`：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git-lfs
git lfs install
```

### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"  # 若当前 shell 找不到 uv
```

### 3. 创建 GR00T 环境

在 **x86_64 GPU 主机**上，上游 `pyproject.toml` 的 `required-environments` 会同时解析 aarch64 专用 wheel，导致 `uv sync` 失败。请使用仓库内 `install.sh`（`uv venv --clear` + `uv pip install -e .`）：

```bash
cd policy/GR00T_N17
bash install.sh
source gr00t_n17/.venv/bin/activate
python -c "import gr00t; print('GR00T installed successfully')"
```

评测时 `policy_conda_env` 填 **`uv`**（`setup_eval_policy_server.sh` 会激活 `gr00t_n17/.venv`）。

若提示 `CUDA_HOME is unset`：

```bash
uv run bash scripts/deployment/dgpu/install_deps.sh
```

### 4. 安装 XPolicyLab

在 `gr00t_n17` 目录下：

```bash
uv pip install -e ../../..
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"
```

## 准备 RoboDojo 数据

源数据集需为 LeRobot v3.0；GR00T 训练入口需要 GR00T-flavored LeRobot v2.1 及 `meta/modality.json`。

```bash
export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$HF_LEROBOT_HOME}"
export DATA_ROOT="${LEROBOT_DATA_ROOT}"
export SRC_DATASET="${GR00T_SRC_DATASET:-RoboDojo_sim_arx-x5_v30}"
export GR00T_DATASET="${GR00T_DATASET:-RoboDojo_sim_arx-x5_gr00t}"

cp -a "${DATA_ROOT}/${SRC_DATASET}" "${DATA_ROOT}/${GR00T_DATASET}"

cd gr00t_n17
uv run --project scripts/lerobot_conversion \
  python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --root "${DATA_ROOT}" \
  --repo-id "${GR00T_DATASET}"
```

或使用 policy 封装脚本：

```bash
bash process_data.sh RoboDojo cotrain arx_x5 3500 joint
```

### 补充 meta/modality.json

在转换后的数据目录创建（RoboDojo arx-x5，14 维 state/action）：

```bash
cat > "${DATA_ROOT}/${GR00T_DATASET}/meta/modality.json" <<'EOF'
{
  "state": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "action": {
    "left_arm": { "start": 0, "end": 7 },
    "right_arm": { "start": 7, "end": 14 }
  },
  "video": {
    "front": { "original_key": "observation.images.cam_high" },
    "left_wrist": { "original_key": "observation.images.cam_left_wrist" },
    "right_wrist": { "original_key": "observation.images.cam_right_wrist" }
  },
  "annotation": {
    "human.task_description": { "original_key": "task_index" }
  }
}
EOF
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `LEROBOT_DATA_ROOT` | LeRobot 数据集根目录（默认 `$HF_LEROBOT_HOME`） |
| `GR00T_SRC_DATASET` | 源 v3.0 数据集 repo id |
| `GR00T_DATASET` | 转换后的 GR00T 数据集 repo id |
| `GR00T_BASE_MODEL_PATH` | GR00T-N1.7-3B 本地目录或 HF id（见 `train.sh`） |
| `GR00T_COSMOS_MODEL_PATH` | 部署推荐 `checkpoints/shared/Cosmos-Reason2-2B`（软链 xspark 共享权重） |

预训练权重也可预先 `huggingface-cli download` 到 `$HF_HOME`，`train.sh` 默认 `HF_HUB_OFFLINE=1` 走本地缓存。

## 安装自检

```bash
cd gr00t_n17
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python gr00t/experiment/launch_finetune.py --help
uv run python -c "import XPolicyLab; print('XPolicyLab ok')"
```

## 评测环境

Policy server 使用 `gr00t_n17/.venv`；env client 可使用任意已安装 XPolicyLab 的 conda 环境：

```bash
bash install.sh
# client 侧示例
conda activate <your_eval_env>
cd ../../..
pip install -e .
```

训练与评测入口见 [README.md](README.md)。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `uv` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `cotrain` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/GR00T_N17/RoboDojo-cotrain-arx_x5-3500-joint-0` |
| 备注 | cosmos: checkpoints/shared/Cosmos-Reason2-2B |

软链 checkpoint（在 `policy/GR00T_N17/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 0 0 uv <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls cotrain arx_x5 joint 0 0 XPolicyLab "ckpt_name=cotrain,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

