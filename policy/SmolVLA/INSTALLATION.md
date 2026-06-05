# SmolVLA 环境配置

SmolVLA 依赖 **LeRobot v0.4.4** 与 `[smolvla]` extra。`install.sh` 会：

1. 创建 conda 环境（默认 `smolvla`，Python 3.10）
2. 将 [huggingface/lerobot](https://github.com/huggingface/lerobot) 克隆到 `policy/SmolVLA/smovla/`
3. 在 `smovla` 目录执行 `pip install -e ".[smolvla]"`
4. 安装 XPolicyLab 根目录（`pip install -e .`）

## 一键安装

```bash
cd XPolicyLab/policy/SmolVLA
bash install.sh
conda activate smolvla
```

## 环境变量（可选）

| 变量 | 默认 | 说明 |
|------|------|------|
| `SMOVLA_CONDA_ENV` | `smolvla` | conda 环境名 |
| `SMOVLA_PYTHON_VERSION` | `3.10` | 创建环境时的 Python 版本 |
| `LEROBOT_REF` | `v0.4.4` | 克隆的 git tag/branch |
| `LEROBOT_REPO` | `https://github.com/huggingface/lerobot.git` | 仓库地址 |
| `SMOVLA_SKIP_CONDA_CREATE` | `0` | 设为 `1` 则跳过 `conda create` |
| `SMOVLA_UPDATE_LEROBOT` | `0` | 设为 `1` 且 `smovla/` 已存在时拉取并 checkout `LEROBOT_REF` |
| `SMOVLA_TORCH_INDEX` | （空） | 例如 `https://download.pytorch.org/whl/cu128`，先装 torch 再装 lerobot |

## 系统依赖（视频编解码）

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg cmake build-essential pkg-config python3-dev \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

## 手动安装（与脚本等价）

```bash
conda create -n smolvla python=3.10 -y
conda activate smolvla

git clone --branch v0.4.4 --depth 1 https://github.com/huggingface/lerobot.git smovla
cd smovla
pip install -e ".[smolvla]"
# 可选: pip install -e ".[smolvla,peft]"

cd ../../..
pip install -e .
pip install h5py
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `SMOVLA_REPO_ID` | LeRobot 数据集 repo id（`train.sh` 可覆盖） |
| 预训练 | LeRobot / HF 默认拉取 `lerobot/smolvla_base` |

## 训练与评测

详见 [README.md](README.md)。评测与 `deploy.sh` 请使用 conda 环境名 `smolvla`（或你设置的 `SMOVLA_CONDA_ENV`）。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `smolvla` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo_sim_arx-x5_seed_0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/SmoVLA/RoboDojo_sim_arx-x5_seed_0` |

软链 checkpoint（在 `policy/SmolVLA/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 3500 joint 0 0 smolvla <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo_sim_arx-x5_seed_0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

