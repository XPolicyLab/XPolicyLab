# Abot_M0 环境配置

ABot 上游安装与数据集准备详见 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md)。

## 一键安装

先按 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md) 创建 **ABot** conda 环境，再：

```bash
bash install.sh
```

`install.sh` 会在 `ABot` conda（可用 `ABOT_CONDA_ENV` 覆盖）中安装 XPolicyLab 与 `h5py` / `opencv-python` / `pyyaml`。

## 手动安装（XPolicyLab 集成）

### 1. 安装 ABot 环境

按 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md) 创建 conda 环境并安装 `abot_m0`（需单独 clone `ABot-Manipulation` 与 `vggt`）。

### 2. 安装 XPolicyLab（须在 ABot conda 内）

```bash
conda activate ABot
cd ../..
pip install -e .
pip install h5py opencv-python pyyaml
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `BASE_VLM` | Qwen3-VL-4B-Instruct-Action 目录或 HF id |
| `PRETRAIN_CKPT` | ABot-M0 预训练 checkpoint 路径 |
| `RELOAD_MODULES` | 例如 `qwen_vl_interface`（避免 action head 形状不匹配） |
| `HF_LEROBOT_HOME` | LeRobot 数据集根（`--dataset-dir`） |

## RoboDojo 数据准备

```bash
cd abot_m0
cp examples/Robotwin/train_files/modality.json \
   "${HF_LEROBOT_HOME}/<your_repo>/meta/modality.json"

python3 examples/RoboDojo/prepare_RoboDojo_abot.py \
  --dataset-dir "${HF_LEROBOT_HOME}/<your_repo>"
```

## 训练与评测

见 [README.md](README.md) 与 `abot_m0/examples/RoboDojo/`。

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `ABot` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo-sim-arx_x5-100-joint-0` |
| expert_data_num | `100` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Abot_M0/RoboDojo-sim-arx_x5-100-joint-0` |

软链 checkpoint（在 `policy/Abot_M0/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo-sim-arx_x5-100-joint-0 arx_x5 100 joint 0 0 ABot <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo-sim-arx_x5-100-joint-0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo-sim-arx_x5-100-joint-0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

