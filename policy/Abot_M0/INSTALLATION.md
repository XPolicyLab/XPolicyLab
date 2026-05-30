# Abot_M0 环境配置

ABot 上游安装与数据集准备详见 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md)。

## 一键安装

```bash
bash install.sh
```

## 手动安装（XPolicyLab 集成）

### 1. 安装 ABot 环境

按 [abot_m0/INSTALLATION.md](abot_m0/INSTALLATION.md) 创建 conda 环境并安装 `abot_m0`（需单独 clone `ABot-Manipulation` 与 `vggt`）。

### 2. 安装 XPolicyLab

```bash
cd ../..
pip install -e .
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
