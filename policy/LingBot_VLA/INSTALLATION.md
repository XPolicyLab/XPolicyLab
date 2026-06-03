# LingBot_VLA 环境配置

## 一键安装

```bash
bash install.sh
```

可选：指定 flash-attn wheel URL（cu128 + torch2.8 + py3.12）：

```bash
export FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n lingbot_vla python=3.12 -y
conda activate lingbot_vla

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install lerobot==0.4.4
```

### 2. 安装 LingBot_VLA

```bash
cd lingbot_vla
git submodule update --init --recursive
# 安装 flash-attn（见上游 release 或 pip install flash-attn）
pip install -e .
pip install -r requirements.txt
cd lingbotvla/models/vla/vision_models/lingbot-depth/
pip install -e . --no-deps
cd ../MoGe
pip install -e .
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

训练 yaml 中常用字段（可用环境变量覆盖，见 `train.sh`）：

| 配置项 / 变量 | 说明 |
|---------------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `train_path` / `LINGBOT_VLA_DATA_PATH` | LeRobot 数据集完整路径 |
| `model_path` | 基座权重（HF repo id 或本地目录） |
| `tokenizer_path` | 分词器（如 `Qwen/Qwen2.5-VL-3B-Instruct`） |
| `LINGBOT_VLA_CONFIG_PATH` | 训练 yaml，默认 `configs/vla/robodojo_sim_arx_x5.yaml` |

## 训练与评测

详见 [README.md](README.md)。
