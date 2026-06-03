# LingBot_VA 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n lingbot_va python=3.10.6 -y
conda activate lingbot_va

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install packaging ninja
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 scipy wandb --no-deps
```

### 2. 安装源码与 XPolicyLab

```bash
cd lingbot_va
pip install -e .

cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `LINGBOT_VA_DATASET_PATH` | LeRobot 训练数据完整目录 |
| `LINGBOT_VA_CONFIG_NAME` | 训练配置名（默认 `robotwin30_train`） |
| Wan 权重 | 数据处理脚本 `--model-root` 指向本地 Wan2.2 目录或 HF 缓存 |

## 训练与评测

详见 [README.md](README.md)。
