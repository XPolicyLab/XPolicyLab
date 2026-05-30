# SmolVLA 环境配置

SmolVLA 依赖 LeRobot v0.4.4 与 `smolvla` extra。使用 policy 目录下的 Python venv。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建 venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2. 系统依赖（视频编解码）

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg cmake build-essential pkg-config python3-dev \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

### 3. 安装 SmolVLA / LeRobot

```bash
cd smovla
pip install -e ".[smolvla]"
# 可选: pip install -e ".[smolvla,peft]"
```

### 4. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
pip install h5py
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `SMOVLA_REPO_ID` | LeRobot 数据集 repo id（`train.sh` 可覆盖） |
| 预训练 | LeRobot / HF 默认拉取 SmolVLA 基座权重 |

## 训练与评测

详见 [README.md](README.md)。
