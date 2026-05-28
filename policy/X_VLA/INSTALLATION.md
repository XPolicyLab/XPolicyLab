# X_VLA 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n XVLA python=3.10 -y
conda activate XVLA
```

### 2. 安装 X-VLA

```bash
cd xvla
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `XVLA_MODEL_PATH` | 预训练权重（HF repo id 或本地目录；`train.sh` 默认） |
| `XVLA_META_PATH` | 训练 metadata JSON（默认 `xvla/meta.json`） |

## 训练与评测

详见 [README.md](README.md)。
