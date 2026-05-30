# OpenVLA_OFT 环境配置

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境

```bash
conda create -n openvla_oft python=3.10.6 -y
conda activate openvla_oft
pip install torch torchvision torchaudio
```

### 2. 安装 OpenVLA-OFT

```bash
cd openvla_oft
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```

### 3. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `TFDS_DATA_DIR` | TensorFlow Datasets 根目录 |
| `OPENVLA_TFDS_DATASET_NAME` | 训练用 TFDS 名称 |

基座 VLA 权重通常由 OpenVLA 配置或 HF 指定，见上游文档。

## 训练与评测

详见 [README.md](README.md)。
