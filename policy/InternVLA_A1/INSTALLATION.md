# InternVLA_A1 环境配置

上游详细说明见 [internvla_a1/tutorials/installation.md](internvla_a1/tutorials/installation.md)。

## 一键安装

```bash
bash install.sh
```

## 手动安装

### 1. 创建环境并安装 InternVLA

```bash
conda create -n internvla_a1 python=3.10 -y
conda activate internvla_a1

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf

cd internvla_a1
pip install -e .

TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/
cp -r src/lerobot/policies/pi0/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models "${TRANSFORMERS_DIR}"
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models "${TRANSFORMERS_DIR}"
```

### 2. 安装 XPolicyLab

```bash
cd ../../..
pip install -e .
```

## 模型与数据路径

| 变量 | 说明 |
|------|------|
| `PRETRAINED_PATH` | 在 `internvla_a1/launch/internvla_a1_3b_finetune.sh` 中设置，可为 HF id 或本地目录 |
| `INTERNVLA_REPO_ID` | LeRobot 数据集 repo id（`train.sh` 可覆盖） |

## 训练与评测

详见 [README.md](README.md)。
