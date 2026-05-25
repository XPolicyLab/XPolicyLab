# InternVLA_A1 环境配置

## 1. 创建环境

```bash
conda create -n internvla_a1 python=3.10 -y
conda activate internvla_a1

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install -e .

TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/

cp -r src/lerobot/policies/pi0/transformers_replace/models        ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models  ${TRANSFORMERS_DIR}
cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models  ${TRANSFORMERS_DIR}
```

## 2. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```