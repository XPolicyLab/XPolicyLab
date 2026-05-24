# OpenVLA_OFT 环境配置

## 1. 创建环境

```bash
conda create -n openvla_oft python==3.10.6
conda activate openvla_oft

pip install torch torchvision torchaudio
```

## 2. 安装 OpenVLA-OFT 源码

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/OpenVLA_OFT/openvla_oft
pip install -e .
```

## 3. 安装训练依赖

```bash
# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
mkdir -p .pip-tmp .pip-cache
TMPDIR=$PWD/.pip-tmp PIP_CACHE_DIR=$PWD/.pip-cache MAX_JOBS=4 pip install "flash-attn==2.5.5" --no-build-isolation
```

## 4. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。