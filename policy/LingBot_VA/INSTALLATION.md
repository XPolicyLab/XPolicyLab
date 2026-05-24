# LingBot_VA 环境配置

## 1. 创建环境

``` bash
conda create -n lingbot_va python==3.10.6
conda activate lingbot_va

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict

pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
mkdir -p .pip-tmp .pip-cache
TMPDIR=$PWD/.pip-tmp PIP_CACHE_DIR=$PWD/.pip-cache MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## 2. 配置训练环境

```bash
pip install lerobot==0.3.3 scipy wandb --no-deps
```

## 3. 安装 LingBot_VA 源码

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/LingBot_VA/lingbot_va
pip install -e .
```

## 4. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。