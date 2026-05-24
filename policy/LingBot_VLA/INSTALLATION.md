# LingBot_VLA 环境配置

## 1. 创建环境

``` bash
conda create -n lingbot_vla python==3.12
conda activate lingbot_vla

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/huggingface/lerobot.git
# cd lerobot
# git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
# pip install -e .
pip install lerobot==0.4.4

cd ../
# Install flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

pip install ./flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/LingBot_VLA/lingbot_vla
git submodule update --remote --recursive
pip install -e .
pip install -r requirements.txt
# Install LingBot-Depth dependency
cd ./lingbotvla/models/vla/vision_models/lingbot-depth/
pip install -e . --no-deps
cd ../MoGe
pip install -e .
```

## 2. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。