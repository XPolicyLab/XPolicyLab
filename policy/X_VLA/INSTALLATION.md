# X_VLA 环境配置

## 1. 创建环境

``` bash
conda create -n XVLA python=3.10 -y
conda activate XVLA
```

## 2. 安装 X-VLA 源码

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/X_VLA/xvla
pip install -r requirements.txt

pip show torch
# 安装对应torch版本的cuda版本, 否则训练可能报错
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 # 举例
```

## 3. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。
