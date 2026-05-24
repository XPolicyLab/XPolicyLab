# SmoVLA 环境配置

SmoVLA 依赖 LeRobot v0.4.4 和 SmolVLA extra。只安装 LeRobot 基础包通常会缺少 `transformers`、`num2words`、`safetensors` 等依赖。

## 1. 创建环境

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/SmoVLA
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 2. 系统依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  git ffmpeg cmake build-essential pkg-config python3-dev \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

## 3. 安装 SmoVLA / LeRobot 源码

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/SmoVLA/smovla
pip install -e ".[smolvla]"
```

如需 PEFT：

```bash
pip install -e ".[smolvla,peft]"
```

## 4. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
pip install h5py
```

## 5. 自检

```bash
python -c "import lerobot; print('lerobot ok')"
python -c "from lerobot.policies.factory import get_policy_class; print(get_policy_class('smolvla'))"
python -c "import XPolicyLab, av, transformers, safetensors, h5py; print('deps ok')"
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。
