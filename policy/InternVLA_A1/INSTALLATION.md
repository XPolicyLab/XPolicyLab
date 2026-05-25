# InternVLA_A1 环境配置

## 1. 创建环境

```bash
conda create -n internvla_a1 python=3.10 -y
conda activate internvla_a1
```

## 2. 安装 policy 源码

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/InternVLA_A1/internvla_a1
pip install -e .
```

## 3. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

## 4. 自检

```bash
python -c "import XPolicyLab; print('XPolicyLab ok')"
python -c "import lerobot; print('lerobot ok')"
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。
