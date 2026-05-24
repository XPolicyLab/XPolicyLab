# GigaWorldPolicy 环境配置

## 1. 创建环境

```bash
conda create -n gigaworld-policy python=3.11 -y
conda activate gigaworld-policy
```

## 2. 安装 GigaWorld 依赖

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/GigaWorldPolicy/giga_world_policy
pip install -e ./third_party/giga-train
pip install -e ./third_party/giga-models
pip install -e ./third_party/giga-datasets
```

如需额外依赖，可继续安装各子项目的 `requirements.txt`：

```bash
pip install -r ./third_party/giga-train/requirements.txt
pip install -r ./third_party/giga-models/requirements.txt
pip install -r ./third_party/giga-datasets/requirements.txt
```

## 3. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

## 4. 自检

```bash
python -c "import XPolicyLab; print('XPolicyLab ok')"
python -c "import giga_train; print('giga_train ok')"
```

训练入口见 `README.md`，统一使用：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```
