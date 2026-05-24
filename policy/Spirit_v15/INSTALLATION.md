# Spirit_v15 环境配置

## 1. 配置模型环境

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/Spirit_v15/spirit_v15
uv sync --extra train
source .venv/bin/activate
uv pip install -e .
```

如果不使用 `uv`，可使用 pip：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/Spirit_v15/spirit_v15
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-base.txt
pip install -r requirements-train.txt
pip install -e .
```

## 2. 安装 XPolicyLab

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
pip install -e .
```

训练入口见 `README.md`，统一使用 `bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>`。