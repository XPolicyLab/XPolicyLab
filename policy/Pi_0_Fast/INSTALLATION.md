# Pi_0_Fast 环境配置

## 1. 配置模型环境

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/Pi_0_Fast/openpi
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 2. 安装 XPolicyLab

```bash
source .venv/bin/activate
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab
uv pip install -e .
```

## 3. 说明

Pi_0_Fast 默认训练配置为 `pi0_fast_aloha_full_sim_arx-x5_seed_0`，可通过 `OPENPI_TRAIN_CONFIG_NAME` 覆盖。