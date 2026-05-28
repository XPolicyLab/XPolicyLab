# Pi_0 环境配置

Pi_0 基于 [openpi](openpi/)，使用 `uv` 管理环境。默认训练配置为 `pi0_base_aloha_full_sim_arx-x5_seed_0`。

## 一键安装

在 policy 目录下执行：

```bash
bash install.sh
```

## 手动安装

### 1. 配置 openpi 环境

```bash
cd openpi
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### 2. 安装 XPolicyLab

```bash
source .venv/bin/activate
cd ../../..
uv pip install -e .
```

## 模型与数据路径

| 用途 | 说明 |
|------|------|
| 预训练权重 | 由 openpi 训练配置自动从 HuggingFace / 配置中的 `assets_dir` 拉取 |
| Checkpoint | `policy/Pi_0/checkpoints/<6-tuple>/`（`train.sh` 的 `--checkpoint-dir-override`） |
| 训练配置名 | 环境变量 `OPENPI_TRAIN_CONFIG_NAME`（默认 `pi0_base_aloha_full_sim_arx-x5_seed_0`） |
| 本地缓存 | `OPENPI_LOCAL_CACHE_ROOT`（默认 `/tmp/openpi-cache-$(hostname)`） |

## 训练与评测

详见 [README.md](README.md)。
