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

## XPolicyLab 部署（eval）

已在 GPU 主机完成 debug client 闭环（`setup_eval_policy_server.sh` + `setup_eval_env_client.sh`）。

| 项 | 说明 |
|----|------|
| Server 环境 | `uv` |
| Client 环境 | `XPolicyLab`（conda） |
| eval 示例 ckpt | `RoboDojo_sim_arx_seed_0` |
| expert_data_num | `3500` |
| action_type | `joint` |
| xspark 权重 | `/mnt/xspark-data/final_ckpt/Pi_0/RoboDojo_sim_arx_seed_0` |
| 备注 | policy_uv_env_path: openpi（uv .venv） |

软链 checkpoint（在 `policy/Pi_0/` 下）：

```bash
mkdir -p checkpoints
ln -sfn <xspark_dir> checkpoints/<6-tuple_dir_name>
```

`ckpt_name` 若已是完整 6-tuple（含多个 `-`），eval 脚本直接传入该目录名。

手动评测：

```bash
# terminal 1 — server
bash setup_eval_policy_server.sh RoboDojo stack_bowls RoboDojo_sim_arx_seed_0 arx_x5 3500 joint 0 0 uv <port> localhost

# terminal 2 — client
bash setup_eval_env_client.sh RoboDojo stack_bowls RoboDojo_sim_arx_seed_0 arx_x5 joint 0 0 XPolicyLab "ckpt_name=RoboDojo_sim_arx_seed_0,action_type=joint" <port> localhost
```

或使用 `eval.sh`（会等待 server 端口就绪后启动 client）。

