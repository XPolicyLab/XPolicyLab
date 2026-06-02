# GigaWorldPolicy

GigaWorldPolicy 基于 `giga_world_policy` 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

```bash
cd giga_world_policy
bash process_data.sh <lerobot_data_path> <wan_pretrained_path>
```

生成 `norm_stats_delta.json` 与 `<lerobot_data_path>/t5_embedding`。

默认 LeRobot 数据：`${XPOLICYLAB_LEROBOT_DATA_ROOT:-<robodojo_test>/data}/<repo_id>`（`arx_x5` 对应 `RoboDojo_sim_arx-x5_v30`）。可用 `GIGAWORLD_DATA_DIR` 覆盖完整路径，或用 `LEROBOT_DATASET_REPO_ID` 覆盖 repo 名。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`（含 `xpolicylab_train_config.json`）

| 变量 | 说明 |
|------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | 数据集 repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `GIGAWORLD_DATA_DIR` | 训练数据完整路径（覆盖上述默认） |
| `GIGAWORLD_NORM_PATH` | norm stats |
| `GIGAWORLD_PRETRAINED_PATH` | Wan 预训练 |
| `GIGAWORLD_DRY_RUN=1` | 仅生成 config |

训练 seed：`train.seed` 与 `DefaultSampler.seed` 均为 `XPolicyLab_seed + 1`（giga-train 要求 `seed > 0`），并设置 `PYTHONHASHSEED`。

## 评估

`deploy.yml` 中 `eval_env: debug` 时 `load_model: false` 可走零动作调试。完整推理设 `load_model: true`。

Checkpoint 目录示例：

```text
checkpoints/<ckpt_name>/models/checkpoint_epoch_4_step_100000/
```

```bash
bash eval.sh RoboDojo debug_task <ckpt_name> arx_x5 1 joint 0 0 0 gigaworld-policy gigaworld-policy
```
