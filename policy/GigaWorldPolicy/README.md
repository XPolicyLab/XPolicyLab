# GigaWorldPolicy

GigaWorldPolicy 基于 `giga_world_policy` 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

```bash
cd giga_world_policy
bash process_data.sh <lerobot_data_path> <wan_pretrained_path>
```

生成 `norm_stats_delta.json` 与 `<lerobot_data_path>/t5_embedding`。

默认数据：`data/<5-tuple>/`，可用 `GIGAWORLD_DATA_DIR` 覆盖。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`（含 `xpolicylab_train_config.json`）

| 变量 | 说明 |
|------|------|
| `GIGAWORLD_DATA_DIR` | 训练数据 |
| `GIGAWORLD_NORM_PATH` | norm stats |
| `GIGAWORLD_PRETRAINED_PATH` | Wan 预训练 |
| `GIGAWORLD_DRY_RUN=1` | 仅生成 config |

## 评估

`deploy.yml` 中 `eval_env: debug` 时 `load_model: false` 可走零动作调试。完整推理设 `load_model: true`。

Checkpoint 目录示例：

```text
checkpoints/<ckpt_name>/models/checkpoint_epoch_4_step_100000/
```

```bash
bash eval.sh RoboDojo debug_task <ckpt_name> arx_x5 1 joint 0 0 0 gigaworld-policy gigaworld-policy
```
