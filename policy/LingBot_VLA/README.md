# LingBot_VLA

LingBot_VLA 使用 LeRobot 与 yaml 配置训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## Norm 统计

默认 norm 配置：`lingbot_vla/configs/norm/robodojo_sim_arx_x5.yaml`（可按本机修改 `train_path` 等路径）。

```bash
cd lingbot_vla
export DATASET_NAME="RoboDojo_sample"

bash compute_norm_stat.sh configs/norm/${DATASET_NAME}_customized.yaml

python scripts/conver_norm_stat.py assets/norm_stats/${DATASET_NAME}_customized.json assets/norm_stats/${DATASET_NAME}.json
```

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

| 变量 | 说明 |
|------|------|
| `LINGBOT_VLA_CONFIG_PATH` | 默认 `configs/vla/robodojo_sim_arx_x5.yaml` |
| `LINGBOT_VLA_DATA_PATH` | 数据集路径 |

Checkpoint：`checkpoints/<6-tuple>/`

## 评估

部署需在 checkpoint 目录保留 `lingbotvla_cli.yaml`。

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
