# LingBot_VLA

LingBot_VLA 使用 LeRobot 与 yaml 配置训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## Norm 统计

```bash
cd lingbot_vla
bash compute_norm_stat.sh /path/to/norm_config.yml
python scripts/conver_norm_stat.py <customized_json> <output_json> <left_arm_dim> <left_ee_dim> <right_arm_dim> <right_ee_dim>
```

配置示例（路径请按本机填写）：

```yaml
data:
  train_path: <lerobot_dataset_dir>
  norm_path: assets/norm_stats/example_customized.json
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
