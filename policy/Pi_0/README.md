# Pi_0

Pi_0 基于 openpi 接入 XPolicyLab。环境安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

如需先转换为 LeRobot/openpi 数据，在 `openpi` 子目录运行：

```bash
cd openpi
python scripts/process_data.py <task_name> <env_cfg_type> <repo_id> <mode> [instruction]
bash scripts/compute_norm_stats.sh <config_name> <max_frames>
```

## 训练

统一 7 参数入口：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0
```

Checkpoint 保存到：

```text
checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>/
```

可覆盖环境变量：

| 变量 | 说明 |
|------|------|
| `OPENPI_TRAIN_CONFIG_NAME` | openpi 训练配置名 |
| `OPENPI_LOCAL_CACHE_ROOT` | HF / JAX 本地缓存根目录 |

## 评估

```bash
bash eval.sh <task_name> <env_cfg_type> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <model_path> <train_config_name> <repo_id>
```
