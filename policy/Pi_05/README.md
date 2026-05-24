# Pi_05

Pi_05 基于 openpi 接入 XPolicyLab，默认训练配置为 `pi05_base_aloha_full_sim_arx-x5_seed_0`。

## 数据处理

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/Pi_05/openpi
python scripts/process_data.py <task_name> <env_cfg_type> <repo_id> <mode> [instruction]
bash scripts/compute_norm_stats.sh <config_name> <max_frames>
```

## 训练

训练入口遵循 XPolicyLab 统一 7 参数：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

默认 checkpoint 保存到：

```text
policy/Pi_05/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

如需切换 openpi config：

```bash
OPENPI_TRAIN_CONFIG_NAME=pi05_base_aloha_full_sim_arx-x5_seed_0 bash train.sh ...
```

## 评估

```bash
bash eval.sh <task_name> <env_cfg_type> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <model_path> <train_config_name> <repo_id>
```