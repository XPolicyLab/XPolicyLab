# LingBot_VLA

LingBot_VLA 使用 LeRobot 数据和 yaml 配置训练，已封装为 XPolicyLab policy。

## 数据与 norm stat

先准备 LeRobot 数据集，并在 `lingbot_vla/assets/norm/` 下编写 norm 统计配置：

```yaml
data:
  datasets_type: vla
  train_path: /path/to/lerobot/dataset
  norm_path: assets/norm_stats/example_customized.json

train:
  global_batch_size: 512
  output_dir: output/norm
```

计算并转换 norm：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/LingBot_VLA/lingbot_vla
bash compute_norm_stat.sh /path/to/norm_config.yml
python scripts/conver_norm_stat.py <customized_json> <output_json> <left_arm_dim> <left_ee_dim> <right_arm_dim> <right_ee_dim>
```

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

默认配置：

```text
LINGBOT_VLA_CONFIG_PATH=configs/vla/robodojo_sim_arx_x5.yaml
LINGBOT_VLA_DATA_PATH=<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

可按需覆盖：

```bash
LINGBOT_VLA_CONFIG_PATH=configs/vla/my_task.yaml \
LINGBOT_VLA_DATA_PATH=/path/to/lerobot/dataset \
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0,1
```

训练输出固定保存到：

```text
policy/LingBot_VLA/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

## 评估

部署时需要在 checkpoint 目录下保留或复制对应的 `lingbotvla_cli.yaml`。

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
