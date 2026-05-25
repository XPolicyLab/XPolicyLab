# InternVLA_A1

InternVLA_A1 已接入 XPolicyLab 的本地 policy server，用于 joint action 推理与训练。

## 训练

训练入口遵循 XPolicyLab 统一的 7 参数约定：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0,1,2,3
```

`train.sh` 会将训练输出固定保存到：

```text
policy/InternVLA_A1/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

底层训练使用 `internvla_a1/launch/internvla_a1_3b_finetune.sh`。默认会将数据集 repo id 设为：

```text
<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

如需覆盖底层数据集名称，可设置：

```bash
INTERNVLA_REPO_ID=<lerobot_repo_id> bash train.sh ...
```

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <ckpt_path> [stats_key] [dtype]
```

当前封装默认使用 `joint` 动作类型。训练脚本中的 `action_type` 主要参与 XPolicyLab 的数据与 checkpoint 命名。
