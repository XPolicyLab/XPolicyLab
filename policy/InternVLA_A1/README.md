# InternVLA_A1

InternVLA_A1 已接入 XPolicyLab 的本地 policy server，用于 joint action 推理与训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 训练
首先修改internvla_a1/launch/internvla_a1_3b_finetune.sh的`PRETRAINED_PATH`.  

训练入口遵循 XPolicyLab 统一的 7 参数约定：

```bash
# 计算norm stat
bash compute_norm.sh <repo_id>

# 开启训练
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
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

## 评测（XPolicyLab）

环境安装见 [INSTALLATION.md](INSTALLATION.md)。手动部署推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh`（便于查看 server 报错）。

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_seed_0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> internvla_a1 XPolicyLab
```

Pi_0 / Pi_0_Fast 需先执行 `Pi_05/install.sh`，server 环境填 `uv`。

