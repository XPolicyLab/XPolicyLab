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

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_seed_0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> internvla_a1 XPolicyLab
```
