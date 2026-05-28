# SmolVLA

SmolVLA 基于 LeRobot SmolVLA 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 训练

```bash
source .venv/bin/activate
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

默认 LeRobot repo id：`<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>`

可用 `SMOVLA_REPO_ID` 覆盖。

Checkpoint：`checkpoints/<6-tuple>/`

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <pretrained_path>
```
