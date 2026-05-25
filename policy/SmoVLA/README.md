# SmoVLA

SmoVLA 基于 LeRobot SmolVLA 接入 XPolicyLab，当前主要用于 ALOHA/双臂 joint action 训练与推理。

## 训练

训练入口遵循 XPolicyLab 统一的 7 参数约定：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0
```

`train.sh` 默认将 LeRobot 数据集 repo id 设为：

```text
<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

如需使用已有 LeRobot repo，可覆盖：

```bash
SMOVLA_REPO_ID=<lerobot_repo_id> bash train.sh ...
```

训练产物默认保存到：

```text
policy/SmoVLA/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> joint <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <pretrained_path>
```

当前封装默认使用 `joint` 动作类型。部署权重路径应指向训练输出目录下可被 LeRobot 加载的 checkpoint。
