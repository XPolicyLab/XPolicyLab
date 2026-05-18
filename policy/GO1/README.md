# GO1

GO1 policy integration for XPolicyLab. This directory provides data conversion, fine-tuning, checkpoint management, and evaluation scripts for running GO1 inside the XPolicyLab policy server/client workflow.

## 环境配置

GO1 依赖内部的 `AgiBot-World` 代码和 GO-1 预训练权重。环境配置请优先参考内部 GO1 / AgiBot-World 环境文档，并确保当前 policy 环境可以正常 import `torch`、`transformers`、`deepspeed`、`lerobot`、`flash_attn` 等依赖。

默认预训练权重路径在 `train.sh`、`AgiBot-World/go1/configs/go1_sft_xpolicylab.py` 和 `model.py` 中使用，可根据机器实际路径调整，例如：

```text
/path/to/models/GO-1
```

## 数据转化

GO1 使用 LeRobot 格式数据。转换入口：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

示例：

```bash
# joint 模式
bash process_data.sh RoboDojo stack_bowls arx_x5 103 joint

# ee 模式
bash process_data.sh RoboDojo stack_bowls arx_x5 103 ee
```

GO1默认使用绝对位置的 ee 控制

转换后的 LeRobot 数据默认保存到：

```text
policy/GO1/data/${dataset_name}-${task_name}-${env_cfg_type}
```

## Norm Stats

GO1 训练时会根据当前 LeRobot 数据集自动统计并保存归一化参数，不需要手动准备默认 norm 文件。

## 训练

训练入口：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed}
```

正式训练示例：

```bash
export REPORT_TO=wandb
export WANDB_PROJECT=go1
export WANDB_API_KEY=<your_wandb_api_key>

bash train.sh RoboDojo stack_bowls arx_x5 103 ee 0,1,2,3 42
```

## 评测 / 部署

评测入口：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH]
```

```bash
conda activate your_env
cd /path/to/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 your_env your_env
```

### 手动指定 checkpoint

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 103 ee 0 42 your_env your_env \
  /path/to/XPolicyLab/policy/GO1/checkpoints/stack_bowls-go1-ee-103eps-seed42-YYYYMMDD_HHMMSS/checkpoint-20000
```
