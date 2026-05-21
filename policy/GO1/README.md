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
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint

# ee 模式
bash process_data.sh RoboDojo stack_bowls arx_x5 103 ee
```

GO1默认使用绝对位置的 ee 控制

转换后的 LeRobot 数据默认保存到：

```text
policy/GO1/data/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}
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
REPORT_TO=tensorboard bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0,1,2,3 42
```

```bash
export REPORT_TO=wandb
export WANDB_PROJECT=go1
export WANDB_API_KEY=<your_wandb_api_key>

bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0,1,2,3 42
```

## 评测 / 部署

评测入口：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${policy_gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH] [env_gpu_id]
```

```bash
conda activate your_env
cd /path/to/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 your_env your_env
```

参数说明：

- `policy_gpu_id`: GO1 模型推理服务使用的 GPU。
- `env_gpu_id`: 环境 / 仿真 client 使用的 GPU。可选；不传时默认等于 `policy_gpu_id`。
- `MODEL_PATH`: 可选；显式指定某个 `checkpoint-*` 目录。不传时脚本会自动从 `checkpoints/` 下查找当前任务最新 checkpoint。

当前评测脚本已按 `demo_policy` 的形式拆成三段：

- `eval.sh`: 总控脚本。负责自动寻找空闲端口、自动定位 checkpoint 和 `dataset_stats.json`、启动 server 和 client。
- `setup_eval_policy_server.sh`: 只负责启动 GO1 policy server。
- `setup_eval_env_client.sh`: 只负责启动环境 client。

因此现在可以直接一键评测，不需要手动分别拉起 server / client。

### 一键测评

如果你已经完成训练，并希望直接用最近一次 checkpoint 做 debug/sim 评测，执行：

```bash
conda activate your_env
cd /path/to/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 your_env your_env
```

如果你想显式指定某个 checkpoint：

```bash
conda activate your_env
cd /path/to/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 lqw lqw \
  /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1/checkpoints/stack_bowls-go1-joint-5eps-seed42-20260521_152745/checkpoint-1000
```

如果你希望模型和环境分开占用 GPU，例如模型跑 `0` 卡、环境跑 `1` 卡：

```bash
conda activate your_env
cd /path/to/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 your_env your_env "" 1
```

说明：

- 第 10 个参数是 `MODEL_PATH`。如果想让脚本自动找最新 checkpoint，但又要传第 11 个参数 `env_gpu_id`，则第 10 个参数传空字符串 `""`。
- `dataset_stats.json` 会由 `eval.sh` 自动从训练输出目录中查找，不需要手动指定。
- 当前默认评测环境由 [deploy.yml](/mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1/deploy.yml) 中的 `eval_env` 控制；默认是 `debug`。

