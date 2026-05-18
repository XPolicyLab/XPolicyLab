# A1

A1 policy integration for XPolicyLab. This directory wraps the original `A1/` project so it can use XPolicyLab data conversion, training, policy server deployment, and debug/sim evaluation.

## 环境配置

A1 原项目位于：

```text
/mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1/A1
```

默认预训练权重位于：

```text
/mnt/pfs/pg4hw0/qiwei/models/a1-pretrain
```

安装入口：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1
bash install.sh
```

A1 加载 `config.yaml` 时会把原始配置中的 `vit_load_path` 和 `llm_load_path` 映射到 `${DATA_DIR}/pretrained_image_encoders` 和 `${DATA_DIR}/pretrained_llms`。默认 `DATA_DIR=/mnt/pfs/pg4hw0/qiwei/models`。如果你的预训练 backbone 不在这里，需要在运行前设置 `DATA_DIR`。

## 数据转换

A1 使用 LeRobot 格式训练数据。转换入口：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

示例：

```bash
bash process_data.sh RoboDojo stack_bowls arx_x5 103 joint
bash process_data.sh RoboDojo stack_bowls arx_x5 103 ee
```

## 训练

训练入口：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1
bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed}
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 103 ee 0,1,2,3 42
```

训练脚本会：

1. 检查并生成 LeRobot 数据。
2. 在 `A1/configs/datasets/xpolicylab_runtime.yaml` 写入当前数据路径。
3. 调用 `A1/launch_scripts/train_vla.py` 从 `${PRETRAIN_CHECKPOINT}` 继续训练。
4. 将 checkpoint 保存到 `policy/A1/checkpoints/${task_name}-a1-${action_type}-${expert_data_num}eps-seed${seed}-时间戳`。

## 评测 / 部署

评测入口：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH]
```

示例：

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 a1 a1
```

不传 `MODEL_PATH` 时，脚本会优先寻找当前任务最新训练出的 `*-unsharded` checkpoint；找不到时回退到：

```text
/mnt/pfs/pg4hw0/qiwei/models/a1-pretrain
```

手动指定 checkpoint：

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 lqw-a1 lqw-a1 \
  /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1/checkpoints/stack_bowls-a1-joint-103eps-seed42-20260515_204431/step16000-unsharded
```