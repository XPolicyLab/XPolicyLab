# TinyVLA

TinyVLA 基于 `tinyvla` 接入 XPolicyLab

## 环境安装

```bash
conda activate <your_env>
bash install.sh
```

## 训练

数据直接从 XPolicyLab 数据集中读取，无需处理

训练入口遵循 XPolicyLab 统一 7 参数：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

训练权重默认保存在 `TinyVLA/checkpoints` 下；子目录名采用上文“命名约定”中的 6 元组 `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`


## 评测

```bash
bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```
