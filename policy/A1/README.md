# A1

## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

## 训练

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash train.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

不开 wandb：

```bash
conda activate lqw-a1
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

export ENABLE_WANDB=false
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0,1,2,3
```

开 wandb：

```bash
conda activate lqw-a1
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

export ENABLE_WANDB=true
export WANDB_PROJECT=a1-xpolicylab
export WANDB_API_KEY=<your_wandb_api_key>
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0,1,2,3
```

多任务共训示例：

```bash
bash train.sh RoboDojo stack_bowls,pick_place cotrain arx_x5 50 joint 42 0,1,2,3
```

## 推理

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH] [env_gpu_id]
```

不指定 ckpt：

```bash
conda activate lqw-a1
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 lqw-a1 lqw-a1
```

指定 ckpt：

```bash
conda activate lqw-a1
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 lqw-a1 lqw-a1 \
  /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1/checkpoints/RoboDojo-stack_bowls-arx_x5-5-joint-42-20260523_160510/latest-unsharded
```

用 `cotrain` 权重评测单任务：

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 50 joint 42 0 lqw-a1 lqw-a1
```

如果要传 `env_gpu_id`，但不指定 `MODEL_PATH`：

```bash
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 lqw-a1 lqw-a1 "" 1
```
