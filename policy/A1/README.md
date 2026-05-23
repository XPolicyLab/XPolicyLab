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

# joint
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint

# ee
bash process_data.sh RoboDojo stack_bowls arx_x5 103 ee
```

## 训练

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed}
```

不开 wandb：

```bash
conda activate lqw-a1
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

export ENABLE_WANDB=false
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0,1,2,3 42
```

开 wandb：

```bash
conda activate your_env
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

export ENABLE_WANDB=true
export WANDB_PROJECT=a1-xpolicylab
export WANDB_API_KEY=<your_wandb_api_key>
export WANDB_API_KEY=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
bash train.sh RoboDojo stack_bowls arx_x5 103 ee 0,1,2,3 42
```

## 推理

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${policy_gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH] [env_gpu_id]
```

不指定 ckpt：

```bash
conda activate your_env
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 your_env your_env
```

指定 ckpt：

```bash
conda activate your_env
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1

bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 lqw-a1 lqw-a1 \
  /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/A1/checkpoints/stack_bowls-a1-joint-5eps-seed42-20260523_160510/latest-unsharded
```

如果要传 `env_gpu_id`，但不指定 `MODEL_PATH`：

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 103 joint 0 42 your_env your_env "" 1
```
