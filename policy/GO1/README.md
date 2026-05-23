# GO1

## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1

# joint
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint

# ee
bash process_data.sh RoboDojo stack_bowls arx_x5 5 ee
```

## 训练

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash train.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed}
```

不开 wandb：

```bash
conda activate lqw
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1

export REPORT_TO=tensorboard
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0,1,2,3 42
```

开 wandb：

```bash
conda activate lqw
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1

export REPORT_TO=wandb
export WANDB_PROJECT=go1
export WANDB_API_KEY=<your_wandb_api_key>
bash train.sh RoboDojo stack_bowls arx_x5 5 joint 0,1,2,3 42
```

## 推理

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash eval.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${policy_gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH] [env_gpu_id]
```

不指定 ckpt：

```bash
conda activate lqw
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 lqw lqw
```

指定 ckpt：

```bash
conda activate your_env
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 lqw lqw \

/mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/GO1/checkpoints/stack_bowls-go1-joint-5eps-seed42-20260521_152745/checkpoint-1000
```

如果要传 `env_gpu_id`，但不指定 `MODEL_PATH`：

```bash
bash eval.sh RoboDojo stack_bowls arx_x5 5 joint 0 42 lqw lqw "" 1
```
