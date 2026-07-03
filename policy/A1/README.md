# A1

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：
## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

## 训练

命令（7 个参数）：

```bash
cd /path/to/XPolicyLab/policy/A1
bash train.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

例子：

```bash
conda activate a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export LEROBOT_DATA_PATH=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21
export SEQ_LEN=1536
export GLOBAL_BATCH_SIZE=128
export DEVICE_TRAIN_MICROBATCH_SIZE=8
export NUM_WORKERS=4
export MAX_CROPS=3
export ENABLE_WANDB=true
export WANDB_PROJECT=A1
export WANDB_API_KEY=<your_wandb_api_key>

bash train.sh RoboDojo cotrain arx_x5 3500 joint 42 0,1,2,3,4,5,6,7
```

## 推理

命令：

```bash
conda activate a1

cd /path/to/XPolicyLab/policy/A1
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

例子：

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 a1 a1
```