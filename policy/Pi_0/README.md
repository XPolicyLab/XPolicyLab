# 使用方法
## 训练

### step1 数据格式转化
``` bash 
cd pi0/openpi/
# task_name对应你采集任务的名称
# env_cfg对应你采集任务的环境配置文件路径
# repo_id对应你希望转化后的lerobot数据集的名称
# mode可以选image / video
# instruction可以不输入, 如果不输入则会默认为“Do your job.”, 该参数生效前提是数据里面的instrusctions为None, 如果数据里面的instructions已经有了具体的指令内容, 则该参数不会生效
python scripts/process_data.py ${task_name} ${env_cfg_type} ${repo_id} ${mode} ${instruction}
# python scripts/process_data.py fold_clothes dual_y1 fold_clothes_v1 fold_clothes_v1 "Fold the clothes and put them in the box."
```
    
### step2 计算norm stat
``` bash
cd pi0/openpi/
bash scripts/compute_norm_stats.py --config_name ${config_name} --max_frames ${max_frames}
# bash scripts/compute_norm_stats.sh pi05_full_base 10000
```

### step3 训练
``` bash
cd pi0/openpi/
bash finetune.sh --config_name ${config_name} --repo_id ${repo_id}
```
### 4. 评估
``` bash
cd path/to/pi0
# MODEL_PATH: 模型权重的路径
# TRAIN_CONFIG_NAME: 训练配置的名称
# REPO_ID: 数据集的名称, 用来指定norm stat
bash eval.sh ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${gpu_id} ${seed} ${policy_conda_env} ${eval_env_conda_env} ${MODEL_PATH} ${TRAIN_CONFIG_NAME} ${REPO_ID}
```