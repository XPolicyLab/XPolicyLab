# starVLA
适配starvla到xpolicylab
## 安装
```bash
conda create -n XPolicyLab python=3.10 -y
conda activate XPolicyLab
cd XPolicyLab
pip install -e .
cd policy/starVLA
bash install.sh
```

## 数据转换

命令（5 个参数）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash process_data.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab
conda activate XPolicyLab
bash policy/starVLA/process_data.sh RoboDojo stack_bowls arx_x5 50 joint
```

## 训练

命令（7 个参数，不含 `task_name`）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash train.sh ${dataset_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```


### 默认多任务训练

```bash
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA/source_starvla
PYTHONPATH=$PWD python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml ../xpolicy_oft_vla.yaml

conda activate XPolicyLab
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0,1,2,3,4,5,6,7
```

训练输出目录：

```text
policy/starVLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
```

## 推理


命令（11 个参数）：

```bash
cd /path/to/XPolicyLab/policy/starVLA
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

不指定 ckpt：在policy/starvla下新建checkpoints文件夹，然后把final_ckpt中的RoboDojo-cotrain-arx_x5-3500-joint-0文件夹拷贝到这个目录下
预权重是放在了/mnt/xspark-data/xspark_shared/model_weights/Qwen3-VL-4B-Instruct，
需要复制一个公开的权重到这里
policy/starVLA/source_starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct
```bash
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA
conda activate XPolicyLab

bash eval.sh RoboDojo stack cotrain arx_x5 3500 joint 0 0 1 XPolicyLab XPolicyLab
```

默认会按 6 元组查找训练目录，并从 `checkpoints/` 下读取唯一的权重文件：

```text
policy/starVLA/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/checkpoints/<checkpoint>.pt
```

也兼容旧的 `policy/starVLA/results/Checkpoints/<6元组>/checkpoints/` 目录。
如果目录下有多个权重文件，请删除多余文件或通过 `STARVLA_CKPT_PATH` 显式指定。

指定 ckpt：

```bash
conda activate XPolicyLab
cd /cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA

export STARVLA_CKPT_PATH=/cpfs_infra/user/wangkaixuan/chengy/demo_env/XPolicyLab/policy/starVLA/checkpoints/RoboDojo-stack_bowls-arx_x5-50-joint-0/final_model/pytorch_model.pt

bash eval.sh RoboDojo stack stack_bowls arx_x5 50 joint 0 0 1 XPolicyLab XPolicyLab
```
