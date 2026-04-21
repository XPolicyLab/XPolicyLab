# 数据转化
## 转aloha格式
#暂时只提供了Xspark v1.0格式数据转化为aloha hdf5的脚本.

你一开始只有一个xspark_data/, 存放了hdf5文件, 其余路径是用于存放转化后的数据的, 你可以根据需要修改路径.
```bash
cd /path/to/XPolicyLab/
python scripts/transform_aloha_hdf5_format.py  /path/to/xspark_data/  /path/to/output_dir/ 
```
## 转tfds格式
脚本里面又些路径需要修改为你自己的路径, 你可以根据需要修改路径. 其中data_sample是转化后的数据存放路径, processed_dir是转化前的数据存放路径, 0.05是测试集占比, 0是GPU.
```bash
cd policy/openvla-oft/openvla_oft/
bash scripts/build_tfds_aloha.sh data_sample /path/to/output_dir/ path/to/processed_dir/ 0.05 0
```
# 训练
```bash
bash scripts/finetune.sh runs/model_sample/ data_sample 0,1,2,3
# runs/model_sample/: 替换为保存模型的路径.
# data_sample: 你的数据集名称.
# 0,1,2,3: 使用的GPU id.
```

# 评估
```bash
bash eval.sh task_name env_cfg expert_data_num action_type gpu_id seed policy_conda_env eval_env_conda_env CHECKPOINT_PATH
# task_name env_cfg expert_data_num seed 仿真中生效
# action_type: 本模型为 joint
# gpu_id: 使用的GPU id
# policy_conda_env: openvla-oft的conda 环境名
# eval_env_conda_env: XpolicyLab的conda 环境名
# CHECKPOINT_PATH是你模型存放的路径. 
# UNNORM_KEYS是你数据集的名称
```
