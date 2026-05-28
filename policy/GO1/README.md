# GO1

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：

- 数据集子目录命名固定为 5 元组：
  `<dataset_name>-<task_name>-<env_cfg_type>-<expert_data_num>-<action_type>`
- 训练产物子目录命名固定为 6 元组：
  `<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`
- `task_name` 表示训练任务标识；多任务场景下可由 policy 自行扩展为逗号分隔，或使用一个统一的数据集标识名
- `ckpt_name` 表示 checkpoint 标识；单任务通常与 `task_name` 同名，多任务共训建议显式写成 `cotrain` 或其他固定名称

## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

## 训练

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash train.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

参数语义与总 README 保持一致：

- `dataset_name`: 数据集名称，如 `RoboDojo`
- `task_name`: 训练任务标识。单任务时填真实任务名；多任务共训时建议填一个统一标识，例如 `robodojo_sim_arx_x5_v21`
- `ckpt_name`: checkpoint 标识。单任务通常与 `task_name` 相同；多任务共训建议填 `cotrain`
- `env_cfg_type`: 环境配置 / 本体类型，如 `arx_x5`
- `expert_data_num`: 训练轨迹数；如果使用外部 LeRobot 数据集且目录已固定，可将其视为命名占位符，建议填与数据版本一致的固定值
- `action_type`: 动作类型，如 `joint`
- `seed`: 随机种子
- `gpu_id`: GPU 编号列表，如 `0,1,2,3`

### 默认单任务训练

不开 wandb：

```bash
conda activate lqw
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

export REPORT_TO=tensorboard
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0,1,2,3
```

开 wandb：

```bash
conda activate lqw
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

export REPORT_TO=wandb
export WANDB_PROJECT=go1
export WANDB_API_KEY=<your_wandb_api_key>
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0,1,2,3
```

### 默认多任务共训

```bash
bash train.sh RoboDojo stack_bowls,pick_place cotrain arx_x5 50 joint 42 0,1,2,3
```

### 直接读取外部 LeRobot 多任务数据集训练

当数据已经是现成的 LeRobot 目录时，不再要求其目录名符合 `policy/GO1/data/<5元组>` 规则；  
但命令行中的 `task_name` / `ckpt_name` 仍建议遵循总 README 的命名语义，用于统一训练产物命名。

以共享多任务数据集
`/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21`
为例，推荐：

- `dataset_name=RoboDojo`
- `task_name=robodojo_sim_arx_x5_v21`
- `ckpt_name=cotrain`
- `env_cfg_type=arx_x5`
- `expert_data_num=3500`
- `action_type=joint`

对应训练产物目录将命名为：

```text
policy/GO1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42-<timestamp>
```

直接运行命令：

```bash
conda activate go1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

export CUDA_HOME=/usr/local/cuda
export REPORT_TO=tensorboard
export LEROBOT_DATA_PATH=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21
export XDG_CACHE_HOME=/xspark-cache/.cache
export HF_DATASETS_CACHE=/xspark-cache/.cache/huggingface/datasets
export GO1_CFG_PATH=go1/configs/go1_sft_robodojo_shared.py
export MODEL_NAME_OR_PATH=/mnt/xspark-data/lqw/models/GO-1
export CTRL_FREQ=25 
export ACTION_CHUNK_SIZE=30

bash train.sh RoboDojo robodojo_sim_arx_x5_v21 cotrain arx_x5 3500 joint 1 0,1,2,3,4,5,6,7
```

说明：

- 不设置 `LEROBOT_DATA_PATH` 时，`train.sh` 仍保持原逻辑，默认读取或生成 `policy/GO1/data/<5元组>`
- 设置 `LEROBOT_DATA_PATH` 后，会直接使用该 LeRobot 根目录，不再触发本地 HDF5 转换
- `GO1_CFG_PATH=go1/configs/go1_sft_robodojo_shared.py` 用于匹配共享数据中的相机字段
- 共享数据的 `fps=25`，因此建议同时设置 `CTRL_FREQ=25` 和 `ACTION_CHUNK_SIZE=25`

## 推理

命令：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH] [env_gpu_id]
```

不指定 ckpt：

```bash
conda activate lqw
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 lqw lqw
```

指定 ckpt：

```bash
conda activate go1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 go1 go1 \
  /mnt/xspark-data/lqw/XPolicyLab/policy/GO1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42-20260525_153403/checkpoint-77484
```

用 `cotrain` 权重评测单任务：

```bash
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 50 joint 42 0 lqw lqw
```

如果要传 `env_gpu_id`，但不指定 `MODEL_PATH`：

```bash
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 0 lqw lqw "" 1
```
