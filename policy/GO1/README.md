# GO1

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：

- 训练产物子目录命名固定为 6 元组：
  `<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`
- `ckpt_name` 表示 checkpoint 标识；多任务共训建议显式写成 `cotrain`
- `task_name` 仅用于评测时指定仿真任务，训练阶段不再需要

## 数采

单任务本地转换仍保留 `task_name` 参数：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash process_data.sh ${bench_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

例子：

```bash
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

## 训练

命令（7 个参数，不含 `task_name`）：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

| 参数 | 含义 |
|---|---|
| `bench_name` | 数据集名称，如 `RoboDojo` |
| `ckpt_name` | checkpoint 标识，多任务共训建议填 `cotrain` |
| `env_cfg_type` | 环境配置 / 本体类型，如 `arx_x5` |
| `expert_data_num` | 训练轨迹数；使用共享多任务数据时可填与版本一致的占位值，如 `3500` |
| `action_type` | 动作类型，如 `joint` |
| `seed` | 随机种子 |
| `gpu_id` | GPU 编号列表，如 `0,1,2,3` |

### 默认路径（软路径）

不设置环境变量时，`train.sh` 默认使用：

| 变量 | 默认值 |
|---|---|
| `LEROBOT_DATA_PATH` | `/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21` |
| `MODEL_NAME_OR_PATH` | `<workspace>/models/GO-1` |
| `GO1_CFG_PATH` | `go1/configs/go1_sft_robodojo_shared.py` |
| `CTRL_FREQ` / `ACTION_CHUNK_SIZE` | `25` / `25` |
| `XDG_CACHE_HOME` | `/xspark-cache/.cache` |

只有显式设置对应环境变量时，才会覆盖上述默认路径。

### 多任务共训（推荐）

```bash
conda activate go1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

export CUDA_HOME=/usr/local/cuda
export REPORT_TO=tensorboard

bash train.sh RoboDojo cotrain arx_x5 3500 joint 42 0,1,2,3
```

训练产物目录示例：

```text
policy/GO1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42
```

如需指定其他数据或预训练权重：

```bash
export LEROBOT_DATA_PATH=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21
export MODEL_NAME_OR_PATH=/mnt/xspark-data/lqw/models/GO-1
bash train.sh RoboDojo cotrain arx_x5 3500 joint 42 0,1,2,3,4,5,6,7
```

## 推理

命令（11 个参数，与 `demo_policy/eval.sh` 对齐）：

```bash
cd /path/to/XPolicyLab/policy/GO1
bash eval.sh ${bench_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

| 参数 | 含义 |
|---|---|
| `task_name` | 仿真器中要跑的任务名（仅评测使用） |
| 其余参数 | 与总 README / demo_policy 一致 |

### 默认 checkpoint 查找

不设置 `MODEL_PATH` 时，`model.py` 会按 6 元组
`<bench_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>`
在 `policy/GO1/checkpoints/` 下自动查找最新 checkpoint。

### 示例

自动查找 checkpoint：

```bash
conda activate go1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/GO1

bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 go1 go1
```

指定 checkpoint 路径：

```bash
export MODEL_PATH=/mnt/xspark-data/lqw/XPolicyLab/policy/GO1/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-42/checkpoint-77484
bash eval.sh RoboDojo stack_bowls cotrain arx_x5 3500 joint 42 0 0 go1 go1
```

用 `cotrain` 权重评测单任务时，只需修改 `task_name`，`ckpt_name` 保持 `cotrain` 即可。
