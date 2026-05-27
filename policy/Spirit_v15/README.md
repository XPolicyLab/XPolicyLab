# Spirit_v15

Spirit_v15 当前训练数据格式是 Spirit 自有目录结构，而不是 LeRobot。根目录训练脚本会先按 XPolicyLab/RoboDojo 原始数据转换，再启动 Spirit 训练。

## 数据格式

转换后的 Spirit 数据目录形如：

```text
<converted_data_root>/
  meta/task_info.json
  data/
    episode_000000/
      meta/episode_meta.json
      states/states.jsonl
      videos/
        head_camera_rgb.mp4
        left_camera_rgb.mp4
        right_camera_rgb.mp4
```

默认原始数据根目录：

```text
<robodojo_test>/data
```

可通过 `SPIRIT_RAW_DATA_ROOT` 覆盖。

## 数据处理

```bash
bash process_data.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type>
```

## 训练

先运行 `process_data.sh`，再运行：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例（`hekun/datasets/RoboDojo/sim_cloud` 原始数据）：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Spirit_v15

bash process_data.sh RoboDojo sweep_blocks arx_x5 50 ee
bash train.sh RoboDojo sweep_blocks arx_x5 50 ee 0 0,1,2,3
```

当 `dataset_name=RoboDojo` 且原始数据在 `.../RoboDojo/sim_cloud/` 下时，会自动使用匹配 pattern `sim_cloud.<task>.<env_cfg_type>`。

### Co-train（35 任务联合训练）

`sim_cloud` 下共 35 个任务，每个任务 100 条 episode（合计 3500）。使用 `ckpt_name=cotrain`，自动匹配 `sim_cloud.*.arx_x5`：

```bash
cd /vepfs-cnbje63de6fae220/niantian/RoboDojo_env/XPolicyLab/policy/Spirit_v15

# 数据处理：每个任务最多 100 条 episode（与 GR00T 的 3500 总量一致）
bash process_data.sh RoboDojo cotrain arx_x5 100 ee

# 训练
bash train.sh RoboDojo cotrain arx_x5 100 ee 0 0,1,2,3,4,5,6,7
```

输出目录：

```text
data/RoboDojo-cotrain-arx_x5-100-ee/          # 3500 episodes
checkpoints/RoboDojo-cotrain-arx_x5-100-ee-0/
```

`expert_data_num=100` 表示每个任务最多转换 100 条；也可写 `3500`（命名与 GR00T 对齐，实际每任务仍 capped 在 100）。

默认匹配 pattern：

```text
<dataset_name>.<ckpt_name>.<env_cfg_type>
```

默认转换输出：

```text
policy/Spirit_v15/data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

默认 checkpoint 输出：

```text
policy/Spirit_v15/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

常用覆盖变量：

```bash
SPIRIT_RAW_DATA_ROOT=/path/to/raw/data
SPIRIT_PATTERNS_CSV=RoboDojo.stack_bowls.arx_x5
SPIRIT_CONVERTED_DATA_ROOT=/path/to/converted/spirit/data
SPIRIT_PRETRAINED_PATH=/path/to/Spirit-v1.5
SPIRIT_SKIP_CONVERT=1
```

## 评估

```bash
bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```
