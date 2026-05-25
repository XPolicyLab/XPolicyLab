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

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 ee 0 0,1,2,3
```

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
