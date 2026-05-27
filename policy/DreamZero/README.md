# DreamZero

本目录是 DreamZero 在 XPolicyLab 中的接入层，目标是按统一规范完成数据转换、训练与评测。DreamZero 原项目源码位于 `dreamzero/`，适配代码尽量放在当前目录顶层，不修改策略主体。

## 环境准备

DreamZero 依赖较重，请先按 `dreamzero/README.md` 准备 Python 3.11、CUDA、PyTorch、flash-attn、Wan2.1/umt5 等环境和权重。当前适配默认使用预训练权重：

```bash
/mnt/pfs/pg4hw0/qiwei/models/checkpoints/DreamZero-AgiBot
```

安装当前 policy 与 XPolicyLab：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero
bash install.sh
```

如 Wan 或 tokenizer 权重不在默认位置，可在训练前设置：

```bash
export WAN_CKPT_DIR=/path/to/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/path/to/umt5-xxl
export DREAMZERO_PRETRAINED_MODEL_PATH=/mnt/pfs/pg4hw0/qiwei/models/checkpoints/DreamZero-AgiBot
```

其中 `TOKENIZER_DIR` 必须是已经下载好的 `google/umt5-xxl` 本地目录；如果目录不存在，Transformers 会把绝对路径误当成 HuggingFace repo id 并报 `HFValidationError`。

## 数据转换

输入为 XPolicyLab 标准 HDF5 数据：

```text
demo_env/data/${dataset_name}/${task_name}/${env_cfg_type}/data/episode_*.hdf5
```

转换命令：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero
bash process_data.sh ${dataset_name} ${task_name} ${env_cfg_type} ${expert_data_num} ${action_type}
```

示例：

```bash
bash process_data.sh RoboDojo stack_bowls arx_x5 5 joint
```

输出目录遵循 XPolicyLab 5 元组命名：

```text
policy/DreamZero/data/${dataset_name}-${task_name}-${env_cfg_type}-${expert_data_num}-${action_type}
```

转换脚本会生成 DreamZero 所需的 LeRobot v2 数据和 `meta/modality.json`、`meta/embodiment.json`、`meta/stats.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`、`meta/relative_stats_dreamzero.json`。图像会统一 resize 到 `320x240` 并从 BGR 转为 RGB。这里不做 7DoF 到 6DoF 转换。

## 动作空间适配

DreamZero-AgiBot 使用 20 维 state 和 22 维 action：

```text
state = left_arm(7) + right_arm(7) + left_effector(1) + right_effector(1) + head(2) + waist_pitch(1) + waist_lift(1)
action = state(20) + robot_velocity(2)
```

XPolicyLab 的真实 arm/ee 维度由 `env_cfg_type` 决定。适配层会将真实 arm/ee 写入对应 DreamZero 字段；缺失的 head、waist、robot_velocity 用 0 填充。评测时只取 DreamZero 输出中的左右臂和左右末端执行器，再按 XPolicyLab 的真实维度裁剪或补零。

## 训练

命令格式严格遵循 XPolicyLab：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero
bash train.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${gpu_id}
```

示例：

```bash
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 5 joint 42 2,3,4,5
```

训练产物输出到 6 元组目录：

```text
policy/DreamZero/checkpoints/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}
```

常用可覆盖变量：

```bash
export DREAMZERO_MAX_STEPS=50000
export DREAMZERO_SAVE_STEPS=20
export DREAMZERO_NUM_GPUS=4
export DREAMZERO_PER_DEVICE_BATCH_SIZE=2
export DREAMZERO_REPORT_TO=tensorboard
export WANDB_PROJECT=dreamzero
export WAN_CKPT_DIR=/mnt/pfs/pg4hw0/qiwei/models/checkpoints/checkpoints/Wan2.1-I2V-14B-480P
export TOKENIZER_DIR=/mnt/pfs/pg4hw0/qiwei/models/checkpoints/checkpoints/umt5-xxl
export DREAMZERO_PRETRAINED_MODEL_PATH=/mnt/pfs/pg4hw0/qiwei/models/checkpoints/DreamZero-AgiBot
```

## 评测

命令格式遵循 XPolicyLab 11 参数，可额外传入 `MODEL_PATH` 指定 checkpoint：

```bash
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero
bash eval.sh ${dataset_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${expert_data_num} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env} [MODEL_PATH]
```

使用默认 6 元组 checkpoint 或 DreamZero-AgiBot 预训练权重：

```bash
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 50 joint 42 0 0 dreamzero XPolicyLab
```

显式指定 checkpoint：

```bash
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 50 joint 42 0 0 dreamzero XPolicyLab \
  /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero/checkpoints/RoboDojo-stack_bowls-arx_x5-5-joint-42/checkpoint-60
```

`deploy.yml` 中 `eval_env` 控制评测环境，支持 `debug`、`sim`、`real`，切换环境时不需要修改 `eval.sh`。
DreamZero 首次推理会触发 Wan 模块 warmup/compile，耗时可能超过 XPolicyLab 默认 30 秒 client timeout；适配层默认将 client timeout 扩展到 1800 秒，可用 `DREAMZERO_MODEL_CLIENT_TIMEOUT` 覆盖。
原项目推理服务会放宽 `torch._dynamo` 的 recompile/cache limit；XPolicyLab 适配层已同步该设置，默认 `DREAMZERO_DYNAMO_RECOMPILE_LIMIT=800`，用于避免 Wan 自回归推理中的 shape 变化触发默认 8 次重编译上限。

## 推理配置

`deploy.yml` 中的重要参数：

```yaml
model_path: null
pretrained_model_path: /mnt/pfs/pg4hw0/qiwei/models/checkpoints/DreamZero-AgiBot
tokenizer_path: null
action_horizon: 24
video_history: 4
inference_method: lazy_joint_forward_causal
```

`model_path` 优先级最高；可传具体 checkpoint 目录，也可传包含 `checkpoint-*` 的训练输出目录。为空时会先按 6 元组 checkpoint 查找训练产物，再回退到 `pretrained_model_path`。`tokenizer_path` 为空时会优先使用环境变量 `TOKENIZER_DIR`，其次尝试 `dreamzero/checkpoints/umt5-xxl`，用于覆盖预训练配置里的 tokenizer 路径。`inference_method` 可选 `forward`、`lazy_joint_forward`、`lazy_joint_forward_causal`，默认使用原项目闭环评测路径 `lazy_joint_forward_causal`。

## 快速流程

```bash
conda activate dreamzero
cd /mnt/pfs/pg4hw0/qiwei/demo_env/XPolicyLab/policy/DreamZero

bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint
bash train.sh RoboDojo stack_bowls stack_bowls arx_x5 50 joint 42 0,1,2,3
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 50 joint 42 0 0 dreamzero XPolicyLab
```

## 注意事项

- DreamZero 本体对依赖、显存和 checkpoint 路径要求较高，本适配不负责创建环境。
- `process_data.py` 会覆盖同名转换输出目录，重新转换前请确认旧数据不再需要。
- 如果目标机器人和 AgiBot 预训练动作空间差异较大，接口可以跑通，但策略效果需要通过训练和闭环评测确认。
