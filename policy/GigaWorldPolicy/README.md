# GigaWorldPolicy

GigaWorldPolicy 基于 `giga_world_policy` 接入 XPolicyLab。底层 `giga-train` 会把日志和模型保存到 config 中的 `project_dir`，因此根目录 `train.sh` 会先生成一份 XPolicyLab 专用 JSON config，并将 `project_dir` 固定到统一 checkpoint 目录。

## 数据处理

GigaWorldPolicy 训练前需要准备 LeRobot 数据、动作归一化统计和 T5 embedding。可使用内层脚本：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/GigaWorldPolicy/giga_world_policy
bash process_data.sh <lerobot_data_path> <Wan2.2-TI2V-5B-Diffusers_path>
```

该脚本会生成：

```text
giga_world_policy/norm_stats_delta.json
<lerobot_data_path>/t5_embedding
```

默认训练数据路径为：

```text
policy/GigaWorldPolicy/data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

如需使用已有 LeRobot 数据目录，可覆盖：

```bash
GIGAWORLD_DATA_DIR=/path/to/lerobot/data bash train.sh ...
```

## 训练

训练入口遵循 XPolicyLab 统一的 7 参数约定：

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 ee 0 0,1,2,3
```

训练产物默认保存到：

```text
policy/GigaWorldPolicy/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

根目录 `train.sh` 会生成：

```text
policy/GigaWorldPolicy/checkpoints/<6元组>/xpolicylab_train_config.json
```

并覆盖底层 config 的关键字段：

- `project_dir`: 指向上述 checkpoint 目录。
- `dataloaders.train.data_or_config[0].data_path`: 指向训练数据目录。
- `dataloaders.train.transform.norm_path`: 指向 norm stats。
- `models.pretrained`: 指向 Wan2.2 预训练权重。
- `models.view_dir`: 指向 checkpoint 目录。
- `launch.gpu_ids`: 来自第 7 个参数 `gpu_id`。

常用覆盖变量：

```bash
GIGAWORLD_DATA_DIR=/path/to/lerobot/data
GIGAWORLD_NORM_PATH=/path/to/norm_stats_delta.json
GIGAWORLD_PRETRAINED_PATH=/path/to/Wan2.2-TI2V-5B-Diffusers
GIGAWORLD_BASE_CONFIG=/path/to/base_config.json
```

## Debug Dry Run

不启动训练、只生成并校验 config：

```bash
GIGAWORLD_DRY_RUN=1 bash train.sh RoboDojo debug arx_x5 1 ee 0 0
```

## Eval Debug

默认 debug 配置在 `deploy.yml` 中：

```text
eval_env: debug
checkpoint_num: checkpoint_epoch_4_step_100000
transformer_subdir: transformer_ema
load_model: false
```

模型必须由 `eval.sh` 传入的 `ckpt_name` 索引，不在 `deploy.yml` 中写 `model_path`。`checkpoint_num` 用来指定 `checkpoints/<ckpt_name>/models/` 下的第一个具体训练 checkpoint。当前使用 symlink：

```text
policy/GigaWorldPolicy/checkpoints/RoboDojo_sim_arx_seed_0/models/checkpoint_epoch_4_step_100000 -> /mnt/xspark-data/final_ckpt/GigaWorldPolicy/RoboDojo_sim_arx_seed_0
```

`load_model: false` 用于 XPolicyLab debug 通路调试：会解析并校验 checkpoint 路径，返回维度正确的零动作，避免加载 10GB 权重导致依赖或显存阻塞。完整模型推理时，将 `load_model` 改为 `true`，并配置 `t5_embedding_pkl`。

debug eval 示例：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/GigaWorldPolicy
bash eval.sh RoboDojo debug_task RoboDojo_sim_arx_seed_0 arx_x5 1 joint 0 0 0 gigaworld-policy gigaworld-policy
```

## 原始项目说明

原始 GigaWorld-Policy 使用方式见内层文档：

```text
policy/GigaWorldPolicy/giga_world_policy/Readme.md
```
