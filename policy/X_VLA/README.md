# X_VLA

X_VLA 已按 XPolicyLab policy 方式封装，训练时通过 `xvla/train.py --output_dir` 保存 checkpoint。

## 数据准备

编辑或生成训练 metadata：

```text
policy/X_VLA/xvla/meta.json
```

如果 metadata 不在默认位置，可设置：

```bash
XVLA_META_PATH=/path/to/meta.json bash train.sh ...
```

预训练模型默认路径：

```text
/mnt/xspark-data/xspark_shared/model_weights/X-VLA-Pt
```

可通过 `XVLA_MODEL_PATH` 覆盖。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 ee 0 0,1,2,3
```

训练输出固定保存到：

```text
policy/X_VLA/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

X-VLA checkpoint 可能只保存 `config.json`、`model.safetensors`、`state.json`。如果部署需要 processor/tokenizer 等文件，请从 base checkpoint 目录复制缺失文件，不要覆盖训练产生的权重文件。

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> ee <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```

当前封装默认使用 `ee` 动作类型。
