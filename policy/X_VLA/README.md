# X_VLA

X_VLA 已封装为 XPolicyLab policy。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据准备

编辑 `xvla/meta.json`，或：

```bash
XVLA_META_PATH=/path/to/meta.json bash train.sh ...
```

预训练模型通过 `XVLA_MODEL_PATH` 指定（HF id 或本地目录）。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`

若 checkpoint 缺少 processor/tokenizer，从 base 模型目录复制，勿覆盖 `model.safetensors`。

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> ee <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
