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

## 评测（XPolicyLab）

环境安装见 [INSTALLATION.md](INSTALLATION.md)。手动部署推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh`（便于查看 server 报错）。

```bash
bash eval.sh RoboDojo stack_bowls XVLA_sim_arx-x5 arx_x5 3500 ee 0 <policy_gpu> <env_gpu> XVLA XPolicyLab
```

Pi_0 / Pi_0_Fast 需先执行 `Pi_05/install.sh`，server 环境填 `uv`。

