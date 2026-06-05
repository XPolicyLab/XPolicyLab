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

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls XVLA_sim_arx-x5 arx_x5 3500 ee 0 <policy_gpu> <env_gpu> XVLA XPolicyLab
```
