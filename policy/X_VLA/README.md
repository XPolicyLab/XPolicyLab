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
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`，
该目录名整体即 eval 侧的 `ckpt_name`。数据量 ablation 改用不同 `ckpt_name`
（如 `stack_bowls_50ep`）区分，episode 数在 `meta.json` 数据准备阶段控制。

训练结束后 `train.sh` 会自动把 base 模型的 processor/tokenizer 文件补进各
`ckpt-<step>/`（不覆盖 `config.json` / `model.safetensors`），保证 eval 可直接加载。

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls XVLA_sim_arx-x5 arx_x5 ee 0 <policy_gpu> <env_gpu> XVLA XPolicyLab
```

### Evaluation environment (`EVAL_ENV_TYPE`)

Set the `EVAL_ENV_TYPE` environment variable before running `eval.sh` or `setup_eval_env_client.sh` (default: **sim** when unset):

| `EVAL_ENV_TYPE` | Mode |
|---|---|
| unset or `sim` | RoboDojo simulation |
| `debug` | Offline shape/IO validation (`debug_env_client.py`) |
| `real` | Not available in open-source release |

```bash
export EVAL_ENV_TYPE=debug
bash eval.sh ...
```

