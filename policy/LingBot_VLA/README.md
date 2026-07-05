# LingBot_VLA

LingBot_VLA 使用 LeRobot 与 yaml 配置训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## Norm 统计

默认 norm 配置：`lingbot_vla/configs/norm/robodojo_sim_arx_x5.yaml`（可按本机修改 `train_path` 等路径）。

```bash
cd lingbot_vla
export DATASET_NAME="RoboDojo_sample"

bash compute_norm_stat.sh configs/norm/${DATASET_NAME}_customized.yaml

python scripts/conver_norm_stat.py assets/norm_stats/${DATASET_NAME}_customized.json assets/norm_stats/${DATASET_NAME}.json
```

## 训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

| 变量 | 说明 |
|------|------|
| `XPOLICYLAB_LEROBOT_DATA_ROOT` / `LEROBOT_DATA_ROOT` | LeRobot 根目录，默认 `<robodojo_test>/data` |
| `LEROBOT_DATASET_REPO_ID` | 数据集 repo_id，默认 `RoboDojo_sim_arx-x5_v30`（`arx_x5`） |
| `LINGBOT_VLA_CONFIG_PATH` | 默认 `configs/vla/robodojo_sim_arx_x5.yaml` |
| `LINGBOT_VLA_DATA_PATH` | 数据集完整路径（默认 `${LEROBOT_DATA_ROOT}/${LEROBOT_DATASET_REPO_ID}`） |

Checkpoint：`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`（内部结构为 `checkpoints/global_step_*/hf_ckpt`）。评测时把该完整目录名作为 `eval.sh` 的 `ckpt_name` 传入。

Ablation（如数据量对比）用不同的 `ckpt_name` 区分 run；数据量在数据处理阶段用可选 `expert_data_num` 控制（留空 = 全部 episode）。

## 评估

部署需在 checkpoint 目录保留 `lingbotvla_cli.yaml`。

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

`ckpt_name` 直接是 `checkpoints/` 下完整的 run 目录名（历史 6-tuple 目录名可整体传入）。

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

