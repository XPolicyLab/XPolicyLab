# Pi_05

Pi_05 基于 openpi 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

```bash
cd openpi
python scripts/process_data.py <task_name> <env_cfg_type> <repo_id> <mode> [instruction]
bash scripts/compute_norm_stats.sh <config_name> <max_frames>
```

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：

```text
checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>/
```

| 变量 | 说明 |
|------|------|
| `OPENPI_TRAIN_CONFIG_NAME` | 默认 `pi05_base_aloha_full_sim_arx-x5_seed_0` |
| `OPENPI_LOCAL_CACHE_ROOT` | HF / JAX 缓存根目录 |

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls Pi_05_sim_arx-x5_seed_1 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> uv XPolicyLab
```
