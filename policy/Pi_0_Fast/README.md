# Pi_0_Fast

Pi_0_Fast 基于 openpi 的 `Pi0FASTConfig` 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

统一 5 参数入口（`expert_data_num` 可选，留空 = 全部 episode）：

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

产物 repo_id / 数据集 tag：`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`。

如需 norm stats，在 `openpi` 子目录运行：

```bash
cd openpi
bash scripts/compute_norm_stats.sh <config_name> <max_frames>
```

## 训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

Checkpoint：

```text
checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/
```

该目录名整体即 eval 侧的 `ckpt_name`。数据量 ablation 改用不同 `ckpt_name`
（如 `stack_bowls_50ep`）区分，episode 数在数据转换阶段控制。

| 变量 | 说明 |
|------|------|
| `OPENPI_TRAIN_CONFIG_NAME` | 默认 `pi0_fast_aloha_full_sim_arx-x5_seed_0` |
| `OPENPI_LOCAL_CACHE_ROOT` | HF / JAX 缓存根目录 |

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_arx_seed_0 arx_x5 joint 0 <policy_gpu> <env_gpu> uv XPolicyLab
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

