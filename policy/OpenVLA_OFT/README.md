# OpenVLA_OFT

OpenVLA_OFT 使用 ALOHA/TFDS 格式数据。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据转换

```bash
# 在 XPolicyLab 根目录
python scripts/transform_aloha_hdf5_format.py <xspark_data_dir> <aloha_output_dir>

cd policy/OpenVLA_OFT/openvla_oft
TFDS_DATA_DIR=<tensorflow_datasets_dir> \
  bash scripts/build_tfds_aloha.sh <data_sample> <aloha_output_dir> <processed_dir> 0.05 0
```

默认 TFDS 名：`aloha_<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>`
（即 `aloha_` + 训练 run 目录名，与 eval 侧 `aloha_<ckpt_name>` 的默认推导一致）。

可用 `OPENVLA_TFDS_DATASET_NAME` 覆盖；覆盖时需在 `deploy.yml` 里同步设置
`tfds_dataset_name`（显式覆盖优先于默认推导）。

## 训练

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`，
该目录名整体即 eval 侧的 `ckpt_name`。数据量 ablation 改用不同 `ckpt_name`
（如 `stack_bowls_50ep`）区分，episode 数在 TFDS 数据转换阶段控制。

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 <policy_gpu> <env_gpu> openvla_oft XPolicyLab
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

