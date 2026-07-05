# Motus

Motus 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)；上游 LeRobot 训练细节见 [motus/README.md](motus/README.md)。

## 要点

- **LeRobot 格式**：可直接训练，指定 `repo_id` 与 `root`（`$LEROBOT_DATA_ROOT/<dataset>`）。
- **RoboTwin 原始数据**：需先经 `motus/data/robotwin2/` 转换流程。

## 环境变量

| 变量 | 说明 |
|------|------|
| `WAN_PATH` | WAN / VLM / Motus 权重根目录（传给 `--wan_path`） |
| `LEROBOT_DATA_ROOT` | LeRobot 数据集父目录 |

## T5 缓存示例

```bash
cd motus
export CUDA_VISIBLE_DEVICES=0

python data/lerobot/add_t5_cache_to_lerobot_dataset.py \
  --repo_id <repo_id> \
  --root "${LEROBOT_DATA_ROOT}/<dataset>" \
  --wan_path "${WAN_PATH}" \
  --device cuda \
  --t5_folder_name t5_embedding
```

## 数据准备

Motus 直接读取 LeRobot 数据集（`repo_id` + `root`），无本地 HDF5→zarr 转换。`process_data.sh` 负责解析并校验数据集是否存在（可选预生成 T5 cache）：

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
```

- 数据集默认取自训练配置 `motus/${MOTUS_TRAIN_CONFIG:-configs/lerobot_RoboDojo_sim.yaml}` 的 `dataset.params`。
- 用 `MOTUS_REPO_ID` / `MOTUS_DATASET_ROOT`（或 `LEROBOT_DATA_ROOT` 父目录）覆盖；`MOTUS_RUN_T5_CACHE=1`（需 `WAN_PATH`）可预生成 T5 embedding。

## 训练

标准入口（与其它 policy 一致的 6 参）：

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
```

内部计算 `ckpt_setting=<bench>-<ckpt>-<env>-<action>-<seed>`，把上游 `--checkpoint_dir` 指向 `checkpoints/<ckpt_setting>/`，权重最终落在
`checkpoints/<ckpt_setting>/<config_name>/<run_name>/checkpoint_step_<N>/pytorch_model/mp_rank_00_model_states.pt`。

评测时把 `ckpt_setting` 作为 `ckpt_name` 传入即可，**无需手工软链**——`model.py` 会在 `checkpoints/<ckpt_name>/` 下递归定位最新 `mp_rank_00_model_states.pt`。可用 `MOTUS_CHECKPOINT_PATH`（绝对目录/文件）或 `MOTUS_CKPT_SETTING`（`checkpoints/` 下的名字）覆盖。

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 joint 0 <policy_gpu> <env_gpu> motus XPolicyLab
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

