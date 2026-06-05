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

## XPolicyLab 训练 / 评测

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
bash eval.sh ...
```

Checkpoint：

```text
checkpoints/<6-tuple>/
```

## 评测（XPolicyLab）

环境安装见 [INSTALLATION.md](INSTALLATION.md)。手动部署推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh`（便于查看 server 报错）。

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-3500-joint-0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> motus XPolicyLab
```

Pi_0 / Pi_0_Fast 需先执行 `Pi_05/install.sh`，server 环境填 `uv`。

