# LingBot_VA

LingBot_VA 使用 LeRobot 视频数据、Wan latent 与动作统计进行后训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

```bash
cd lingbot_va
# 转化出30维state & action数据
python dataset/transform.py --raw_dir <processed_data_task_dir> --repo_id <repo_id>
# 添加指定额外信息
python scripts/add_action_config.py --dataset-root <lerobot_dataset_dir> --backup
# wan22编码
python scripts/extract_wan_22_latents.py --dataset-root <lerobot_dataset_dir> --model-root <wan_model_dir>
# 添加空编码用于初始化
python scripts/make_empty_embedding.py --model-root <wan_model_dir> --output <lerobot_dataset_dir>/empty_emb.pt
# 计算stat
python scripts/compute_action_stat.py --dataset-root <lerobot_dataset_dir> --output <lerobot_dataset_dir>/action_norm_stats.json
```

默认数据目录：`data/<5-tuple>/`，可用 `LINGBOT_VA_DATASET_PATH` 覆盖。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
