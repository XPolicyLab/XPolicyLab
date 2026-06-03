# LingBot_VA

LingBot_VA 使用 LeRobot 视频数据、Wan latent 与动作统计进行后训练。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 数据处理

```bash
cd lingbot_va
# 转化出30维state & action数据
python dataset/transform.py --raw_dir <processed_data_task_dir> --repo_id <repo_id>
# --raw_dir: 原始数据路径, 指向RoboDojo数据集的路径
# --repo_id: 转化后lerobot数据的名称

# 添加指定额外信息
python dataset/add_action_config.py --dataset-root <lerobot_dataset_dir> --backup
# --dataset-root: 指向生成的lerobot数据集的所在路径, 例如.cache/huggingface_hub/lerobot/<repo_id>/ 
# --backup: 备份被修改的文件(meta/episodes.jsonl)

# wan22编码
python dataset/extract_wan_22_latents.py --dataset-root <lerobot_dataset_dir> --model-root <wan_model_dir>
# --dataset-root: 指向生成的lerobot数据集的所在路径, 例如.cache/huggingface_hub/lerobot/<repo_id>/ 
# --model-root: Wan2.2-TI2V-5B-Diffusers的路径

# 添加空编码用于初始化
python dataset/make_empty_embedding.py --model-root <wan_model_dir> --output <lerobot_dataset_dir>/empty_emb.pt
# --model-root: Wan2.2-TI2V-5B-Diffusers的路径
# <lerobot_dataset_dir>: 例如.cache/huggingface_hub/lerobot/<repo_id>/

# 计算stat
python dataset/compute_action_stat.py --dataset-root <lerobot_dataset_dir> --output <lerobot_dataset_dir>/action_norm_stats.json
# --dataset-root: 指向生成的lerobot数据集的所在路径, 例如.cache/huggingface_hub/lerobot/<repo_id>/ 
# <lerobot_dataset_dir>: 例如.cache/huggingface_hub/lerobot/<repo_id>/
```

默认 LeRobot 数据：`${XPOLICYLAB_LEROBOT_DATA_ROOT:-<robodojo_test>/data}/<repo_id>`（`arx_x5` → `RoboDojo_sim_arx-x5_v30`）。可用 `LINGBOT_VA_DATASET_PATH` 覆盖完整路径，或用 `LEROBOT_DATASET_REPO_ID` 覆盖 repo 名。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

Checkpoint：`checkpoints/<6-tuple>/`

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
