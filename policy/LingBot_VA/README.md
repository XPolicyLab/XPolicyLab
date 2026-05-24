# LingBot_VA

LingBot_VA 使用 LeRobot 视频数据、Wan2.2 latent 与动作统计进行后训练。

## 数据处理

在 `lingbot_va` 子目录中按以下顺序准备数据：

```bash
cd /mnt/nfs/niantian/robodojo_test/XPolicyLab/policy/LingBot_VA/lingbot_va
python dataset/transform.py --raw_dir <processed_data_task_dir> --repo_id <repo_id>
python scripts/add_action_config.py --dataset-root <lerobot_dataset_dir> --backup
python scripts/extract_wan_22_latents.py --dataset-root <lerobot_dataset_dir> --model-root <Wan2.2-TI2V-5B-Diffusers>
python scripts/make_empty_embedding.py --model-root <Wan2.2-TI2V-5B-Diffusers> --output <lerobot_dataset_dir>/empty_emb.pt
python scripts/compute_action_stat.py --dataset-root <lerobot_dataset_dir> --output <lerobot_dataset_dir>/action_norm_stats.json
```

默认训练脚本会把数据路径设为：

```text
policy/LingBot_VA/data/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>
```

可通过 `LINGBOT_VA_DATASET_PATH` 覆盖。

## 训练

```bash
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

示例：

```bash
bash train.sh RoboDojo stack_bowls arx_x5 50 joint 0 0,1,2,3
```

训练输出固定保存到：

```text
policy/LingBot_VA/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-<seed>
```

`train.sh` 会把上述目录作为 `--save-root` 传给 `lingbot_va/script/run_va_posttrain.sh`。配置名默认 `robotwin30_train`，可通过 `LINGBOT_VA_CONFIG_NAME` 覆盖。

## 评估

```bash
bash eval.sh <task_name> <env_cfg> <expert_data_num> <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env> <checkpoint_path>
```
