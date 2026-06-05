# SmolVLA

SmolVLA 基于 LeRobot SmolVLA 接入 XPolicyLab。安装见 [INSTALLATION.md](INSTALLATION.md)。

## 训练

```bash
conda activate smolvla   # 或 install.sh 里 SMOVLA_CONDA_ENV 指定的名字
bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>
```

默认 LeRobot `dataset.repo_id` 与 task（`ckpt_name`）对应，与 v30 批量转换一致：

`RoboDojo_sim_<task>_v30`（例如 `build_tower` → `RoboDojo_sim_build_tower_v30`）

可用 `SMOVLA_REPO_ID` 覆盖；前缀/后缀：`SMOVLA_REPO_ID_PREFIX`、`SMOVLA_REPO_ID_SUFFIX`。

训练前会自动 `source /mnt/nfs/niantian/.bashrc`（`SMOVLA_BASHRC` 可改）。

Checkpoint：`checkpoints/<6-tuple>/`

### 批量训练（多 GPU + tmux）

每个 `ckpt_name`（task）绑定一张 GPU，在独立 tmux 里跑 `train.sh`，自动 `conda activate smolvla`：

```bash
# 方式 1：环境变量（task:gpu:seed，seed 可省略则用 SMOVLA_SEED）
TASK_GPU_MAP="stack_bowls:0:0,push_T:1:42,build_tower:2" bash train_batch.sh

# 方式 2：命令行
bash train_batch.sh stack_bowls:0:0 push_T:1:42 build_tower:2

# 查看 / 进入会话
tmux list-sessions | grep '^smolvla_'
tmux attach -t smolvla_stack_bowls
```

若 conda 环境名为 `smo_vla`：`export SMOVLA_CONDA_ENV=smo_vla`

## 部署

环境安装见 [INSTALLATION.md](INSTALLATION.md)。首次请执行 `bash install.sh`。

推荐分别执行 `setup_eval_policy_server.sh` 与 `setup_eval_env_client.sh` 便于查看 server 报错；同机也可使用 `eval.sh`：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo_sim_arx-x5_seed_0 arx_x5 3500 joint 0 <policy_gpu> <env_gpu> smolvla XPolicyLab
```
