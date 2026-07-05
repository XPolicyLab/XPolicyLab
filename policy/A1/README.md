# A1

遵循 `XPolicyLab/README.md` 中的统一参数语义与命名约定：

- 处理后数据目录：`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`
- 训练产物目录：`checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>`（即 eval 的 `ckpt_name`）
- 做 ablation（如不同数据量）请换用不同 `ckpt_name`，并按需传可选 `expert_data_num`

## 数采

命令：

```bash
cd /path/to/XPolicyLab/policy/A1
bash process_data.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} [expert_data_num] [raw_task_dirs] [fps] [output_dir]
```

- `expert_data_num`：可选；留空 = 使用全部 episode
- `raw_task_dirs`：原始 HDF5 任务目录（`data/<bench_name>/` 下），逗号分隔可合并多任务；缺省为 `${ckpt_name}`

例子：

```bash
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1
# 全量数据
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
# ablation：只取 5 条，换一个 ckpt_name 区分
bash process_data.sh RoboDojo stack_bowls_n5 arx_x5 joint 5 stack_bowls
```

## 训练

命令（6 个参数）：

```bash
cd /path/to/XPolicyLab/policy/A1
bash train.sh ${bench_name} ${ckpt_name} ${env_cfg_type} ${action_type} ${seed} ${gpu_id}
```

例子：

```bash
conda activate a1
cd /mnt/xspark-data/lqw/XPolicyLab/policy/A1

export LEROBOT_DATA_PATH=/mnt/xspark-data/xspark_shared/lerobot/RoboDojo_sim_arx-x5_v21
export SEQ_LEN=1536
export GLOBAL_BATCH_SIZE=128
export DEVICE_TRAIN_MICROBATCH_SIZE=8
export NUM_WORKERS=4
export MAX_CROPS=3
export ENABLE_WANDB=true
export WANDB_PROJECT=A1
export WANDB_API_KEY=<your_wandb_api_key>

bash train.sh RoboDojo cotrain arx_x5 joint 42 0,1,2,3,4,5,6,7
```

训练产物目录示例：`policy/A1/checkpoints/RoboDojo-cotrain-arx_x5-joint-42`

## 推理

命令：

```bash
conda activate a1

cd /path/to/XPolicyLab/policy/A1
bash eval.sh ${bench_name} ${task_name} ${ckpt_name} ${env_cfg_type} ${action_type} ${seed} ${policy_gpu_id} ${env_gpu_id} ${policy_conda_env} ${eval_env_conda_env}
```

`ckpt_name` 即 `checkpoints/` 下的完整 run 目录名。例子：

```bash
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-42 arx_x5 joint 42 0 0 a1 a1
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

