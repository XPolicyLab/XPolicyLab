# H_RDT XPolicyLab Adaptation

## 1. Extract Task Instructions

```bash
cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT/datasets/xpolicylab

python extract_task_instructions.py \
  /vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud \
  --env_cfg_type arx_x5
```

Output:

```text
H_RDT/datasets/xpolicylab/task_instructions.csv
```

## 2. Calculate q01/q99 Action Stats

```bash
cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT

source datasets/xpolicylab/setup_xpolicylab.sh

python datasets/xpolicylab/calc_stat.py \
  --data_root /vepfs-cnbje63de6fae220/hekun/datasets/RoboDojo/sim_cloud \
  --raw_dataset_name RoboDojo \
  --env_cfg_type arx_x5 \
  --action_type joint \
  --tasks all \
  --output_path datasets/xpolicylab/stats.json
```

Output:

```text
H_RDT/datasets/xpolicylab/stats.json
```

`XPolicyLabDataset` clips actions and states to `[q01, q99]`, then maps them to `[-1, 1]` during training.

## 3. Generate Language Embeddings

```bash
cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT

source /vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh
conda activate hrdt

export T5_MODEL_PATH=/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT/t5-v1_1-xxl
export HRDT_CONFIG_PATH=/vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT/configs/hrdt_finetune.yaml
export HRDT_LANG_GPU=0

python datasets/xpolicylab/encode_lang_batch.py
```

Output:

```text
H_RDT/datasets/xpolicylab/lang_embeddings/*.pt
```

## 4. Run the XPolicyLab Pipeline

```bash
cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT/H_RDT

source datasets/xpolicylab/setup_xpolicylab.sh
ENABLE_STATS_CALCULATION=true ./datasets/xpolicylab/run_xpolicylab_pipeline.sh
```

Optional:

```bash
ENABLE_TASK_INSTRUCTION_EXTRACTION=true ./datasets/xpolicylab/run_xpolicylab_pipeline.sh
ENABLE_LANGUAGE_ENCODING=true ./datasets/xpolicylab/run_xpolicylab_pipeline.sh
```

## 5. Top-level Train Script

`policy/H_RDT/train.sh` now uses the direct `xpolicylab` loader instead of converting data into the RobotWin2 layout.
Run the pipeline first so `stats.json` and language embeddings are already available.
By default, it trains all available XPolicyLab tasks together. The fourth
argument is the total number of training episodes, not the per-task count.
`train.sh` converts it to the per-task episode count before passing it to the
dataset loader. Use `3500` for the full RoboDojo cotrain set, which becomes
`100` episodes per task across 35 tasks.
It starts finetuning from the H-RDT human pretrain backbone by default:
`H_RDT/checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin`.

```bash
cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT

bash train.sh RoboDojo cotrain arx_x5 3500 joint 0 0
```

To override the pretrain backbone, pass a different path as the eighth argument.

## 6. Co-train Checkpoint and Task Embeddings

`train.sh` trains one shared co-train checkpoint over all XPolicyLab tasks.
During evaluation, each rollout still targets one concrete task, so the policy
server must receive that task's language embedding.

Example:

```text
checkpoint_path = checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0
task_name = stack_bowls
lang_embedding_path = H_RDT/datasets/xpolicylab/lang_embeddings/stack_bowls.pt
```

To evaluate another task with the same co-train checkpoint, keep
`checkpoint_path` unchanged and replace both `task_name` and
`lang_embedding_path`:

```text
task_name = sweep_blocks
lang_embedding_path = H_RDT/datasets/xpolicylab/lang_embeddings/sweep_blocks.pt
```

In short: the checkpoint is shared across tasks, while the language embedding
selects the current evaluation task.

tensorboard --logdir checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/logs --host 0.0.0.0 --port 6006

## Eval

先把权重复制过来：demo_env/XPolicyLab/policy/H_RDT/checkpoints/RoboDojo-cotrain-arx_x5-3500-joint-0/checkpoint-100000

cd /vepfs-cnbje63de6fae220/mobile/chengy/xpolicy/demo_env/XPolicyLab/policy/H_RDT

bash [eval.sh](http://eval.sh) RoboDojo stack_bowls cotrain arx_x5 3500 joint 0 0 0 hrdt hrdt