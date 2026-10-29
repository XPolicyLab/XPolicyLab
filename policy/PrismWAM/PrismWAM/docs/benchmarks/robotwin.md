# RoboTwin 2.0

Clean → Randomized: train on the 50 tasks in clean scenes, evaluate in clean and randomized scenes. 14-D absolute joint actions for both arms, from a head camera and two wrist cameras.

## Download

| | Hugging Face | Local path |
| --- | --- | --- |
| PrismWAM | [`robotwin/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robotwin) | `weights/robotwin/` |
| PrismWAM\* | [`robotwin/`](https://huggingface.co/THUSI-Lab/PrismWAM-Star/tree/main/robotwin) in PrismWAM-Star | `weights_star/robotwin/` |
| Training initializers (shared) | [`init/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/init) | `weights/init/` |

```bash
hf download THUSI-Lab/PrismWAM --include "robotwin/*" --local-dir weights        # evaluation
hf download THUSI-Lab/PrismWAM-Star --include "robotwin/*" --local-dir weights_star   # evaluation, PrismWAM*
hf download THUSI-Lab/PrismWAM --include "init/*" --local-dir weights   # training
```

Training also needs the [pretrained video model](../models.md#training-assets).

## Data

Collect `demo_clean` episodes with depth (`data_type.depth: true`) for all 50 tasks, plus `demo_clean_val`, under `data/robotwin/<task>/<phase>/`, and copy RoboTwin's instruction templates and object descriptions ([layout](../data.md#expected-layout)). Stage 2 decodes RGB with XPolicyLab's helper: keep the RoboTwin checkout that contains `XPolicyLab/` under `external/RoboTwin` (or set `ROBOTWIN_ROOT`). Then build `data/robotwin/action_stats.pt` and `data/robotwin/text_cache/` ([data preparation](../data.md)).

## Train

```bash
bash scripts/robotwin/stage1_depth.sh     # Stage 1: depth warm-up
bash scripts/robotwin/stage2_joint.sh     # Stage 2: joint RGB / depth / action
```

The three cameras are tiled into one 384 × 320 image, separately for depth and RGB.

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Depth input | current frame | current frame |
| RGB input | — | 5 frames, one every 4 steps (current + 4 past) |
| Robot state | — | 14-D joint state |
| Predicts | 8 future depth frames | 1 RGB goal frame (32 steps ahead), 8 future depth frames, 32-step action chunk |
| Steps · global batch | 15K · 256 | 15K · 256 |
| Recipe | [`stage1_depth.yaml`](../../configs/robotwin/stage1_depth.yaml) | [`stage2_joint.yaml`](../../configs/robotwin/stage2_joint.yaml) |

Validation uses 10 `demo_clean_val` episodes per task. Multi-node: set `NNODES`, `NODE_RANK` and `MASTER_ADDR` on every node ([GPUs and nodes](../training.md#gpus-and-nodes)). Export `step_015000.pt` with [`export_model.py`](../deployment.md#export-a-trained-model).

## Evaluate

Install the [XPolicyLab adapter](../xpolicylab.md) into RoboTwin's XPolicyLab. The policy reads
depth from every camera, which RoboTwin does not render by default: set `data_type.depth: true`
in `env_cfg/task_config/demo_clean.yml` and `demo_randomized.yml`. One task:

```bash
cd /path/to/RoboTwin/XPolicyLab/policy/PrismWAM
export PRISMWAM_ROOT=/path/to/PrismWAM PRISMWAM_RECIPE=robotwin
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin <task> "$PRISMWAM_ROOT/weights/robotwin" \
  aloha_agilex joint 0 0 0 /path/to/envs/prismwam <robotwin_env>
```

All 50 tasks, with RoboTwin's scheduler, once per setting:

```bash
cd /path/to/RoboTwin
for setting in demo_clean demo_randomized; do
  bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml \
    --policy-name PrismWAM --ckpt-name "$PRISMWAM_ROOT/weights/robotwin" \
    --env-cfg-type aloha_agilex --policy-conda-env /path/to/envs/prismwam \
    --eval-env-conda-env <robotwin_env> --task-config "$setting" \
    --test-num 100 --instruction-type unseen --seed 0
done
```

| Setting | Value |
| --- | --- |
| Scenes | `demo_clean` and `demo_randomized`, 50 tasks each |
| Episodes per task and setting | 100 (5,000 per setting) |
| Score | mean success of the two settings |
| Instructions | unseen templates (training uses seen ones) |
| Seeds | scene seeds from 100000, policy noise seed 0 |
| Inference | 32-step chunk, execute 24, 10 denoising steps, no conditioning noise |

The reported results drew instructions with a fixed seed (42) in an earlier RoboTwin
revision; the current evaluator draws them per scene, so scores can differ slightly.

### PrismWAM\*

PrismWAM\* runs through the same adapter and recipe, with its own model folder. One task:

```bash
cd /path/to/RoboTwin/XPolicyLab/policy/PrismWAM
export PRISMWAM_ROOT=/path/to/PrismWAM PRISMWAM_RECIPE=robotwin
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin <task> "$PRISMWAM_ROOT/weights_star/robotwin" \
  aloha_agilex joint 0 0 0 /path/to/envs/prismwam <robotwin_env>
```

All 50 tasks, once per setting:

```bash
cd /path/to/RoboTwin
for setting in demo_clean demo_randomized; do
  bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml \
    --policy-name PrismWAM --ckpt-name "$PRISMWAM_ROOT/weights_star/robotwin" \
    --env-cfg-type aloha_agilex --policy-conda-env /path/to/envs/prismwam \
    --eval-env-conda-env <robotwin_env> --task-config "$setting" \
    --test-num 100 --instruction-type unseen --seed 0
done
```

The policy and the simulator together use about 55 GB of GPU memory on one GPU. Besides the T5
text encoder, the first load fetches the CLIP image encoder of Wan2.1-I2V-14B-480P (about 5 GB)
to `weights/Wan-AI/Wan2.1-I2V-14B-480P` (or under `PRISMWAM_WEIGHTS_ROOT`).

## Results

| Model | Clean → Randomized (avg. success) |
| --- | ---: |
| PrismWAM | 70.57 % |
| PrismWAM\* | 83.67 % |

Comparisons with other models are on the [project page](https://prism-wam.github.io/).
