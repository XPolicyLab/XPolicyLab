# RoboDojo

[RoboDojo](https://robodojo-benchmark.com/LeaderBoard): 54 tasks in five categories (generalization, precision, long-horizon, memory, open-ended) on the ARX X5 dual-arm robot. We train on the 35 training tasks; 14-D absolute joint actions for both arms, from a head camera and two wrist cameras.

## Download

| | Hugging Face | Local path |
| --- | --- | --- |
| Model | [`robodojo/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo) | `weights/robodojo/` |
| Training initializers (shared) | [`init/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/init) | `weights/init/` |

```bash
hf download THUSI-Lab/PrismWAM --include "robodojo/*" --local-dir weights        # evaluation
hf download THUSI-Lab/PrismWAM --include "init/*" --local-dir weights   # training
```

Training also needs the [pretrained video model](../models.md#training-assets).

## Data

Put the ARX X5 recordings under `data/robodojo/<task>/arx_x5/data/` and a RoboDojo checkout under `external/RoboDojo` (or set `ROBODOJO_ROOT`). Then build the episode indexes, statistics and text cache ([data preparation](../data.md)). Training uses every episode of the 35 tasks, validation episodes included.

## Train

```bash
bash scripts/robodojo/stage1_depth.sh     # Stage 1: depth warm-up
bash scripts/robodojo/stage2_joint.sh     # Stage 2: joint training with sparse RGB memory
```

The three cameras are tiled into one 384 × 320 image, separately for depth and RGB.

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Depth input | current frame | current frame |
| RGB input | — | sparse RGB memory: first frame, frame 32 steps ago, current frame |
| Robot state | — | 14-D joint state |
| Predicts | 8 future depth frames | 1 RGB goal frame (32 steps ahead), 8 future depth frames, 32-step action chunk |
| Steps · global batch | 15K · 1024 | 40K · 1024 |
| Recipe | [`stage1_depth.yaml`](../../configs/robodojo/stage1_depth.yaml) | [`stage2_joint.yaml`](../../configs/robodojo/stage2_joint.yaml) |

Sparse RGB memory is for the memory and long-horizon tasks: Stage 2 sees the first frame, the frame 32 steps ago and the current frame instead of the five most recent ones, each noised with probability 0.2 during training.

Multi-node: set `NNODES`, `NODE_RANK` and `MASTER_ADDR` on every node ([GPUs and nodes](../training.md#gpus-and-nodes)). Export `step_040000.pt` with [`export_model.py`](../deployment.md#export-a-trained-model).

## Evaluate

The reported results use RoboDojo
[`ee67a14`](https://github.com/RoboDojo-Benchmark/RoboDojo/tree/ee67a1468510da7624a089164402359f2afc72c8)
with its assets at revision `c70d27e`; later revisions change render synchronization and
articulation resets and score differently. RoboDojo renders no depth by default:
[`patches/robodojo_depth.patch`](../../patches/robodojo_depth.patch) adds a
`distance_to_image_plane` capture to the three cameras and turns on `approximate_depth`
(uint16 millimetres, as in the training data). Check out that revision with its submodules,
install it with the [upstream guide](https://robodojo-benchmark.com/doc/usage/install-and-download/),
then pin XPolicyLab, apply the patch, fetch the assets and add the adapter:

```bash
git clone --recurse-submodules https://github.com/RoboDojo-Benchmark/RoboDojo.git
git -C RoboDojo checkout ee67a1468510da7624a089164402359f2afc72c8
git -C RoboDojo submodule update --init --recursive
```

```bash
export PRISMWAM_ROOT=/path/to/PrismWAM
cd /path/to/RoboDojo
git -C XPolicyLab fetch origin ddf5dcfff5e089a323da1d22891909b72864647e
git -C XPolicyLab checkout ddf5dcfff5e089a323da1d22891909b72864647e
git apply --check "$PRISMWAM_ROOT/patches/robodojo_depth.patch"
git apply "$PRISMWAM_ROOT/patches/robodojo_depth.patch"
HF_REVISION=c70d27efddca73b784441942bc3b8d1b361c684e bash scripts/init_assets.sh
python "$PRISMWAM_ROOT/integrations/xpolicylab/install_adapter.py" --xpolicylab-root XPolicyLab
```

One task:

```bash
cd /path/to/RoboDojo/XPolicyLab/policy/PrismWAM
export PRISMWAM_RECIPE=robodojo PRISMWAM_DEPTH_KEY=approximate_depth
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PRISMWAM_ROOT/weights/robodojo" \
  arx_x5 joint 0 0 0 /path/to/envs/prismwam robodojo
```

All 54 tasks with RoboDojo's benchmark runner, then the official aggregation:

```bash
cd /path/to/RoboDojo
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/PrismWAM \
  --ckpt "$PRISMWAM_ROOT/weights/robodojo" --policy-env /path/to/envs/prismwam \
  --action-type joint --eval-num native --seed 0
bash scripts/robodojo.sh summarize
```

| Setting | Value |
| --- | --- |
| Tasks | all 54 tasks, 2,100 episodes, seed 0 |
| Score | official RoboDojo aggregation: mean over the five categories |
| Inference | 32-step chunk, execute 16, 10 denoising steps, no conditioning noise |

## Results

| Model | Overall score | Success rate |
| --- | ---: | ---: |
| PrismWAM | 25.10 | 18.55 % |
| PrismWAM\* | 36.23 | 29.30 % |

Comparisons with other models are on the [project page](https://prism-wam.github.io/).
