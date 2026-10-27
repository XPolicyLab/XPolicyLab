# Phoenix-MWA · RoboTwin 2.0

[Project](https://openphoenix-team.github.io/Phoenix-pages/) · [Original code](https://github.com/OpenPhoenix-Team/Phoenix-MWA/tree/16bc6e307ee71b3797e03a4f072a69340f078dab) · [Weights](https://huggingface.co/AnyverseDynamics/Phoenix-RoboTwin)

Supports `RoboTwin / arx_x5 / joint` (aloha-agilex). The released model uses **230K-step initialization + 35K-step posttraining**. `runtime/` contains the model and its dependencies; `training/` contains data preparation and training entry points.

```text
model.py / deploy.py / deploy.yml   XPolicyLab interface
runtime/                           model, configuration and checkpoints
training/                          data preparation and training
tests/                            regression tests
```

## Installation

Use a separate environment for the [RoboTwin simulator](https://robotwin-platform.github.io/doc/usage/install.html).

```bash
cd XPolicyLab/policy/Phoenix_MWA
bash install.sh phoenix-mwa
conda activate phoenix-mwa
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k
```

The downloader retrieves `policy.pt`, DINO/STVAE, WanVAE, T5 and the tokenizer. Sources and checksums are in `runtime/weights_manifest.json`; add `--verify-only` to verify an existing download.

## Data Processing

Training uses the upstream **LeRobot v2.1** export: 50 Hz, native 14D actions/state and `cam_high`, `cam_left_wrist`, `cam_right_wrist` images. The reader selects the first 50 episodes of each 550-episode task block (2,500 clean demonstrations). This layout differs from arbitrary official-converter exports; retain the prepared export's `meta/`, `data/` and `videos/` directories.

```bash
bash process_data.sh RoboTwin cotrain arx_x5 joint \
  --data /path/to/robotwin-v21 --text-cache /path/to/robotwin-text
```

To create the task-indexed T5 cache, add `--build-text-cache --weights "$PWD/checkpoints/phoenix-robotwin-35k" --device cuda:0` and use a new cache directory.

## Training

The [Joint96 pretraining checkpoint (256013 steps)](https://www.modelscope.cn/api/v1/models/ssuuuuu/phoenix-wam/repo?Revision=master&FilePath=phoenix-mwa-pretrain-joint96-step_256013.pt) is available for training initialization. `--pretrain-only` downloads this file separately and verifies its size and SHA256; add `--verify-only` to check an existing copy. The default download command continues to retrieve the 35K evaluation policy and its dependencies.

```bash
bash download_checkpoint.sh checkpoints/phoenix-pretrain-256013 --pretrain-only
bash download_checkpoint.sh checkpoints/phoenix-robotwin-35k --training-backbones
bash train.sh RoboTwin cotrain arx_x5 joint 42 0,1,2,3,4,5,6,7 \
  --init "$PWD/checkpoints/phoenix-pretrain-256013/phoenix-mwa-pretrain-joint96-step_256013.pt" \
  --weights "$PWD/checkpoints/phoenix-robotwin-35k" \
  --data /path/to/robotwin-v21 --text-cache /path/to/robotwin-text \
  --nnodes 1 --nproc 8 --batch-size 8 --max-steps 35000
```

Add `--dry-run` to check resources and configuration. The original schedule uses 8 nodes × 8 GPUs (global batch 512); the example uses one node (global batch 64). Outputs go to `checkpoints/RoboTwin-cotrain-arx_x5-joint-42/`.

## Evaluation

```bash
bash eval.sh RoboTwin adjust_bottle phoenix-robotwin-35k arx_x5 joint 0 \
  0 0 phoenix-mwa robotwin
```

For all 50 tasks, use the [official multi-task evaluator](https://robotwin-platform.github.io/doc/usage/xpolicylab.html) with `--policy-name Phoenix_MWA --ckpt-name phoenix-robotwin-35k`, 100 episodes per task and unseen instructions. Select `demo_clean` for C2C or `demo_randomized` for C2R.

Inference preserves three RGB views, source resize/center-crop, z-score normalization and `repeat9_full_stvae`. Defaults in `deploy.yml`: `replan_steps: 32`, `num_inference_steps: 10`, `policy_file: policy.pt`, `device: cuda:0`, `request_timeout_s: 600`. The model runs in BF16.

## Results

| C2C | C2R | Average | Episodes |
| ---: | ---: | ---: | ---: |
| 86.54% | 72.42% | 79.48% | 10,000 |

The supplied result index reports 4,327/5,000 C2C successes and 3,621/5,000 C2R successes. These scores use the [original frozen-scene protocol](https://github.com/OpenPhoenix-Team/Phoenix-MWA/blob/16bc6e307ee71b3797e03a4f072a69340f078dab/docs/ROBOTWIN.md). The XPolicyLab adapter separately passed 10 plain-RGB and 10 encoded-observation GPU debug episodes.
