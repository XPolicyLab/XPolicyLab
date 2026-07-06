# ABot Installation And Dataset Preparation

This document records the setup that works in this repository today, plus the shortest path for training on a single LeRobot dataset.

## Environment Setup

```bash
git clone https://github.com/amap-cvlab/ABot-Manipulation.git
git clone https://github.com/facebookresearch/vggt.git
cd ABot-Manipulation

conda create -n ABot python=3.10 -y
conda activate ABot

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints/abot-train-cu121.txt

pip install -e ../vggt
pip install -e .
```

If the environment has already drifted to a different torch or CUDA stack, reset it first:

```bash
pip uninstall -y torch torchvision torchaudio flash-attn
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt -c constraints/abot-train-cu121.txt
```

## Required Model Weights

ABot training expects two model paths:

1. A Qwen3-VL-4B-Instruct-Action model directory.
2. An ABot-M0 pretrain checkpoint.

Example paths that are already used in this workspace:

```bash
BASE_VLM=/path/to/model_weights/Qwen3-VL-4B-Instruct-Action
PRETRAIN_CKPT=/path/to/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt
```

For the current SimStackBowls setup, the safe default is to load only `qwen_vl_interface` from the pretrain checkpoint:

```bash
RELOAD_MODULES=qwen_vl_interface
```

This avoids an action-head shape mismatch between the current config and the released pretrain checkpoint.

## SimStackBowls Quick Start

The repository now contains a complete SimStackBowls training path:

1. Dataset normalization: `examples/SimStackBowls/prepare_sim_stack_bowls_for_abot.py`
2. Config: `examples/SimStackBowls/train_files/ABot_sim_stack_bowls.yaml`
3. Launcher: `examples/SimStackBowls/train_files/run_sim_stack_bowls_train.sh`

Minimal command:

```bash
conda activate ABot

BASE_VLM=/path/to/model_weights/Qwen3-VL-4B-Instruct-Action \
PRETRAIN_CKPT=/path/to/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt \
RELOAD_MODULES=qwen_vl_interface \
NUM_GPUS=8 \
bash examples/SimStackBowls/train_files/run_sim_stack_bowls_train.sh
```

Useful overrides:

```bash
BATCH_SIZE=2
NUM_WORKERS=2
MAX_TRAIN_STEPS=1000
SAVE_INTERVAL=200
```

## Training On Your Own LeRobot Dataset

### Quick Path Assumption

The quick path in this repository assumes your dataset can be trained with the existing `robotwin` data config.

That means your dataset should expose these ABot-side keys through `meta/modality.json`:

1. `video.cam_high`
2. `video.cam_left_wrist`
3. `video.cam_right_wrist`
4. `state.left_joints`
5. `state.right_joints`
6. `state.left_gripper`
7. `state.right_gripper`
8. `action.left_joints`
9. `action.right_joints`
10. `action.left_gripper`
11. `action.right_gripper`
12. `annotation.human.action.task_description`

If your embodiment is not a dual-arm joint-plus-gripper layout compatible with `robotwin`, you should not force it into this quick path. In that case you need a new robot type config in `ABot/dataloader/gr00t_lerobot/data_config.py` and a matching mixture registration.

### Required LeRobot Layout

Your dataset root should look like this:

```text
your_dataset/
  data/
    chunk-000/
      episode_000000.parquet
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
  videos/
    chunk-000/
      observation.images.cam_high/
      observation.images.cam_left_wrist/
      observation.images.cam_right_wrist/
```

Notes:

1. `info.json` must contain `data_path` and `video_path`.
2. ABot expects video-backed LeRobot data. If your dataset only has `images/` and no `videos/`, encode the per-episode frames into `videos/chunk-xxx/.../episode_xxxxxx.mp4` first.
3. `stats_gr00t.json` is not required ahead of time. ABot can compute it on first load.

### Normalize Metadata For ABot

Use the helper script added in this repository:

```bash
python scripts/prepare_lerobot_for_abot.py \
  --dataset-dir /path/to/lerobot_dataset \
  --cam-high-key observation.images.cam_high \
  --cam-left-key observation.images.cam_left_wrist \
  --cam-right-key observation.images.cam_right_wrist
```

If your dataset is single-task and you want to overwrite all task text with one instruction:

```bash
python scripts/prepare_lerobot_for_abot.py \
  --dataset-dir /path/to/lerobot_dataset \
  --task "stack the bowls"
```

What the helper does:

1. Verifies the required `meta/` files exist.
2. Verifies the required `videos/chunk-000/...` directories exist.
3. Rewrites `meta/modality.json` so ABot sees `video.cam_high`, `video.cam_left_wrist`, `video.cam_right_wrist`, and `annotation.human.action.task_description`.
4. Checks that `state` and `action` already expose `left_joints`, `right_joints`, `left_gripper`, and `right_gripper` slices.

What the helper does not do:

1. It does not invent `state/action` slice definitions for a new embodiment.
2. It does not encode videos for you if your dataset is image-only.
3. It does not register a brand new robot type.

### Train A Single Custom Dataset

The SimStackBowls launcher has been generalized so you can reuse it for a single custom LeRobot dataset.

Example:

```bash
conda activate ABot

DATA_ROOT=/path/to/lerobot_parent_dir \
DATASET_REPO=my_dataset \
DATA_MIX=my_dataset_mix \
ROBOT_TYPE=robotwin \
PREPARE_SCRIPT=scripts/prepare_lerobot_for_abot.py \
PREPARE_TASK_TEXT= \
BASE_VLM=/path/to/model_weights/Qwen3-VL-4B-Instruct-Action \
PRETRAIN_CKPT=/path/to/model_weights/ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt \
RELOAD_MODULES=qwen_vl_interface \
NUM_GPUS=4 \
BATCH_SIZE=2 \
bash examples/SimStackBowls/train_files/run_sim_stack_bowls_train.sh
```

Meaning of the dataset arguments:

1. `DATA_ROOT` is the parent directory.
2. `DATASET_REPO` is the dataset folder name under `DATA_ROOT`.
3. `DATA_MIX` is the temporary single-dataset mixture name registered through environment variables.
4. `ROBOT_TYPE` defaults to `robotwin` for the quick path.
5. `PREPARE_TASK_TEXT=` with an empty value keeps your original per-task metadata instead of rewriting everything to one sentence.

## Troubleshooting

### `FlowmatchingActionHead` size mismatch when loading checkpoint

Cause:

The pretrain checkpoint action head does not match the current training config.

Fix:

```bash
RELOAD_MODULES=qwen_vl_interface
```

### CUDA out of memory during the first training step

Reduce:

```bash
NUM_GPUS=1
BATCH_SIZE=1
NUM_WORKERS=0
```

Then scale back up gradually.