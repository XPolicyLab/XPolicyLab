# Adding a New Embodiment to DreamZero

How to take a LeRobot v2 dataset for a new robot, convert it to GEAR format, define its modality config, and train a DreamZero policy.

Throughout this guide, replace `<EMBODIMENT>` with your robot's name (e.g. `myrobot`, `franka`, `aloha`).

---

## Overview

```
Step 1  Convert LeRobot v2 dataset → GEAR metadata
Step 2  Register the embodiment tag
Step 3  Add modality config + transforms to base YAML
Step 4  Create a dataset YAML
Step 5  Create a training script
Step 6  Train
```

---

## Step 1: Convert Dataset to GEAR Format

The converter reads a LeRobot v2 dataset and generates the metadata files DreamZero needs. It does **not** modify your parquet files or videos — it only writes to `meta/`.

### Expected input structure

```
your_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam0/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.cam1/
│           └── ...
└── meta/
    └── info.json          # must contain: features, total_episodes, fps
```

### Run the converter

```bash
python scripts/data/convert_lerobot_to_gear.py \
    --dataset-path /path/to/your_dataset \
    --embodiment-tag <EMBODIMENT> \
    --state-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \
    --action-keys '{"joint_pos": [0, 6], "gripper_pos": [6, 7]}' \
    --relative-action-keys joint_pos gripper_pos \
    --task-key annotation.task
```

`--state-keys` and `--action-keys` tell the converter how to split a packed vector column into named sub-keys. The JSON maps sub-key name → `[start_index, end_index]`. Omit these flags to let the converter auto-detect.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | *(required)* | Path to the LeRobot v2 dataset |
| `--output-path` | *(in-place)* | Write to a different directory instead of in-place |
| `--embodiment-tag` | `xdof` | Tag for `meta/embodiment.json`; must match the key you use in Step 3 |
| `--state-keys` | *(auto)* | JSON: sub-key name → `[start, end]` index range |
| `--action-keys` | *(auto)* | JSON: sub-key name → `[start, end]` index range |
| `--relative-action-keys` | *(none)* | Sub-key names to compute relative action stats for |
| `--task-key` | *(auto)* | Column name for language/task annotations |
| `--fps` | *(from info.json)* | Override dataset FPS |
| `--action-horizon` | `24` | Horizon for relative stats computation |
| `--force` | `false` | Overwrite existing metadata files |

### Generated files

The converter creates these under `meta/`:

| File | Contents |
|---|---|
| `modality.json` | Maps state, action, video, and annotation keys with index ranges and dtypes |
| `embodiment.json` | `{"embodiment_tag": "<EMBODIMENT>"}` |
| `stats.json` | Per-feature statistics (mean, std, min, max, q01, q99) |
| `relative_stats_dreamzero.json` | Relative action statistics (action − reference state) |
| `tasks.jsonl` | Unique task descriptions |
| `episodes.jsonl` | Per-episode metadata (index, tasks, length) |

---

## Step 2: Register the Embodiment Tag

1. Add to the enum in `groot/vla/data/schema/embodiment_tags.py`:

```python
class EmbodimentTag(str, Enum):
    ...
    MY_ROBOT = "<EMBODIMENT>"
```

2. Add to `VALID_EMBODIMENT_TAGS` in `scripts/data/convert_lerobot_to_gear.py` (if you want the converter to accept the tag without `--force`):

```python
VALID_EMBODIMENT_TAGS = [
    ...,
    "<EMBODIMENT>",
]
```

---

## Step 3: Add Modality Config and Transforms

Edit `groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml`.

### 3a. Understanding modality

**Modality** connects your dataset columns to the training pipeline. The `modality.json` from Step 1 contains entries like:

```json
{
  "state": {
    "joint_pos": {"original_key": "observation.state", "start": 0, "end": 6},
    "gripper_pos": {"original_key": "observation.state", "start": 6, "end": 7}
  },
  "action": {
    "joint_pos": {"original_key": "action", "start": 0, "end": 6},
    "gripper_pos": {"original_key": "action", "start": 6, "end": 7}
  },
  "video": {
    "cam0": {"original_key": "observation.images.cam0"}
  },
  "annotation": {
    "task": {"original_key": "annotation.task"}
  }
}
```

The YAML config must reference **exactly these key names** with type prefixes:

| Modality | YAML key format | Example |
|---|---|---|
| State | `state.<name>` | `state.joint_pos` |
| Action | `action.<name>` | `action.joint_pos` |
| Video | `video.<name>` | `video.cam0` |
| Language | `annotation.<name>` | `annotation.task` |

If the YAML keys don't match `modality.json`, training will fail with missing-key errors.

### 3b. Add `modality_config_<EMBODIMENT>`

```yaml
modality_config_<EMBODIMENT>:
  video:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    eval_delta_indices: [0]
    modality_keys:                        # one entry per camera, matching modality.json
      - video.cam0
      - video.cam1
      - video.cam2
  state:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:                        # matching modality.json state keys
      - state.joint_pos
      - state.gripper_pos
  action:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    modality_keys:                        # matching modality.json action keys
      - action.joint_pos
      - action.gripper_pos
  language:
    _target_: groot.vla.data.dataset.ModalityConfig
    delta_indices: [0]
    modality_keys:
      - annotation.task
```

**`delta_indices` explained:**

- **Video** — frame offsets to sample (25 entries = 25 frames from the trajectory).
- **State / Language** — `[0]` = current timestep only.
- **Action** — future offsets (24 entries = 24-step action chunk).

Adjust these to match your `num_frames` and `action_horizon` training settings.

### 3c. Add `transform_<EMBODIMENT>`

```yaml
transform_<EMBODIMENT>:
  _target_: groot.vla.data.transform.ComposedModalityTransform
  transforms:
    # Video
    - <<: *totensor_cfg
      apply_to: ${modality_config_<EMBODIMENT>.video.modality_keys}
    - <<: *crop_cfg
      apply_to: ${modality_config_<EMBODIMENT>.video.modality_keys}
    - <<: *resize_cfg
      apply_to: ${modality_config_<EMBODIMENT>.video.modality_keys}
    - <<: *color_jitter_cfg
      apply_to: ${modality_config_<EMBODIMENT>.video.modality_keys}
    - <<: *to_numpy_cfg
      apply_to: ${modality_config_<EMBODIMENT>.video.modality_keys}

    # State
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_<EMBODIMENT>.state.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_<EMBODIMENT>.state.modality_keys}
      normalization_modes:
        state.joint_pos: q99              # every state key needs a normalization mode
        state.gripper_pos: q99

    # Action
    - _target_: groot.vla.data.transform.StateActionToTensor
      apply_to: ${modality_config_<EMBODIMENT>.action.modality_keys}
    - _target_: groot.vla.data.transform.StateActionTransform
      apply_to: ${modality_config_<EMBODIMENT>.action.modality_keys}
      normalization_modes:
        action.joint_pos: q99             # every action key needs a normalization mode
        action.gripper_pos: q99

    # Concat
    - _target_: groot.vla.data.transform.ConcatTransform
      video_concat_order: ${modality_config_<EMBODIMENT>.video.modality_keys}
      state_concat_order: ${modality_config_<EMBODIMENT>.state.modality_keys}
      action_concat_order: ${modality_config_<EMBODIMENT>.action.modality_keys}

    # Model-specific (required, don't change)
    - ${model_specific_transform}
```

Every state and action key **must** appear in `normalization_modes`. The strategy is typically `q99`.

### 3d. Register in the global maps

Add your embodiment to each of these four maps (at the bottom of the base YAML):

```yaml
modality_configs:
  ...
  <EMBODIMENT>: ${modality_config_<EMBODIMENT>}

transforms:
  ...
  <EMBODIMENT>: ${transform_<EMBODIMENT>}

metadata_versions:
  ...
  <EMBODIMENT>: '0221'

fps:
  ...
  <EMBODIMENT>: 30          # set to your dataset's FPS
```

---

## Step 4: Create a Dataset YAML

Create `groot/vla/configs/data/dreamzero/<EMBODIMENT>_relative.yaml`:

```yaml
# @package _global_

defaults:
  - dreamzero/base_48_wan_fine_aug_relative
  - _self_

max_state_dim: 64
use_global_metadata: false
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - joint_pos                    # sub-key names (without state./action. prefix)
  - gripper_pos                  # that should use relative actions
max_chunk_size: 5
dataset_shard_sampling_rate: 0.1
mixture_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotMixtureDataset.from_mixture_spec
single_dataset_cls: groot.vla.data.dataset.lerobot_sharded.ShardedLeRobotSubLangSingleActionChunkDatasetDROID

<EMBODIMENT>_data_root: ???     # set via CLI or env var

train_dataset:
  _target_: ${mixture_dataset_cls}
  _convert_: object
  mixture_spec:
    - dataset_path:
        <EMBODIMENT>:            # must match key in modality_configs/transforms
          - ${<EMBODIMENT>_data_root}
      dataset_weight: 1.0
      distribute_weights: true

  dataset_class: ${single_dataset_cls}
  all_modality_configs: ${modality_configs}
  all_transforms: ${transforms}
  metadata_versions: ${metadata_versions}
  fps: ${fps}
  dataset_kwargs:
    video_backend: decord
    use_global_metadata: ${use_global_metadata}
    max_chunk_size: ${max_chunk_size}
    relative_action: ${relative_action}
    relative_action_keys: ${relative_action_keys}
    relative_action_per_horizon: ${relative_action_per_horizon}
  mixture_kwargs:
    training: true
    balance_dataset_weights: false
    seed: 42
    shard_sampling_rate: ${dataset_shard_sampling_rate}
```

The critical things to get right:

- `<EMBODIMENT>` in `mixture_spec.dataset_path` must match the key in `modality_configs` and `transforms`.
- `relative_action_keys` lists the sub-key names (without `state.`/`action.` prefix) that exist in **both** state and action modalities.

---

## Step 5: Create a Training Script

Create `scripts/train/<EMBODIMENT>_training.sh`:

```bash
#!/bin/bash
export HYDRA_FULL_ERROR=1

# ============ CONFIGURATION ============
DATA_ROOT=${DATA_ROOT:?"Set DATA_ROOT to your GEAR-converted dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_<EMBODIMENT>_lora"}

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =======================================

# Auto-download weights if missing
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATA_ROOT"
    exit 1
fi
if [ ! -f "$DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: meta/embodiment.json missing — run convert_lerobot_to_gear.py first"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone \
    groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/<EMBODIMENT>_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    <EMBODIMENT>_data_root=$DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
```

### Key parameters to adjust per embodiment

| Parameter | Default | When to change |
|---|---|---|
| `num_views` | `3` | Number of cameras your robot has |
| `action_horizon` | `24` | Must match the number of action `delta_indices` |
| `num_frames` | `33` | Must be `len(video delta_indices) + num_frame_per_block * (blocks - 1)` |
| `image_resolution_width` | `320` | Match your camera resolution (or desired resize) |
| `image_resolution_height` | `176` | Match your camera resolution (or desired resize) |
| `max_steps` | `100000` | Scale with dataset size |
| `per_device_train_batch_size` | `4` | Adjust for GPU memory |

---

## Step 6: Train

### Download the pretrained checkpoint

The training scripts load from a pretrained DreamZero checkpoint for LoRA fine-tuning. Download [DreamZero-AgiBot](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot) (~45GB) to `./checkpoints/DreamZero-AgiBot`:

```bash
git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot ./checkpoints/DreamZero-AgiBot
```

Or with the Hugging Face CLI:

```bash
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot
```

### Launch training

```bash
DATA_ROOT=/path/to/your_dataset bash scripts/train/<EMBODIMENT>_training.sh

# With overrides:
DATA_ROOT=/path/to/your_dataset OUTPUT_DIR=./checkpoints/run1 NUM_GPUS=4 \
    bash scripts/train/<EMBODIMENT>_training.sh
```

---

## Pre-Training Checklist

- [ ] `meta/embodiment.json` exists and has the correct tag
- [ ] `meta/modality.json` state/action/video/annotation keys are populated
- [ ] `meta/stats.json` and `meta/relative_stats_dreamzero.json` exist
- [ ] `meta/tasks.jsonl` and `meta/episodes.jsonl` exist
- [ ] Embodiment tag in `embodiment.json` matches the key in `modality_configs` / `transforms` / `metadata_versions` / `fps`
- [ ] YAML `modality_keys` match `modality.json` keys exactly (with `state.`/`action.`/`video.`/`annotation.` prefix)
- [ ] Every state and action key appears in `normalization_modes` in the transform block
- [ ] `relative_action_keys` are sub-key names that exist in both state and action
- [ ] Wan2.1-I2V-14B-480P and umt5-xxl weights are available
- [ ] DreamZero-AgiBot checkpoint is downloaded to `./checkpoints/DreamZero-AgiBot`

---

## Quick Reference: Existing Embodiments

| Embodiment | Data Config | Layout |
|---|---|---|
| `oxe_droid` | `droid_relative.yaml` | 3 cameras, joint_position + gripper_position |
| `agibot` | `agibot_relative.yaml` | 3 cameras, 6 state keys, 7 action keys |
| `yam` | `yam_relative.yaml` | 3 cameras (top/left/right), bimanual left/right joint_pos + gripper_pos |

Use these as concrete examples when building your own config.
