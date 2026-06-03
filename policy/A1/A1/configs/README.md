# VLA Training Configuration System

This directory contains organized configuration files for VLA (Vision-Language-Action) training.

## Directory Structure

```
configs/
├── models/           # Model configuration files (action_head settings)
│   ├── libero_32d.yaml
│   └── vlabench_32d.yaml
├── datasets/         # Dataset configuration files
│   ├── libero_4_tasks.yaml
│   ├── libero_spatial.yaml
│   └── vlabench.yaml
├── experiments/      # Combined experiment configs (model + dataset)
│   ├── libero_simulation.yaml
│   └── vlabench.yaml
└── README.md
```

## Usage

### 1. New Format (Recommended)

Use combined experiment configs that reference separate model and dataset configs:

```bash
python launch_scripts/train_vla.py \
    --vla_config_path experiments/libero_simulation.yaml
```

Or:

```bash
python launch_scripts/train_vla.py \
    --vla_config_path experiments/vlabench.yaml
```

### 2. Old Format (Backward Compatible)

You can still use old-style configs in `launch_scripts/`:

```bash
python launch_scripts/train_vla.py \
    --vla_config_path vla_config_simulation.yaml
```

The system will automatically detect and load configs from:
1. `configs/<yaml_name>` (new location)
2. `launch_scripts/<yaml_name>` (legacy location)

## Creating Custom Configs

### Model Config

Create a file in `configs/models/`:

```yaml
# configs/models/my_model.yaml
action_head:
  fixed_action_dim: 32
  num_actions_chunk: 50
  action_tokens_mapping:
    left_end_effector: 7
    right_end_effector: 7
    mobile_base: 0
  use_left_eef: False
  use_mobile_base: False
```

### Dataset Config

Create a file in `configs/datasets/`:

```yaml
# configs/datasets/my_dataset.yaml
rlds:
  name: my_dataset_name
  path: data/my_data
  weight: 1.0
  action_proprio_normalization_type: bounds_q99
  image_augmentation: False

lerobot: []
droid: []
robochallenge: []

open-source-real-world:
  rlds:
  lerobot: []
  agibot:
```

### Experiment Config

Create a file in `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
model_config: models/my_model.yaml
dataset_config: datasets/my_dataset.yaml
```

## Configuration Priority

1. If `model_config` and/or `dataset_config` keys are present, their referenced configs are loaded and merged.
2. If these keys are absent, the config file is treated as a standalone config.
3. For backward compatibility, old configs in `launch_scripts/` are still supported.
