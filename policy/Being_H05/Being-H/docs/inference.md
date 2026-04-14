# Inference Guide

This guide covers the BeingHPolicy API for running inference with Being-H models.

## BeingHPolicy

```python
from BeingH.inference.beingh_policy import BeingHPolicy

policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",
    data_config_name="<config-name>",
    dataset_name="<dataset-name>",
    embodiment_tag="<robot-tag>",
    instruction_template="<prompt-template>",
    device="cuda:0",
)
```

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to checkpoint directory |
| `data_config_name` | Config name from `DATA_CONFIG_MAP` (see `configs/data_config.py`) |
| `dataset_name` | Dataset group name for loading metadata |
| `embodiment_tag` | Robot type identifier |
| `instruction_template` | Task prompt template with `{task_description}` placeholder |

## Cross-Embodiment Metadata Parameters

For cross-embodiment models trained on multiple robots/tasks, you need to specify which metadata variant to use for normalization:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metadata_variant` | `None` | Specific variant name (task or embodiment) |
| `stats_selection_mode` | `"auto"` | Selection strategy: `"auto"`, `"task"`, or `"embodiment"` |

**Selection Modes:**
- `"task"`: Use task-specific statistics (most precise)
- `"embodiment"`: Use embodiment-merged statistics (cross-task generalization)
- `"auto"`: Automatically select first available variant

See [Cross-Embodiment Metadata](#cross-embodiment-metadata) for details.

## Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `cuda` | Inference device |
| `num_inference_timesteps` | from ckpt | Flow matching ODE steps |
| `enable_rtc` | `True` | Enable Real-Time Chunking |
| `use_mpg` | from ckpt | Override MPG setting |
| `mpg_lambda` | from ckpt | MPG residual strength |

## get_action()

Pass observations as a dictionary. Keys must match your `DataConfig` class attributes:

```python
observations = {
    # Images - keys from VIDEO_KEYS
    "video.<camera_name>": image_array,       # (H, W, 3) uint8

    # States - keys from STATE_KEYS
    "state.<modality_name>": np.array([...]), # Shape defined in DataConfig

    # Language instruction
    "language.instruction": "<task-description>",
}

result = policy.get_action(observations)
```

## Output Format

```python
result = {
    "action.<modality_name>": [[...], ...],   # (chunk_size, dim)
    "action_unified": [...],                   # (chunk_size, 200) if RTC enabled
}
```

The output keys match your `DataConfig.ACTION_KEYS`. Actions are unnormalized using the loaded metadata statistics.

---

## Cross-Embodiment Metadata

Being-H saves **metadata** during training that is essential for inference. This metadata contains:
- **Normalization statistics** (mean, std, min, max, q01, q99) for each modality
- **Modality definitions** (absolute vs relative, rotation type, shape)
- **Embodiment tag** for the robot type

### Why Metadata Matters

Without correct metadata, the model cannot:
1. Normalize input states correctly
2. Unnormalize output actions to the correct scale
3. Select the right slots in the unified action space

### Metadata File Location

Metadata is saved during training at:
```
<checkpoint>/experiment_cfg/<dataset_name>_metadata.json
```

### Hierarchical Metadata Structure

For cross-embodiment training, Being-H uses a **3-level hierarchical metadata system**:

| Level | Key | Description |
|-------|-----|-------------|
| 0 | `<dataset_name>` | Default merged statistics |
| 1 | `<dataset_name>_variants[<task>]` | Task-specific statistics |
| 2 | `<dataset_name>_variants[<embodiment>]` | Embodiment-merged statistics |

### Selecting a Metadata Variant

When loading a cross-embodiment model, specify which variant to use:

```python
policy = BeingHPolicy(
    model_path="<path-to-checkpoint>",
    data_config_name="<config-name>",
    dataset_name="uni_posttrain",           # Cross-embodiment dataset
    metadata_variant="<task-or-embodiment>", # e.g., "libero_spatial" or "franka"
    stats_selection_mode="task",            # or "embodiment", "auto"
    # ... other parameters
)
```

**When to use each mode:**
- **Task-specific** (`stats_selection_mode="task"`): Best for evaluation on known tasks
- **Embodiment-merged** (`stats_selection_mode="embodiment"`): Best for new tasks on known robots
- **Auto** (`stats_selection_mode="auto"`): Uses first available variant

