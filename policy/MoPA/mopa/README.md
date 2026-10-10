# MoPA model package

This package contains the model, data pipeline, trainer and inference runtime used
by the [MoPA adapter](../README.md). Qwen3-VL-4B-Instruct supplies visual/language
features to separate arm and base query banks. The default action head uses
16 layers, width 768, 12 attention heads and 1024-wide state/action MLPs. Training
uses flow matching with uniform time sampling and 8 noise samples per example;
inference uses 4 Euler integration steps. See [configs/model.json](configs/model.json).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The generic RGB-array dataset, trainer and runtime can be installed independently.
The mobile HDF5 reader and host integration require XPolicyLab for its shared RGB
decoder. Supply local Qwen weights, tokenizer, processor and chat template with
`--base-vlm` or `MOPA_BASE_VLM`. Dependencies pin `transformers==4.57.0`; the
backbone uses SDPA attention.

## Data Processing

Generic episodes are NPZ files containing:

| Field | Shape and type |
| --- | --- |
| `state` | float32 `[T,D]` |
| `action` | float32 `[T,D]` |
| `instruction` | Nonempty scalar string |
| `image_0`, `image_1`, ... | uint8 RGB `[T,H,W,3]`, in metadata camera order |

The generic joint order is arm then gripper, with the left pair before the right
pair for a two-arm robot. A metadata file declares the layout:

```json
{
  "action_type": "joint",
  "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
  "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
  "image_size": [224, 224]
}
```

```bash
mopa-prepare --source /path/to/raw_npz_episodes \
  --metadata /path/to/metadata.json --output /path/to/dataset
```

Output contains validated NPZ episodes, `metadata.json` and
`dataset_statistics.json`. Generic normalization uses per-dimension q01/q99,
clips to `[-1,1]` and maps constant dimensions to zero. Padding repeats the final
action in an episode. Existing output directories are rejected.

Mobile conversion is available through the parent adapter's `process_data.sh`.
It packs 69 raw joint, pose and base-velocity values into a 75-value vector, with
manipulation `[0,56)` and mobility `[56,75)`. Source HDF5 paths remain in a lazy
manifest; images are decoded through XPolicyLab only when sampled. Keep source
files accessible. Mobile normalization uses exact min/max with an unclipped
affine map and a minimum span of `1e-6`; sampled q01/q99 values are diagnostic.
The reservoir holds at most 200000 real frames for state and action separately.

## Training

```bash
mopa-train --dataset /path/to/dataset --output /path/to/checkpoint \
  --base-vlm /path/to/Qwen3-VL-4B-Instruct --seed 0 --device cuda
```

Defaults are 100000 steps, batch size 8, backbone learning rate `1e-5` and head
learning rate `1e-4`. The default action slices target the mobile layout; generic
layouts require compatible ranges in `--config`. See `mopa-train --help`.
Distributed training is supported through `torchrun -m mopa.training.cli` or the
parent `train.sh`. Batch size is per process. Checkpoints are written by rank 0.
CPU execution uses float32 for the backbone.

The trainer writes resolved configuration and data statistics alongside model
weights. Optimizer/scheduler resume is not implemented. Nonempty output directories
are rejected. Reproducing a run requires the same data, assets, dependencies and
configuration; numerical results may vary across hardware.

## Evaluation

A checkpoint is a directory containing `config.json`, `model.pt` and
`dataset_statistics.json`. Keep all three together; weights and generated datasets
are excluded from version control.

```python
import numpy as np
from mopa.runtime import Policy

policy = Policy("/path/to/checkpoint", device="cuda")
actions = policy.predict(
    images=[rgb_views],  # RGB arrays in policy.cameras order.
    instructions=["Move the object to the target."],
    states=np.asarray([state_vector], dtype=np.float32),
)
assert actions.shape == (1, 32, policy.action_dim)
```

Actions are returned in their original physical units. `states` must use the
saved layout and camera arrays must already be RGB. If the Qwen assets move,
provide `Policy(..., base_vlm="/new/path")` with the same asset contents.
The checkpoint format is `xpl-mopa-query-dmot-v2`; this package uses
`mobile_m92uw` as its mobile layout tag. Older labels with the same explicit
mobile dimension metadata remain readable.
