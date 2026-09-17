# MoPA

[MoPA: Coordinated Mobile Manipulation via Subsystem-Specific Perception Alignment](https://mopa-policy.github.io/)
models perception and actions through subsystem-specific visual queries and action experts.
This implementation provides an arm policy with one set of **8 manipulation queries**,
the base branch disabled, and **4-step joint action chunks**.
Inputs are RGB images, language instructions, and arm/gripper states; base states and scene context are not required.

## Model

The vision-language backbone is Qwen3-VL-4B-Instruct-Action with M-RoPE retained.
The action head uses a 16-layer DiT-B with an attention width of 768 and 12 heads.
The state encoder and action decoder MLPs have a hidden dimension of 1024.
Training uses flow matching with uniform time sampling and 8 noise samples per training sample;
inference uses 4 Euler integration steps. See
[configs/model.json](configs/model.json) for the default configuration.

```text
mopa/
├── pyproject.toml
├── requirements.txt
├── README.md
├── models/                     # Qwen, query policy, and action head
├── configs/model.json
├── common.py                   # Joint layout, RGB resizing, and normalization
├── data/                       # Data preparation and training dataset
├── training/                   # Training entry point, optimizer, and checkpoints
├── runtime.py                  # Standalone checkpoint inference
└── integrations/               # Optional host protocol adapters
```

## Installation

Create a Python 3.11 environment in this directory and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

This directory can be copied and installed independently. Native data preparation,
training, and inference do not depend on a host project.
Prepare a local Qwen3-VL-4B-Instruct-Action directory containing the weights, config,
tokenizer, processor, and chat template. Specify it during training with `--base-vlm`
or `MOPA_BASE_VLM`. The dependency is `transformers==4.57.0`, and attention uses SDPA.

## Data Preparation

Each raw episode is an NPZ file with the following fields:

| Field | Format |
| --- | --- |
| `state` | float32 `[T,D]`, arm/gripper states |
| `action` | float32 `[T,D]`, joint action targets |
| `instruction` | Nonempty scalar string |
| `image_0`, `image_1`, … | uint8 RGB `[T,H,W,3]`, in the camera order specified by metadata |

For two arms, states and actions are ordered as left arm, left gripper, right arm,
and right gripper. For one arm, the order is arm, then gripper.
`metadata.json` declares the joint dimensions and camera layout, for example:

```json
{
  "action_type": "joint",
  "robot_action_dim_info": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
  "cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
  "image_size": [224, 224]
}
```

```bash
mopa-prepare --source /path/to/raw_episodes \
  --metadata /path/to/metadata.json --output /path/to/dataset
```

The output contains episode NPZ files, `metadata.json`, and `dataset_statistics.json`.
Images are resized to the specified dimensions and remain RGB. The q01/q99 statistics
use only actual frames. During training, states and actions are normalized to `[-1,1]`,
with constant dimensions mapped to zero. Action chunks at the end of an episode are
padded by repeating the final action. Existing dataset directories are not overwritten.

## Training

```bash
mopa-train --dataset /path/to/dataset --output /path/to/checkpoint \
  --base-vlm /path/to/Qwen3-VL-4B-Instruct-Action --seed 0 --device cuda
```

You can also run `python -m mopa.training.cli`. Defaults are 100000 training steps,
a batch size of 8, a backbone learning rate of `1e-5`, and a learning rate of `1e-4`
for all other parameters. `--config` accepts a JSON file of model parameters;
the query count, action horizon, and disabled base branch remain fixed.
See `mopa-train --help` for all options. Use `--device cpu` for CPU execution;
the backbone dtype is automatically set to float32.

Training saves the resolved model configuration, joint layout, camera order, data paths,
random seed, and training parameters. To reproduce a run, use the same data, Qwen assets,
dependencies, configuration, and seed; GPU numerical results may still vary with hardware.
Training uses a single process and does not restore optimizer or scheduler state.
Nonempty output directories are not overwritten.

## Inference

A checkpoint contains these three files, which must be kept together:

```text
config.json
model.pt
dataset_statistics.json
```

```python
import numpy as np
from mopa.runtime import Policy

policy = Policy("/path/to/checkpoint", device="cuda")
# rgb_views: List of uint8 RGB images in policy.cameras order.
# joint_state: Raw state vector [D] in joint layout order.
actions = policy.predict(
    images=[rgb_views],
    instructions=["Place the bowl on the plate."],
    states=np.asarray([joint_state], dtype=np.float32),
)
# actions.shape == (1, 4, policy.action_dim), restored to the original action units.
```

If the Qwen assets are moved, set `Policy(..., base_vlm="/new/path")`; their contents
must match those used during training. Batched inputs are supported, and inference
uses the same image preprocessing and statistics as training.
The checkpoint format is `xpl-mopa-arm-v1`; weights containing a base branch or two
query sets cannot be loaded directly.
