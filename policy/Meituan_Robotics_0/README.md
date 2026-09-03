# Meituan Robotics 0

**Contributor:** Meituan Robotics | **Paper:** Not released | **arXiv:** Not released | **Original code:** [StarVLA](https://github.com/starVLA/starVLA)

Evaluation-only adapter for one RoboDojo StarVLA checkpoint family: Qwen3.5-4B
with MMDiT-Psi0, 640x480 RGB, and 14D dual-arm absolute joint actions. The
minimal inference runtime vendored in `source_starvla/` is copied from the
open-source StarVLA checkout; `policy/starVLA` is neither imported nor changed.

Shared argument conventions and split-machine deployment are documented in the
[XPolicyLab README](../../README.md). Official results are published on the
[RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
conda create -n meituan_robotics_0 python=3.11 -y
conda activate meituan_robotics_0
bash policy/Meituan_Robotics_0/install.sh
```

The tested runtime is Python 3.11, PyTorch 2.7.1, torchvision 0.22.1,
transformers 5.7.0, flash-attn 2.8.3, flash-linear-attention 0.5.1,
and causal-conv1d 1.6.2.post1.

## Data Processing

Not included in this eval-only adapter.

## Training

Not included in this eval-only adapter.

## Model Assets

Download the evaluated checkpoint, its resolved training configuration and
normalization statistics, and the Qwen3.5 tokenizer/processor assets together:

```bash
hf download wz7in/mr0 \
  --local-dir policy/Meituan_Robotics_0/checkpoints/mr0
```

The adapter resolves `pytorch_model.pt`, `config.full.yaml`,
`dataset_statistics.json`, and `base_vlm/` from that directory. The released
checkpoint contains the complete VLM and action-head weights; `base_vlm/`
contains configuration and preprocessing assets only, so no second copy of the
Qwen weights is required.

## Evaluation

Keep `config.full.yaml`, `dataset_statistics.json`, and `base_vlm/` beside the
checkpoint, as produced by the download command above. The legacy training
layout with the checkpoint inside `checkpoints/` or `final_model/` is also
supported. Startup rejects checkpoints whose framework, data mix, resolution,
action mode, normalization, horizon, or noise settings differ from this
profile.

```bash
bash policy/Meituan_Robotics_0/eval.sh \
  RoboDojo stack_bowls mr0 arx_x5 joint 0 \
  0 1 meituan_robotics_0 robodojo
```

Supported settings are `bench_name=RoboDojo`,
`env_cfg_type=arx_x5`, and `action_type=joint`. The checkpoint predicts 50
steps with 10 Euler denoising steps; the adapter returns the first 16 actions.

## Configuration

- `checkpoint_path`: exact checkpoint file or run directory; the
  `MEITUAN_ROBOTICS_CKPT_PATH` environment variable overrides it.
- `base_vlm`: automatically resolves to `base_vlm/` beside the released
  checkpoint; `STARVLA_BASE_VLM` can override it.
- `image_size`: fixed at `[640, 480]`.
- `execute_horizon`: fixed at `16`.

Images passed to `model.py` are already decoded RGB arrays and are never
channel-swapped. State order is left arm 6, left gripper 1, right arm 6, right
gripper 1. State uses the training q01/q99 transform with clipping to
`[-1.0, 1.0]`; actions use its inverse. Grippers remain continuous. Relative
or delta conversion, correlated noise, gripper thresholds, temporal smoothing,
and overlap blending are not enabled.
