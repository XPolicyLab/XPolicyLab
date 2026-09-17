# Post-training

Public posttrain and pretrain recipes share the torchrun/DeepSpeed bootstrap in
`recipes/common/launcher_common.sh`.

## Generalist vs expert


| Launcher family                       | Config                             | Role                           |
| ------------------------------------- | ---------------------------------- | ------------------------------ |
| `posttrain_libero.sh`                 | `configs/posttrain/libero.yaml`    | Multi-task LeRobot generalist  |
| `posttrain_robotwin2.sh`              | `configs/posttrain/robotwin2.yaml` | Multi-task LeRobot generalist  |
| `posttrain_robotwin2_full.sh`         | `configs/posttrain/robotwin2.yaml` | Full-data RoboTwin2 generalist |
| `posttrain_agilex.sh` + task wrappers | `configs/posttrain/agilex.yaml`    | Single-task AgileX expert      |


AgileX task recipes only select `mode`, visual prompt, and dataset profile.
They do not introduce task-specific YAML files.

Example AgileX expert launch:

```bash
PRETRAINED_MODEL_PATH=/models/wsp DATA_ROOT=/data/agilex \
./recipes/posttrain/posttrain_agilex_fold_shirt_text.sh training.max_steps=1000
```

Example LIBERO generalist launch:

```bash
PRETRAINED_MODEL_PATH=/models/wsp DATA_ROOT=/data/libero \
./recipes/posttrain/posttrain_libero.sh
```

Use the full-data RoboTwin2 wrapper when training on the complete dataset:

```bash
PRETRAINED_MODEL_PATH=/models/wsp \
DATA_ROOT=/data/robotwin2 \
ZSCORE_STATS_PATH=/data/robotwin2/dataset_stats.json \
./recipes/posttrain/posttrain_robotwin2_full.sh
```

Both launchers train 14-D absolute joint actions with global z-score normalization; use statistics computed from the same dataset profile.

Posttrain YAMLs use a strict top-level `includes` list. Paths are relative to
the YAML that declares them; included files merge in order, followed by the
declaring file. Selector profiles are resolved after composition, and command
line dot-list overrides are applied last. The public platform paths and recipe
commands remain unchanged.

Shared Wan2.2 defaults live in `configs/posttrain/common_wan22.yaml`, which
recursively composes model, conditioning, memory, and WAM-owned fragments.
Platform YAMLs contain only selector defaults, dataset constraints, loader
differences, evaluation linkage, and platform prompt/geometry details.

## Auto VLM token flow

The default Auto path uses Qwen's final hidden layer for perception and appends
the final-layer hidden states of autoregressively generated planning tokens.
Both are in the Qwen token width (`VLM_TOKEN_DIM`) and share one projector into
the WAM condition width (`VLM_CONTEXT_DIM`). QFormer remains optional:

```bash
VLM_TOKEN_MODE=qformer FREEZE_QFORMER=false \
./recipes/posttrain/posttrain_agilex_fold_shirt_text.sh
```

In that mode QFormer compresses only the multimodal perception prefill. Planning
tokens remain Qwen autoregressive states, so QFormer output is deliberately kept
at `VLM_TOKEN_DIM` before the shared WAM projector.

## Freeze-policy ownership

`freeze.config` is the only training-time ownership point for parameter
trainability. Conditioning/model fragments describe structure and checkpoint
paths only; model constructors do not apply their own freeze flags. A QFormer
created with `VLM_TOKEN_MODE=qformer` inherits the parent `vlm: true` rule by
default. Set `FREEZE_QFORMER=false` to add the more-specific trainable override.
In `last` mode QFormer is absent and no QFormer freeze rule is generated.

Each platform posttrain config also owns its evaluation transform metadata.
Successful saves make every `checkpoint-N/` directly evaluable and resumable.
Eval recipes select the highest complete checkpoint by default, while an
explicit task-specific eval model path still takes precedence. No duplicate
`final/` weights are written. Resume uses either fast mode (default, worker
prefetch retained) or exact mode (`RESUME_MODE=exact`, workers forced to zero).