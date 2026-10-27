# Training

Each benchmark is trained in two stages, both with flow matching.

## The two stages

### Stage 1: depth warm-up

Adapts the pretrained video transformer from RGB to depth video, so it learns how the scene
moves before RGB targets and actions are added.

| | |
| --- | --- |
| Input | current depth frame, instruction (precomputed text features) |
| Target | next 8 depth frames, one every 4 steps (the 32-step action horizon; LIBERO: 4 frames, 16 steps) |
| Trained | visual transformer |
| Loss | depth flow matching |
| Output | `outputs/<benchmark>/stage1_depth/checkpoints/weights/step_015000.pt` |

### Stage 2: joint RGB / depth / action

Trains all three towers together.

| Tower | Input | Target | Weights |
| --- | --- | --- | --- |
| RGB | 5 observed frames at stride 4 (RoboDojo: first, 32 steps ago, current) | one RGB frame 32 steps ahead (LIBERO: 16) | shared with depth |
| Depth | current depth frame | 8 future depth frames, as in Stage 1 (LIBERO: 4) | shared with RGB |
| Action | depth features | 32-step action chunk (LIBERO: 16) | separate action expert |

- The shared RGB/depth blocks start from the Stage-1 checkpoint, the RGB input/output layers
  from the pretrained video model, and the action expert from `init/action_initializer.pt`.
- The instruction and the robot state form one context that all three towers attend to (on
  RoboDojo the RGB tower sees the instruction only).
- Attention is one-way, RGB → depth → action. Nothing reads the predicted RGB frame, so it is
  skipped at inference.
- Loss = RGB + depth + action, equal weights, with a separate noise level per tower.
- Inference encodes the observed RGB once and denoises depth and actions for 10 steps.

## Launching

```bash
bash scripts/robotwin/stage1_depth.sh
bash scripts/robotwin/stage2_joint.sh
```

Paths are set at the top of each script, or per run:

```bash
DATA_ROOT=/data/prismwam WEIGHTS_ROOT=/models/prismwam OUTPUT_ROOT=/runs \
  bash scripts/robotwin/stage1_depth.sh
```

| Variable | Default | Points to |
| --- | --- | --- |
| `DATA_ROOT` | `./data` | `<benchmark>/` training data, see [data](data.md) |
| `WEIGHTS_ROOT` | `./weights` | pretrained video model and `init/`, see [models](models.md) |
| `OUTPUT_ROOT` | `./outputs` | `<benchmark>/<stage>/` checkpoints, logs, resolved config |
| `ROBODOJO_ROOT` | `./external/RoboDojo` | RoboDojo checkout for RGB decoding (RoboDojo only) |
| `ROBOTWIN_ROOT` | `./external/RoboTwin` | RoboTwin checkout with `XPolicyLab/`, for RGB decoding (RoboTwin Stage 2 only) |

Extra arguments are passed through:

```bash
bash scripts/robotwin/stage1_depth.sh --dry-run          # resolve and check, print the launch command
bash scripts/robotwin/stage1_depth.sh batch_size=8             # less memory per GPU, same global batch
bash scripts/robotwin/stage1_depth.sh wandb.enabled=true wandb.mode=online
```

### GPUs and nodes

The launcher uses all visible GPUs. For multiple nodes, run the same command on each node:

```bash
export NNODES=4 NODE_RANK=0 MASTER_ADDR=10.0.0.1   # NODE_RANK = 0, 1, 2, 3 on the four nodes
bash scripts/robotwin/stage1_depth.sh
```

`GPUS_PER_NODE` overrides the GPU count; `MASTER_PORT` defaults to 29500. `batch_size` is per
GPU. Gradient accumulation is `global_batch_size / (batch_size × GPUs)`, so the global batch
stays the same when the GPU count changes. If it does not divide evenly, the launcher stops
and suggests a `batch_size`.

| Recipe | Global batch | Steps |
| --- | ---: | ---: |
| `robotwin/stage1_depth` | 256 | 15K |
| `robotwin/stage2_joint` | 256 | 15K |
| `robodojo/stage1_depth` | 1024 | 15K |
| `robodojo/stage2_joint` | 1024 | 40K |
| `libero/stage1_depth` | 256 | 15K |
| `libero/stage2_joint` | 256 | 15K |
| `realworld/stage1_depth` | 256 | 15K |
| `realworld/stage2_joint` | 256 | 15K |

All recipes: AdamW (β = 0.9, 0.95; weight decay 0.01), lr 1e-4 with 5% linear warm-up and cosine
decay to 1e-6, gradient clipping 1.0, bf16, DeepSpeed ZeRO-2
([`configs/distributed/zero2.yaml`](../configs/distributed/zero2.yaml)).

### Resuming and the Stage-1 checkpoint

Training state is saved every 5,000 steps under `<output_dir>/checkpoints/state/`:

```bash
bash scripts/robotwin/stage2_joint.sh \
  resume=./outputs/robotwin/stage2_joint/checkpoints/state/step_010000
```

Stage 2 starts from the final Stage-1 checkpoint,
`<OUTPUT_ROOT>/<benchmark>/stage1_depth/checkpoints/weights/step_015000.pt`. To use one from
elsewhere, pass `model.depth_dit_checkpoint_path=/path/to/step_015000.pt`.

## Outputs

```
outputs/<benchmark>/<stage>/
  config.yaml                       resolved config
  checkpoints/weights/step_*.pt     weights, every 1,000 steps (5,000 for realworld)
  checkpoints/state/step_*/         training state for resume=
```

Export the final Stage-2 weights with
[`scripts/export_model.py`](deployment.md#export-a-trained-model).

## Configs

```
configs/base.yaml                   paths, optimizer, precision, logging
configs/stages/stage1_depth.yaml    Stage-1 model and schedule
configs/stages/stage2_joint.yaml    Stage-2 model, schedule, inference
configs/<benchmark>/<stage>.yaml    data, batch size, steps
```

Parents listed under `defaults:` are merged first, then the file's own keys, then command-line
`key=value` overrides. For example,
[`configs/libero/stage2_joint.yaml`](../configs/libero/stage2_joint.yaml) keeps the Stage-2
model and sets the LIBERO data (two cameras at 224 × 448, 7-D actions, 8-D state), batch size
and replanning interval. To print a merged config:

```bash
python -c "from prismwam.config import load_config; from omegaconf import OmegaConf; \
print(OmegaConf.to_yaml(load_config('configs/libero/stage2_joint.yaml')))"
```

For a new dataset, copy the closest benchmark configs and change the `data` section and paths.
