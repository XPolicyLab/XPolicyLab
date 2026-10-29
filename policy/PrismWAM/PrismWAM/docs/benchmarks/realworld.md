# Real robot

The recipe of our six-task real-world model. The recordings are private; use it as a template for your own robot.

| Task | Arms | Action |
| --- | --- | --- |
| `move_plate`, `grab_plate`, `reorder_cubes`, `rgb` | one | 7-D joints + gripper |
| `pickup_plate`, `microwave` | two | 14-D joints + grippers |

All tasks share one 14-D action expert: single-arm tasks are normalized on their 7 real dimensions and zero-padded to 14. Each task keeps its own normalization statistics.

## Download

| | Hugging Face | Local path |
| --- | --- | --- |
| Training initializers (shared) | [`init/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/init) | `weights/init/` |

```bash
hf download THUSI-Lab/PrismWAM --include "init/*" --local-dir weights
```

Training also needs the [pretrained video model](../models.md#training-assets). No real-robot policy is released.

## Data

Each task is one entry of `data.train.sources`, pointing at a LeRobot-style recording root (RGB, processed depth under `depth_namespace`, joint states, actions). Per-source fields:

| Field | Purpose |
| --- | --- |
| `action_dim` | 7 or 14 |
| `action_stats_cache_path` | per-task statistics, fitted with `prepare_data.py stats --node data.train.sources.<i>` |
| `episode_filter_path`, `episode_frame_bounds_path` | which episodes and frame ranges to train on |
| `task_instruction_dir`, `text_embedding_cache_dir` | instructions and their encoded features |

Shared fields go under `source_defaults`. Every 20th episode of each task is held out for validation (`validation_modulus: 20`). The recipe expects one external RGB-D camera.

## Train

```bash
bash scripts/realworld/stage1_depth.sh    # Stage 1: depth warm-up
bash scripts/realworld/stage2_joint.sh    # Stage 2: joint RGB / depth / action
```

Both stages run 15K steps at global batch 256. To add or remove a task, edit the `sources` lists in both [`stage1_depth.yaml`](../../configs/realworld/stage1_depth.yaml) and [`stage2_joint.yaml`](../../configs/realworld/stage2_joint.yaml), keeping the training and validation lists in the same order.

## Deploy

Export one model folder per task; `--task-name` picks its action size and statistics, `--text-cache` adds its instruction features:

```bash
python scripts/export_model.py \
  --config outputs/realworld/stage2_joint/config.yaml \
  --checkpoint outputs/realworld/stage2_joint/checkpoints/weights/step_015000.pt \
  --task-name grab_plate --text-cache data/realworld/text_cache \
  --output exports/grab_plate --sanitize-checkpoint
```

Serve it with `scripts/serve.py --model exports/grab_plate`. Observation and action format: [deployment guide](../deployment.md#observations-and-actions). The policy executes 24 of every 32 predicted actions.
