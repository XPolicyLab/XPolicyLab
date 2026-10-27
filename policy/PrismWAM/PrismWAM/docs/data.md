# Data preparation

Training reads RGB-D recordings from `DATA_ROOT/<benchmark>/` (default `./data`). Each recipe
also needs action statistics, a text cache and, for RoboDojo, an episode index. Build them
with `scripts/prepare_data.py` from the repository root; it refuses to overwrite an index or
statistics file, and the text cache skips prompts that are already encoded.

## Expected layout

```
data/
  robotwin/
    <task>/demo_clean/data/episode*.hdf5        RoboTwin 2.0 RGB-D episodes
    <task>/demo_clean/scene_info.json           per-episode scene info (written by RoboTwin)
    <task>/demo_clean_val/...                   validation episodes, same layout
    instructions/  object_descriptions/         RoboTwin's description/task_instruction/ and
                                                description/objects_description/
    action_stats.pt  text_cache/                built below
  robodojo/
    <task>/arx_x5/data/episode_*.hdf5           RoboDojo ARX X5 recordings, 35 tasks x 100 episodes
    index.json  holdout/{train,val}.json        episode index and 90/10 partition, built below
    action_stats.pt  text_cache/                built below
  libero/
    libero_spatial/  libero_object/  libero_goal/  libero_10/   *_demo.hdf5 with depth
    action_stats.json  text_cache/              built below
  realworld/                                    your recordings; see the real-robot guide
```

File names come from `configs/<benchmark>/*.yaml`; change them there if your layout differs.

- **RoboTwin 2.0**: `demo_clean` episodes for the 50 tasks, collected with depth enabled
  (`data_type.depth: true` in the task config), plus a `demo_clean_val` set (10 episodes per
  task are used for validation). Stage 2 decodes RGB with XPolicyLab's helper from a RoboTwin
  checkout (`ROBOTWIN_ROOT`, default `./external/RoboTwin`, the directory containing
  `XPolicyLab/utils/process_data.py`).
- **RoboDojo**: RGB decoding uses XPolicyLab's helper from a RoboDojo checkout
  (`ROBODOJO_ROOT`, default `./external/RoboDojo`, the directory containing
  `XPolicyLab/utils/process_data.py`).
- **LIBERO**: the four standard suites re-rendered with depth (source revision `f13aa24`),
  10 tasks × 50 demos each, agent-view and wrist-view cameras.

## 1. Episode index (RoboDojo only)

```bash
python scripts/prepare_data.py robodojo-index \
  --root data/robodojo --output data/robodojo/index.json
python scripts/prepare_data.py robodojo-holdout \
  --index data/robodojo/index.json --output-dir data/robodojo/holdout
```

Training uses every episode in `index.json`; `holdout/val.json` (10 episodes per task) is
the validation set and `holdout/train.json` the split the statistics are fitted on. Indexes
store absolute paths; after moving the data, rewrite each file with
`relocate-index --index ... --root ... --output ...`.

## 2. Normalization statistics

Fit on training data only; both stages and the exported policy use the same file.

```bash
# RoboTwin: z-score statistics of the 14-D joint actions
python scripts/prepare_data.py stats \
  --config configs/robotwin/stage1_depth.yaml \
  --output data/robotwin/action_stats.pt paths.data_root=./data

# RoboDojo: z-score statistics on the 90% training split
python scripts/prepare_data.py joint14-stats \
  --index data/robodojo/holdout/train.json --output data/robodojo/action_stats.pt

# LIBERO: min/max statistics of the 7-D actions and 8-D states
python scripts/prepare_data.py stats \
  --config configs/libero/stage1_depth.yaml \
  --output data/libero/action_stats.json paths.data_root=./data
```

LIBERO statistics record the data root; use the same `DATA_ROOT` for training, and rebuild
them if the data moves. For mixed real-robot data, fit one file per task by selecting its
source, e.g. `--node data.train.sources.0`.

Released checkpoints ship their statistics in the model folder. Do not replace them with a newly
fitted file.

## 3. Text cache

Encode instructions with the pretrained video model's text encoder
([training assets](models.md#training-assets)). Run once per benchmark, after steps 1 and 2:
the command builds the training and validation datasets to collect their instructions.

```bash
python scripts/prepare_data.py text \
  --config configs/robotwin/stage2_joint.yaml \
  --checkpoint weights/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer weights/Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl \
  --device cuda --batch-size 8
```

Replace `robotwin` with `robodojo`, `libero` or `realworld`. Both stages share the same
`text_cache/`.
