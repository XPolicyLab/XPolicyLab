# SLA: Scaling Spatial-Language-Action Model via Continual Pre-training

For the XPolicyLab RoboTwin adapter, follow the [adapter guide](../README.md). The sections below describe upstream workflows.

## 0. TODO List
✅ LIBERO-Plus post-training and evaluation<br>
✅ RoboTwin post-training and evaluation<br>
✅ Continual pre-training corpus<br>
❌ RoboDojo-Real post-training and evaluation (before 20261109)<br>
❌ Training efficiency optimization<br>
❌ Continual pre-training data<br>

## 1. Environment Setup
We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies:
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```


## 2. Data Preparation for Training
All the post-training data are from the official benchmark settings and data sources:
```bash
# mkdir Data dictionary
mkdir Data/

# LIBERO Plus benchmark use the LIBERO-only training data
hf download physical-intelligence/libero --repo-type dataset --local-dir Data/libero_lerobot/

# RoboTwin benchmark use the 50 tasks and 50 episodes per task for co-training
hf download TianxingChen/RoboTwin2.0 lerobot_dataset/RoboTwin_lerobot_v21.zip --repo-type dataset --local-dir Data/
unzip Data/lerobot_dataset/RoboTwin_lerobot_v21.zip -d Data/robotwin_lerobot/

# Real-world deployment demo
hf download haofuly/SLA_real_world_demo  --repo-type dataset --local-dir Data/real_world_handover_vegetable/
```

## 3. Training
First, download our pre-trained checkpoints and other necessary training components:
```bash
# Mkdir Data dictionary
mkdir checkpoints

# Our SLA pre-trained ckpts
hf download haofuly/SLA_base --repo-type model --local-dir checkpoints/SLA_base/

# Other components (`e.g.`, OmniVGGT, DINOv2)
hf download haofuly/SLA_training_component --repo-type model --local-dir checkpoints/
```

Next, write the config file. We give some examples at [src/openpi/external_config](./src/openpi/external_config).

Then, calculate the action and state normalization statistics:
```bash
source .venv/bin/activate
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>

# example for LIBERO-Plus
uv run scripts/compute_norm_stats.py --config-name train_libero_plus
```

Finally, run training commands:
```bash
source .venv/bin/activate
uv run --no-sync scripts/train_align.py <CONFIG_NAME> --exp-name=<EXP_NAME>

# example for LIBERO-Plus
uv run --no-sync scripts/train_align.py train_libero_plus --exp-name=experiments
```

## 4. Inference
> We have provided the already post-trained checkpoints for this inference.  
> You can download them from [SLA_benchmark_ckpts](https://huggingface.co/haofuly/SLA_benchmark_ckpts) and place them into checkpoints dir.


<details>
<summary><strong>LIBERO-Plus</strong></summary>

First, initialize the [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) submodule:
```bash
git submodule update --init --recursive third_party/libero_plus
```
Then, download the official [assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main). Then, unzip and place them like:
```text
third_party/
└── libero_plus/
    └── libero/
        └── libero/
            └── assets/
                ├── articulated_objects/
                ├── new_objects/
                ├── scenes/
                ├── stable_hope_objects/
                ├── stable_scanned_objects/
                ├── textures/
                ├── turbosquid_objects/
                ├── serving_region.xml
                ├── wall_frames.stl
                └── wall.xml
```

Then, Terminal window 1:
```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero_plus/requirements.txt third_party/libero_plus/extra_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero_plus

# Run the simulation
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero_plus
source examples/libero/.venv/bin/activate
python examples/libero/main.py --args.task-suite-name <TASK>  --args.num_trials_per_task 1 --args.port <PORT>

# example
python examples/libero/main.py --args.task-suite-name libero_spatial  --args.num_trials_per_task 1 --args.port 8000
```

Terminal window 2:
```bash
# Run the server
source .venv/bin/activate
uv run scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<CKPT_DIR> --port <PORT>

# example
uv run scripts/serve_policy.py policy:checkpoint --policy.config=train_libero_plus --policy.dir=checkpoints/train_libero_plus/experiments/29999 --port 8000
```
</details>


<details>
<summary><strong>RoboTwin</strong></summary>
</details>
