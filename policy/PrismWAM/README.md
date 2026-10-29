# PrismWAM

**Contributor:** PrismWAM contributors | **Paper:** PrismWAM: Decoupling State Representations in World Action Models | **arXiv:** https://arxiv.org/abs/2610.26564 | **Original code:** https://github.com/THUSI-Lab/PrismWAM

`PrismWAM` adapts the PrismWAM RGB/depth world action model to XPolicyLab for RoboTwin 2.0
(`aloha_agilex`) and RoboDojo (`arx_x5`), with 14-D absolute joint actions (`action_type=joint`).
For RoboTwin it also serves PrismWAM\*.
Integration scripts live at this directory level; the vendored upstream implementation lives in `PrismWAM/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requires Linux x86-64, an NVIDIA GPU and the CUDA 12.8 toolkit (`nvcc`). `install.sh` creates a
conda-forge environment, installs PyTorch, the vendored `PrismWAM/` and this XPolicyLab checkout.
The simulator client keeps its own environment.

```bash
cd XPolicyLab/policy/PrismWAM
bash install.sh            # environment name, default: prismwam
conda activate prismwam
```

## Model Assets

```bash
bash download_checkpoint.sh prismwam/robotwin   # or prismwam/robodojo, prismwam_star/robotwin; writes checkpoints/<name>/
```

Each folder holds the model, its VAEs and action statistics. The T5 text encoder is fetched into
`PrismWAM/weights/` on first use (set `PRISMWAM_WEIGHTS_ROOT` to use another location);
`prismwam_star/robotwin` (PrismWAM\*) also fetches the CLIP image encoder of Wan2.1-I2V-14B-480P there.

## Data Processing

Training reads the native RGB/depth HDF5 recordings, not the LeRobot exports, plus the recipe's
statistics and text cache.

```bash
bash process_data.sh BENCH CHECKPOINT ROBOT joint COMMAND [ARGS...]
# Example: RoboDojo episode index
bash process_data.sh RoboDojo my_run arx_x5 joint robodojo-index \
  --root /path/to/data/robodojo --output /path/to/data/robodojo/index.json
```

Arguments after the first four go to `PrismWAM/scripts/prepare_data.py`; the statistics and text
cache are built the same way ([data preparation](PrismWAM/docs/data.md)).

## Training

```bash
bash train.sh BENCH CHECKPOINT ROBOT joint SEED GPU_IDS RECIPE stage1_depth|stage2_joint [KEY=VALUE...]
# Example: RoboTwin, Stage 1 then Stage 2
bash train.sh RoboTwin my_run aloha_agilex joint 42 0,1,2,3,4,5,6,7 robotwin stage1_depth paths.data_root=/path/to/data
bash train.sh RoboTwin my_run aloha_agilex joint 42 0,1,2,3,4,5,6,7 robotwin stage2_joint paths.data_root=/path/to/data
```

Recipes are `PrismWAM/configs/<recipe>/<stage>.yaml`; Stage 2 starts from the Stage-1 checkpoint
under `outputs/<run>/<recipe>/stage1_depth/`. For multi-node runs set `NNODES`, `NODE_RANK` and
`MASTER_ADDR`; the global batch stays fixed. Export a trained Stage-2 checkpoint as a model folder
with `PrismWAM/scripts/export_model.py`.

## Evaluation

The policy reads depth from every camera, which RoboTwin does not render by default: set
`data_type.depth: true` in `env_cfg/task_config/demo_clean.yml` and `demo_randomized.yml`.
RoboDojo needs the depth setup in [RoboDojo evaluation](PrismWAM/docs/benchmarks/robodojo.md#evaluate)
and `PRISMWAM_DEPTH_KEY=approximate_depth`.

```bash
PRISMWAM_RECIPE=<recipe> bash eval.sh BENCH TASK CHECKPOINT ROBOT joint SEED POLICY_GPU ENV_GPU POLICY_ENV EVAL_ENV
# Example: RoboTwin
export PRISMWAM_RECIPE=robotwin
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin adjust_bottle checkpoints/prismwam/robotwin aloha_agilex joint 0 0 0 prismwam RoboTwin
# PrismWAM*: the same recipe with checkpoints/prismwam_star/robotwin
```

`CHECKPOINT` is a model folder: an absolute path, a path relative to this directory, or a name
under `checkpoints/`. All 50 RoboTwin tasks run through RoboTwin's scheduler, once per scene setting:

```bash
cd /path/to/RoboTwin
bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml --policy-name PrismWAM \
  --ckpt-name "$PWD/XPolicyLab/policy/PrismWAM/checkpoints/prismwam/robotwin" --env-cfg-type aloha_agilex \
  --policy-conda-env prismwam --eval-env-conda-env RoboTwin --task-config demo_clean \
  --test-num 100 --instruction-type unseen --seed 0
```

## Configuration

| `deploy.yml` key | Meaning |
| --- | --- |
| `recipe` | `robotwin` or `robodojo` (required); or `PRISMWAM_RECIPE`. |
| `model_path` | Model folder; overrides `ckpt_name`; or `PRISMWAM_MODEL`. |
| `camera_map` | External / left / right wrist cameras; default `cam_head`, `cam_left_wrist`, `cam_right_wrist`. |
| `depth_key`, `depth_scale` | Depth field and its scale to millimetres (`1` for mm, `1000` for m); or `PRISMWAM_DEPTH_KEY`, `PRISMWAM_DEPTH_SCALE`. |
| `device`, `dtype` | Default `cuda`, `bfloat16`; or `PRISMWAM_DEVICE`, `PRISMWAM_DTYPE`. |
| `inference_seed` | Model-noise seed; default `null` reuses `seed`; or `PRISMWAM_INFERENCE_SEED`. |

Denoising steps, the replanning interval and normalization come from the model folder.

## Notes

- Actions are ordered left-arm joints and gripper, then right-arm joints and gripper; dimensions come from XPolicyLab's robot helpers.
- One server serves one evaluation client, which may batch several environments.
