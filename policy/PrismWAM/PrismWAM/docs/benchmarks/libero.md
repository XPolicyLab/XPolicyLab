# LIBERO

One policy for LIBERO Spatial, Object, Goal and Long (40 tasks × 50 demonstrations). 7-D delta end-effector actions with an absolute gripper, from the agent-view and wrist cameras and an 8-D robot state.

## Download

| | Hugging Face | Local path |
| --- | --- | --- |
| Model | [`libero/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/libero) | `weights/libero/` |
| Training initializers (shared) | [`init/`](https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/init) | `weights/init/` |

```bash
hf download THUSI-Lab/PrismWAM --include "libero/*" --local-dir weights        # evaluation
hf download THUSI-Lab/PrismWAM --include "init/*" --local-dir weights   # training
```

Training also needs the [pretrained video model](../models.md#training-assets).

## Data

Put the four suites re-rendered with depth (`<suite>/*_demo.hdf5`, source revision `f13aa24`) under `data/libero/`. Then build `action_stats.json` (min/max) and the text cache ([data preparation](../data.md)). There is no validation split.

## Train

```bash
bash scripts/libero/stage1_depth.sh     # Stage 1: depth warm-up
bash scripts/libero/stage2_joint.sh     # Stage 2: joint RGB / depth / action
```

The two cameras sit side by side in one 224 × 448 image, separately for depth and RGB.

| | Stage 1 | Stage 2 |
| --- | --- | --- |
| Depth input | current frame | current frame |
| RGB input | — | 5 frames, one every 4 steps (current + 4 past) |
| Robot state | — | 8-D end-effector and gripper state |
| Predicts | 4 future depth frames | 1 RGB goal frame (16 steps ahead), 4 future depth frames, 16-step action chunk |
| Steps · global batch | 15K · 256 | 15K · 256 |
| Recipe | [`stage1_depth.yaml`](../../configs/libero/stage1_depth.yaml) | [`stage2_joint.yaml`](../../configs/libero/stage2_joint.yaml) |

Export `step_015000.pt` with [`export_model.py`](../deployment.md#export-a-trained-model).

## Evaluate

The model server runs in the PrismWAM environment, the simulator client in a LIBERO
environment (LIBERO plus `future`, `matplotlib` and `h5py`):

```bash
python scripts/libero/eval.py serve --model libero --port 8001                          # PrismWAM env
MUJOCO_GL=egl python scripts/libero/eval.py run --port 8001 --output outputs/libero_eval   # LIBERO env
```

`run` covers the four suites, resumes from `episodes.jsonl` in the output folder and writes
per-suite success rates to `summary.json`; `--suites`, `--tasks` and `--episodes` pick a subset.

| Setting | Value |
| --- | --- |
| Suites | `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` |
| Episodes | 50 per task, 2,000 in total, seed 42 |
| Episode length | 400 steps (`libero_10`: 700), after 30 settling steps |
| Actions | 16-step chunk, execute 16; binarized gripper; no action ensembling |
| Inference | 10 denoising steps, no conditioning noise |

In your own rollout loop, [`LiberoPolicy`](../../src/prismwam/integrations/libero.py) takes raw
simulator observations:

- **Images**: agent-view and wrist RGB in the simulator's raw OpenGL array orientation, not rotated by 180°.
- **Depth**: metric depth in metres from the same cameras, 0 for invalid pixels.
- **State**: `ee_pos` (3), axis-angle of `ee_quat` (3), `gripper_qpos` (2).
- **Actions**: un-normalized with the model's `norm_stats.json`; the gripper is returned as −1 (open) / +1 (close).

## Results

| Model | Spatial · Object · Goal · Long (avg. success) |
| --- | ---: |
| PrismWAM | 99.35 % |

Comparisons with other models, and LIBERO-Plus and LIBERO-Pro results, are on the [project page](https://prism-wam.github.io/).
