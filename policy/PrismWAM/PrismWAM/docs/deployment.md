# Deployment

Benchmarks have their own adapters ([guides](../README.md#evaluation-and-deployment)). On a robot, your
adapter builds the observations and executes the actions below.

## Load a policy

```python
from prismwam import load_pretrained

policy = load_pretrained("robodojo", device="cuda:0")   # released model, see models.md
policy = load_pretrained("exports/my_policy")           # your own exported model
policy.reset()                                          # at the start of every episode
out = policy.predict(observation, seed=0)
actions = out["action"][: out["replan_steps"]]          # execute, observe, predict again
```

Inference runs in bf16 on a GPU: one forward pass over the observed RGB, then 10 joint
depth/action denoising steps. Unless the model folder has a text cache, instructions are
encoded with the T5 encoder of Wan2.2-TI2V-5B, downloaded on first use to
`weights/Wan-AI/Wan2.2-TI2V-5B` (about 11 GB; `PRISMWAM_WEIGHTS_ROOT` changes `weights/`).

## Observations and actions

`policy.predict` expects tensors already in the training layout. Camera tiling and
history are the adapter's job.

| Key | Shape | Content |
| --- | --- | --- |
| `input_image` | `[1, 1, H, W]` | current depth mosaic, 0–1.5 m normalized to `[-1, 1]` |
| `rgb_input_image` | `[1, 3, H, W]` | current RGB mosaic, channel-first, in `[-1, 1]` |
| `rgb_history_images` | `[1, 3, T, H, W]` | RGB history (5 frames at stride 4; RoboDojo: first, 32-steps-ago, current) |
| `proprio` | `[1, 1, D_state]` | robot state in physical units (normalized by the model's statistics) |
| `prompt` | text | instruction, wrapped with `policy.format_prompt(instruction)` |
| `context`, `context_mask` | `[1, L, C]`, `[1, L]` | precomputed instruction features instead of `prompt` (real robot) |

`action` is the full `[H, D_action]` chunk (H = 32; LIBERO: 16), in physical units when the
model has statistics (otherwise it is marked `normalized: true`). Execute the first
`replan_steps` actions (16 for RoboDojo and LIBERO, 24 for RoboTwin), then predict again.

- **History.** Append every observed frame, including between replans, to
  `RGBHistoryBuffer.from_config(policy.config)` ([`deployment/history.py`](../src/prismwam/deployment/history.py));
  it builds `rgb_history_images` with the frames the model was trained on. `reset()` it
  and call `policy.reset()` every episode. RGB/depth normalization helpers are in
  [`deployment/observations.py`](../src/prismwam/deployment/observations.py).
- **Cameras.** RoboTwin and RoboDojo tile the head and both wrist cameras into 384 × 320;
  LIBERO puts agent view and wrist view side by side (224 × 448); the real-robot recipe
  uses one external camera.
- **LIBERO** normalizes with its dataset processor (min/max), not z-score statistics: pass
  the processor-normalized state and invert the actions with the same processor.
- **Real robot.** The six-task model has 14-D actions. Single-arm tasks normalize their 7
  real dimensions and are zero-padded to 14; take the first 7 outputs before
  un-normalizing. Export one model folder per task with `--task-name`.
- Joint/end-effector conversion, arm order and gripper units are up to the adapter.

## Export a trained model

```bash
python scripts/export_model.py \
  --config outputs/robodojo/stage2_joint/config.yaml \
  --checkpoint outputs/robodojo/stage2_joint/checkpoints/weights/step_040000.pt \
  --output exports/robodojo --sanitize-checkpoint
```

This collects the weights, config, VAEs and statistics into one folder for `load_pretrained`
and `serve.py --model`. Real-robot models also need `--task-name <task>` and `--text-cache <dir>`.

## Serve over HTTP

```bash
python scripts/serve.py --model robodojo --device cuda:0 --host 127.0.0.1 --port 8000
```

| Endpoint | Request | Response |
| --- | --- | --- |
| `GET /healthz` | | readiness, action size, horizon, replanning interval |
| `POST /infer` | `{"observation": {...}, "seed": 42}`, tensors as nested lists | the action chunk |
| `POST /reset` | `{}` | resets the episode state |

One process serves one episode at a time.
