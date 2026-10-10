# Focus-VLWA

English | [简体中文](README_zh-CN.md)

Focus-VLWA jointly models visual history, future world states, and robot actions for dual-arm manipulation.
This release contains post-training, dataset processing, and local inference. This source copy is bundled with the RoboDojo/XPolicyLab adapter under `policy/ME_Brain_1`.

## Demo

[![Watch the Focus-VLWA demo](https://raw.githubusercontent.com/MachEmbodied/Focus-VLWA/870aa92/docs/media/demo-preview.jpg)](https://cdn.jsdelivr.net/gh/MachEmbodied/Focus-VLWA@870aa92/docs/media/focus-vlwa-demo-web.mp4)

Click the preview to watch the full demo (6 minutes).

## Model design

The implementation is organized around these project-specific components:

- **World-model expert**: an independent 18-layer, 300M-configuration transformer denoises one event action and 270 future-state tokens, trained with event and future-state supervision.
- **Joint expert attention**: visual-language, action, and world representations retain independent weights while sharing attention computations. Action and world tokens exchange information bidirectionally; the prefix cannot attend to the suffix.
- **Joint flow matching**: action and world predictions evolve together over the same Euler timesteps. The action expert outputs 50 actions; the world expert outputs the event action and future state.
- **Full head history encoder**: the current post-training profile, `head_history`, prepends 20 full head frames, sampled every 25 control steps from completed episode-grid slots at 25 FPS. Missing frames are zero-padded and masked; the current frame is excluded from its own history. Each frame contributes 16 tokens; three current views retain 256 tokens each (1088 visual tokens in total). Current and historical head images share augmentation per sample.

## Source layout

```text
src/focus_vlwa/
  model/          # Head history, world expert, joint attention and flow matching
  configs/        # Architecture and post-training configuration
  data/           # LeRobot samples, normalization and full-frame history
  inference/      # Tokenizer and local inference policy
  post_training/  # PyTorch/DDP training and checkpoint export
  scripts/        # Command implementations
tests/
```

## Installation

```bash
uv sync --extra train
uv run focus-vlwa install-transformers-patch
uv run focus-vlwa check-checkpoint /path/to/checkpoint
```

Use a dedicated environment: the compatibility patch installs into that environment's `transformers==4.53.2`.
The model uses 20-frame `head_history`, 400 prompt tokens, 32 internal action dimensions, and 50 action steps. Checkpoint loading reads `model_config.json` and maps the released Focus-VLWA weights to the refactored module names. `check-checkpoint` applies the same mapping and checks every stored tensor shape.

## focus-vlwa checkpoint

The inference bundle contains exactly these required files:

```text
focus-vlwa-release/
  model.safetensors
  model_config.json
  assets/arx_x5_sim/norm_stats.json
```

The original training directory also contains `metadata.pt` and `optimizer.pt`; neither is needed in an inference release. Keep `model_config.json` alongside the weights so inference never has to deserialize training metadata. The PaliGemma tokenizer is resolved separately: pass `tokenizer_path` for offline use, or allow the package to download it into its cache. Use `asset_id="arx_x5_sim"` when constructing `FocusVLWAPolicy`.

Validate the published directory before evaluation:

```bash
uv run focus-vlwa check-checkpoint /path/to/focus-vlwa-release
```

## Local inference

```python
from focus_vlwa.inference import FocusVLWAPolicy

policy = FocusVLWAPolicy(
    '/path/to/checkpoint',
    tokenizer_path='/path/to/tokenizer.model',
    device='cuda',
)
prediction = policy.infer({
    'state': joint_state,  # [14]
    'images': {'cam_high': head_rgb, 'cam_left_wrist': left_rgb, 'cam_right_wrist': right_rgb},
    'prompt': 'stack the bowls',
    'hist_images': history_images,  # [20, 224, 224, 3], uint8
    'hist_mask': history_mask,  # [20], valid=1, padding=0
})
actions = prediction['actions']  # [50, 14], absolute robot joint targets
```

For deterministic comparison, provide `noise`, `event_noise`, and `world_noise` with shapes `[1,50,32]`, `[1,1,32]`, and `[1,270,32]`.
The `head_history` input contains only historical head-camera frames, with `hist_images: [20,224,224,3]` and `hist_mask: [20]`; it has no `hist_cells`. Current observations still contain all three cameras. Each historical frame contributes 16 tokens (320 total), and each current view contributes 256 tokens (768 total), giving 1088 visual tokens before text.

Use `HeadHistoryBuffer` to produce these arrays: observe every control step, including steps between replans, and reset the buffer at every new episode. Initial missing history is zero-padded and masked. See the buffer example below. The XPolicyLab adapter supplies the client-side collector.

Inference preserves the reference loader's post-load BF16 conversion, including its FP32 vision/norm roundtrip; post-training loads unrounded FP32 parameters.

The history buffer is used as follows:

```python
from focus_vlwa.data.head_history import HeadHistoryBuffer
from focus_vlwa.inference import FocusVLWAPolicy

policy = FocusVLWAPolicy(
    "checkpoints/frozen/SELECTED_CHECKPOINT",
    asset_id="arx_x5_sim",
)
history = HeadHistoryBuffer()
# Call observe once per consecutive control step, including steps between replans.
history.observe(0, observation["images"]["cam_high"])
observation.update(history.pack(0))
result = policy.infer(observation)
# Call history.reset() at every new episode.
```

For XPolicyLab, the adapter in `policy/ME_Brain_1` observes every control step and maintains a separate history buffer for each environment.

## Post-training

Post-training has two stages: `joint` trains the action and world-model experts together; `frozen` loads a selected joint checkpoint and freezes the world-model expert. Both use 20 full head-history frames and start with a fresh optimizer.

### Data and environment

Install `.[train]` for LeRobot v3. For v2 datasets, install `.[train-v2]` in a separate environment. The v3 extra uses NumPy 2.2.6; the v2 extra uses NumPy 1.26.4. Do not install both readers together. Install the supplied Transformers patch in the training environment.

The reader accepts native action sequences or precomputed 50-step chunks. Datasets must provide aligned `s1`, `s1_mask`, `event_action`, and `event_action_mask`; frozen training also needs these WM inputs. Head history is queried from the original 25 FPS episode images. WM latents are not quantile-normalized. Set `FOCUS_VLWA_LEROBOT_REPOS` to override shard selection.

An initialization checkpoint must include `model_config.json` and normalization assets. Set `--tokenizer-path` when the tokenizer is stored separately.

### Joint and frozen stages

```bash
focus-vlwa post-train \
  --stage joint --dataset data/lerobot \
  --init-checkpoint checkpoints/posttrain_init \
  --norm-stats-dir checkpoints/posttrain_init/assets/arx_x5_sim \
  --output-dir checkpoints/joint --batch-size 288

focus-vlwa post-train \
  --stage frozen --dataset data/lerobot \
  --init-checkpoint checkpoints/joint/SELECTED_CHECKPOINT \
  --norm-stats-dir checkpoints/joint/SELECTED_CHECKPOINT/assets/arx_x5_sim \
  --output-dir checkpoints/frozen --batch-size 288
```

Replace `SELECTED_CHECKPOINT` with the directory of the joint checkpoint to load.


| Setting                                | Joint           | Frozen WM       |
| -------------------------------------- | --------------- | --------------- |
| Warmup                                 | 200             | 200             |
| Default training steps                 | 5000            | 10000           |
| Peak / final learning rate             | 2.5e-5 / 2.5e-6 | 2.5e-5 / 2.5e-6 |
| Cosine decay horizon                   | 30000           | 10000           |
| WM loss weights: total / event / state | 1 / 1 / 0.3     | 0 / 1 / 0.3     |

Joint decay spans 30000 steps, so stopping at 5000 does not reach the final learning rate. Use `--decay-steps` to change the schedule. WM loss weights can be configured independently.

The frozen stage freezes the world-model expert and its input/output projections, while preserving attention interactions and input gradients. It omits unused WM supervision outputs. Existing WM weights are preserved; `--init-world-model-from-action` explicitly initializes the WM expert and projections from the action expert. Missing WM weights otherwise cause a loading error.

Parameter storage uses mixed BF16/FP32 by default. `--parameter-precision float32` enables FP32 parameter updates with BF16 autocast.

### Multiple GPUs and resume

For multiple GPUs, replace `focus-vlwa post-train` with `torchrun --standalone --nproc-per-node=8 -m focus_vlwa.scripts.post_train`, keeping the stage arguments above. `--batch-size` is global and must be divisible by the number of processes.

`--num-workers` sets the DataLoader worker count per GPU process; increase it when video decoding stalls training, within the node's CPU and memory capacity. The 20 history frames are encoded in groups of four by default. Set `FOCUS_VLWA_HISTORY_CHUNK_SIZE` to a divisor of 20 to trade peak GPU memory for fewer vision-encoder calls; `1` restores single-frame encoding.

Checkpoints save model weights, optimizer state, model and training configurations, and normalization assets under their original asset ID. Resume an interrupted stage with `--resume-checkpoint checkpoints/joint/SELECTED_CHECKPOINT`; this restores the model, optimizer, and sample cursor. Stage changes use `--init-checkpoint` with a fresh optimizer.

If the global batch size changes, also provide `--resume-batch-change-step` and `--resume-old-batch-size`, referring to the step where the old sample cursor was anchored. Resume does not promise bitwise RNG continuation across restarts.

## License and attribution

Code is distributed under Apache License 2.0 with upstream notices retained.
Gemma-derived components are additionally subject to `LICENSE_GEMMA.txt`.
See `NOTICE` for the origins of reused implementations.

The foundation and flow-matching action-expert design reference [the pi0.5 paper](https://arxiv.org/abs/2504.16054).
Source reuse is described in `NOTICE`; the components above do not imply independent authorship of the foundation implementations.
