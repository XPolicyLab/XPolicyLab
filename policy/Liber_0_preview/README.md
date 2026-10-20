# Liber_0_preview

**Contributor:** LiberAI | **Paper:** Pending | **arXiv:** Pending | **Original code:** Included

RoboDojo SFT and evaluation policy for `arx_x5` joint control.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requires Linux, Python 3.10, a CUDA 13-compatible GPU/driver and FlashAttention 2.8.4. Building FlashAttention may require a matching CUDA toolkit and compiler. Keep the simulator environment separate. This adapter reuses the included `Liber_0_lite` runtime; retain both policy directories. Third-party notices remain in [Liber_0_lite/NOTICE.md](../Liber_0_lite/NOTICE.md).

For training only, the environment installed by `../Liber_0_lite/install.sh` is sufficient. Data loading uses PyArrow, TorchCodec and a compatible FFmpeg build.

```bash
cd XPolicyLab/policy/Liber_0_preview
bash install.sh /path/to/env
export PATH=/path/to/env/bin:$PATH
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python download_checkpoint.py --destination ./checkpoint --assets-dir ./assets
export LIBER0_MODEL_PATH="$PWD/assets"
export LIBER0_CONTEXT_PATH="$PWD/checkpoint/context"
```

Checkpoint: [Liber0-Preview-Robodojo](https://huggingface.co/LiberAI/Liber0-Preview-Robodojo).

The downloader retrieves and verifies the complete checkpoint, including `context/`, at revision `2773a04e3b8f09096ec7beb6105bd6ed80c91da8`. No separate Lite checkpoint download is needed. Base model assets are downloaded separately at a pinned revision.

## Data Processing

Use LeRobot v2.1 video data prepared with the [official converter](../../scripts/transform_lerobot_v21_format.py). Camera, state and action keys match that converter. This entrypoint checks the prepared data; it does not convert raw trajectories, select an episode subset or regenerate normalization statistics.

```bash
export DATASET_ROOT=/path/to/robodojo_lerobot_v21
python ../Liber_0_lite/download_checkpoint.py --pretrained \
  --destination ./pretrained --assets-dir ./assets
bash process_data.sh RoboDojo sft arx_x5 joint \
  --dataset "$DATASET_ROOT" --checkpoint ./pretrained
```

The action-space contract is `arx_x5` joint control. Select subsets with the converter before running this preflight; the optional standard `expert_data_num` argument is not supported here.

## Training

This entrypoint trains the main policy only.

RoboDojo SFT uses the shared [training implementation](../Liber_0_lite/liber0/training/) and [configuration](../Liber_0_lite/configs/robodojo_sft.yaml). Inputs and targets use the 288×384 four-grid layout; action windows contain 32 steps. Pretraining initialization and continued fine-tuning are both supported.

Set `GPU_IDS` to the comma-separated allocated devices, and set `MICRO_BATCH`, `ACCUMULATION` and `NUM_WORKERS` for your hardware. The effective batch size must match `global_batch_size` in the config.

```bash
bash train.sh RoboDojo sft arx_x5 joint 42 "$GPU_IDS" \
  --dataset "$DATASET_ROOT" --checkpoint ./pretrained \
  --init-weights ./pretrained/pretrain/model.pt --assets ./assets \
  --batch-size "$MICRO_BATCH" --accumulation "$ACCUMULATION" \
  --num-workers "$NUM_WORKERS"
```

The six positional arguments follow the official training convention. Optional arguments are `--config`, `--init-weights`, `--resume`, `--stop-after` and `--dry-run`; input paths resolve relative to this policy directory. To continue fine-tuning a task checkpoint, pass its directory through `--checkpoint` and omit `--init-weights`. For multiple nodes, also set `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`.

Each policy writes its own `checkpoints/RoboDojo-sft-arx_x5-joint-42/step_XXXXXX/` directory. Add `--resume ./checkpoints/RoboDojo-sft-arx_x5-joint-42/step_005000` to the same command to restore a completed training state. `--dry-run` checks the launch configuration without starting training; `--stop-after N` runs to update N without changing the learning-rate schedule.

For evaluation, pass the completed step directory as `ckpt_name`. It contains `model.pt`, `config.yaml` and `dataset_stats.json`; optimizer state and training metadata are not inference dependencies. Keep any separately required evaluation assets configured as in the Evaluation section.

Full-size CUDA training qualification for this integrated entrypoint is pending.

## Evaluation

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env /path/to/env
EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env RoboDojo
```

The checkpoint directory contains `model.pt`, `config.yaml` and `dataset_stats.json`. RoboDojo simulator/assets and sibling `env_cfg/` are separate dependencies. Keep `eval_batch: false`; batched evaluation is not supported. Use `DEBUG_OBS_ENCODED=1` to test encoded observations.

### Deployment configuration

Policy-specific settings in [deploy.yml](deploy.yml):

| Key | Default | Meaning / override |
| --- | --- | --- |
| `device` | `cuda` | Uses the GPU selected by the evaluation command's `policy_gpu_id`. |
| `model_assets_path` | `null` | Base model assets directory; the launcher requires and supplies `LIBER0_MODEL_PATH`. |
| `checkpoint_path` | `null` | Optional explicit checkpoint directory, checked before `ckpt_name`. Normally leave unset and pass the directory as `ckpt_name`. |
| `weights_file` | `model.pt` | Weight filename inside the checkpoint directory; the launcher supplies `LIBER0_WEIGHTS_FILE`, defaulting to `model.pt`. |
| `replan_steps` | `16` | Number of actions returned per request; must be between 1 and the trained horizon of 32. |
| `num_inference_steps` | `10` | Number of denoising steps per action prediction. |
| `ws_ping_interval_s` | `null` | WebSocket keepalive interval in seconds; `null` disables periodic pings. |
| `ws_ping_timeout_s` | `null` | WebSocket pong timeout in seconds; `null` disables the timeout. |
| `context_model_path` | `null` | Additional model assets under `checkpoint/context`; the launcher requires and supplies `LIBER0_CONTEXT_PATH`. |

Edit `deploy.yml` for settings without a launcher override. Launcher overrides
take precedence over YAML values. Use absolute asset and checkpoint paths;
relative checkpoint paths resolve against this policy directory. Retain the
default inference settings when evaluating the released checkpoint.
