# Liber_0_lite

**Contributor:** LiberAI | **Paper:** Pending | **arXiv:** Pending | **Original code:** [liber0/](liber0/)

RoboDojo SFT and evaluation policy for `arx_x5` joint control.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Inference source is included. Use Python 3.10 and a CUDA 13-compatible NVIDIA
driver. Install into a dedicated environment; keep the simulator environment
separate. Component notices: [NOTICE.md](NOTICE.md).
`/path/to/env` must be an existing Python environment with `bin/python` and pip;
the installer installs packages but does not create that environment.

External dependencies:

| Dependency | Required for |
| --- | --- |
| Linux, NVIDIA GPU/driver, Python 3.10, PyTorch 2.11.0 (CUDA 13), torchvision 0.26.0 | Policy inference; installed Python package versions are pinned by `install.sh` and [requirements.txt](requirements.txt). |
| Checkpoint and base model assets | Downloaded separately into `checkpoint/` and `assets/`; requires network access. Set `LIBER0_MODEL_PATH` to the assets directory. |
| XPolicyLab and RoboDojo `env_cfg/` | Required for both debug and simulation. Place `XPolicyLab/` beside `env_cfg/`, including `arx_x5.yml` and `robot/_robot_info.json`. |
| RoboDojo simulator, scene/robot assets, Isaac Sim and its Conda environment | Simulation only; install separately using the benchmark's environment instructions. Not installed by this policy. |

The inference engine and backend source are bundled; no external runtime/source
checkout or FlashAttention installation is required.

Training data loading additionally uses PyArrow, TorchCodec and a compatible FFmpeg build. Python dependencies are included in `requirements.txt`.

```bash
cd XPolicyLab/policy/Liber_0_lite
bash install.sh /path/to/env
export PATH=/path/to/env/bin:$PATH
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
python download_checkpoint.py --destination ./checkpoint --assets-dir ./assets
export LIBER0_MODEL_PATH="$PWD/assets"
```

Checkpoint: [Liber0-Lite-Robodojo](https://huggingface.co/LiberAI/Liber0-Lite-Robodojo).

The downloader pins both repositories and verifies checkpoint checksums.
Downloads total approximately 53 GB. The downloader does not use proxy environment variables.

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

The checkpoint directory must contain `model.pt`, `config.yaml` and
`dataset_stats.json`.

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env /path/to/env

EVAL_ENV_TYPE=sim bash eval.sh RoboDojo stack_bowls "$PWD/checkpoint" \
  arx_x5 joint 0 0 0 /path/to/env RoboDojo
```

Repeat debug mode with `DEBUG_OBS_ENCODED=1` for encoded observations. Debug mode
accepts an environment path as argument 10; simulation uses the simulator's
Conda environment name. The parent workspace must contain RoboDojo `env_cfg/`;
see the shared installation guide.

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

Edit `deploy.yml` for settings without a launcher override. Launcher overrides
take precedence over YAML values. Use absolute asset and checkpoint paths;
relative checkpoint paths resolve against this policy directory. Retain the
default inference settings when evaluating the released checkpoint.

### Batch inference

Set `eval_batch: true` in `deploy.yml` to use the same commands with batched
observations. The default remains `false`. Image preprocessing is per item;
the vision encoder and denoising decoder run on the combined batch, including
different-length instructions. Batch size is determined by the active observations.

Each observation must carry a unique, stable `env_idx`. The policy preserves
each environment's first-frame cue and replan counter; outputs follow the requested
`env_idx_list` order. The seed remains `run_seed + per_environment_replan_count`.
`reset()` clears all state; use `reset_envs(env_idx_list)` before reusing selected
environment IDs for new episodes (WebSocket: `call(func_name="reset_envs", obs=ids)`).
Use one coordinating client per policy server.

Batch size 1 retains the single-item inference path. Larger batches can differ
numerically from separate BF16 calls; task success equivalence requires a separate
simulator evaluation. Batch inference uses more working memory; no automatic
serial fallback is applied on memory errors.
