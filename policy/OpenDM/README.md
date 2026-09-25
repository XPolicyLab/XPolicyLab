# OpenDM — RoboDojo memory SFT and evaluation

**Contributor:** DM0.5 team | **Technical blog:** [DM0.5](https://www.dexmal.com/blog/dm0.5) | **Original code:** [dexmal/opendm](https://github.com/dexmal/opendm)

**Initial model:** [Dexmal/DM05-MEM](https://huggingface.co/Dexmal/DM05-MEM) · **Released policy:** [Dexmal/DM05-MEM-Robodojo-Sim](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim)

This adapter supports data conversion, full supervised fine-tuning, resumable checkpoints, export and XPolicyLab evaluation for **RoboDojo / `arx_x5` / `joint`**. State and action are 14D absolute joint/gripper vectors. Memory uses 20 head-camera frames at 1 Hz; the model predicts 50 actions and evaluation executes 25 at 25 Hz.

`opendm/` contains source from [dexmal/opendm at `fbab441`](https://github.com/dexmal/opendm/tree/fbab441b63c789e5c37f7293e61fea9ed356c6c5) under the [Apache-2.0 license](opendm/LICENSE). Consult each public model repository for its model asset license.

Retraining is not guaranteed to reproduce the released weights or their benchmark results. Download the released policy to evaluate that checkpoint. See [DATA.md](DATA.md) for the dataset format and selection rules.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux x86-64, Python 3.10, a CUDA 12.8 compatible NVIDIA driver and the CUDA 12.8 toolkit (`nvcc`) for building FlashAttention. Keep this checkout at `<workspace>/XPolicyLab`; runtime robot configs live at `<workspace>/env_cfg/`, supplied by the RoboDojo workspace.

```bash
cd <workspace>/XPolicyLab/policy/OpenDM
bash install.sh
conda activate opendm
```

`install.sh` creates the `opendm` conda environment (`OPENDM_CONDA_ENV` overrides it), installs [requirements.lock](requirements.lock), then FlashAttention `2.8.3.post1`. The lock resolves the upstream dependencies plus the adapter dependencies with PyTorch `2.11.0+cu128`, torchvision `0.26.0+cu128`, Transformers `5.3.0` and Accelerate `1.14.0`. The simulator needs its own RoboDojo environment; it is not installed by this script.

For inference only, `OPENDM_INSTALL_FLASH_ATTN=0 bash install.sh` omits FlashAttention. Inference defaults to SDPA. The training recipe uses FlexAttention for the VLM, FlashAttention 2 for vision and SDPA for actions; `train.sh --sdpa` switches all attention to SDPA and does not require FlashAttention.

Hardware planning estimates; peak memory and throughput depend on the device and configuration:

| Use | Suggested resources |
| --- | --- |
| Released FP32 inference | One 48–80 GB GPU; weights alone occupy about 23.3 GB, plus activations/history/cache |
| Full recipe, microbatch 8 | One node with 8 × 80 GB GPUs, fast interconnect, at least 256 GB host RAM |
| Storage | Budget about 1 TB including data, downloads and 10 optimizer checkpoints; adjust checkpoint retention to available disk |

Fewer GPUs are supported through gradient accumulation, subject to memory capacity. Accumulation reduces the number of simultaneous samples; it does not shard the model on a single GPU. Exact peak memory and throughput still need measurement. An A100/H100 class GPU is appropriate for the training attention backends.

## Model assets

Model assets and datasets follow the latest version on each official repository’s `main` branch. Rerun the download commands to fetch updates.

```bash
bash download_checkpoint.sh policy  # complete released policy, approximately 23.3 GB weights
bash download_checkpoint.sh base    # initial DM05-MEM, approximately 11.7 GB BF16 weights
bash download_checkpoint.sh norm    # released RoboDojo joint normalization statistics
```

| Asset | Local directory |
| --- | --- |
| DM05-MEM-Robodojo-Sim | `checkpoints/DM05-MEM-Robodojo-Sim/` |
| DM05-MEM | `checkpoints/DM05-MEM/` |
| RoboDojo norm stats | `checkpoints/robodojo-norm/norm_stats.json` |

The base model's pretraining statistics describe a different action space. SFT always uses the released **14D RoboDojo joint** statistics, without recomputing them from the training data. The finished checkpoint includes these same statistics.

## Data Processing

Download the latest official joint+gripper LeRobot **v3.0 video** export from the dataset’s `main` branch, then convert it into OpenDM's video-backed dexdataset JSONL format:

```bash
bash download_checkpoint.sh data-v30
bash process_data.sh RoboDojo robodojo-mem arx_x5 joint \
  --source data/source/data/RoboDojo_lerobot_v30_video
```

LeRobot v2.1 is also supported: download `data-v21` and use `data/source/data/RoboDojo_lerobot_v21_video`. Use the official joint+gripper exports with `observation.state`, `action` and three `observation.images.*` views; do not use `RoboDojo_ee_*`. Conversion produces OpenDM JSONL records referencing the original videos.

All tasks except `dlc` are included by default. Selected episodes are used for training without a held-out split. Add `--tasks 3` to select source task index 3; `dlc` cannot be selected. Use `--output PATH` for a custom output directory and pass the same path to training with `--data-root PATH`.

Keep the source export in place: converted data references its videos. If the source or output directory moves, follow the [relocation instructions](DATA.md#output). See [DATA.md](DATA.md) for the full data format and selection rules.

## Training

Run from the activated policy environment:

```bash
bash train.sh RoboDojo robodojo-mem arx_x5 joint 0 0,1,2,3,4,5,6,7
```

Positional arguments are `bench_name ckpt_name env_cfg_type action_type seed gpu_ids`. `gpu_ids` is a comma-separated list on **one node**. Training output is saved to `checkpoints/RoboDojo-robodojo-mem-arx_x5-joint-0/`.

| Parameter | Adapter training recipe |
| --- | --- |
| Updates / warmup | 100,000 optimizer updates / 1,000 linear-warmup updates |
| Global batch | 1,024; `accumulation = 1024 / (GPU count × microbatch)` |
| Default microbatch | 8; on 8 GPUs accumulation is 16; on 4 GPUs it is 32 |
| Optimizer / learning rate | MuonAdamW; cosine decay from `5e-5` to `2.5e-5` |
| Distributed / precision | FSDP; FP32 parameters and action computation, BF16 VLM compute |
| Trainable model | Full SFT with frozen VLM embeddings |
| Action | 50-step chunks of 14D absolute joint/gripper targets |
| History | 20 prior head-camera frames at 1 FPS |
| Image processing | Current views: color jitter then square-pad/resize 448; history: square-pad/resize only |
| Checkpoints | Save every 10,000 updates; keep up to 100 |

Useful options: `--per-device-batch 4` (accumulation recalculated), `--data-root PATH`, `--base-model PATH`, `--norm-stats-root PATH`, `--save-steps N`, `--workers N`, `--dry-run`. A non-divisible global batch raises an error. `--global-batch` and `--steps` override the global batch size and training duration; they change the training recipe.

### Resume and export

Rerun the **same command and run name** to resume the latest `checkpoint-N`, including optimizer, scheduler, trainer and RNG state. Keep batch, world size, dataset and schedule settings unchanged for a continuation. `--stop-after N` deliberately saves and stops at optimizer update N without changing the scheduled total; omit it when resuming.

Checkpoints include model weights, configuration, processor/tokenizer files and `norm_stats.json`. Periodic checkpoints also retain the state needed to resume training. The final run directory can be evaluated directly, without merging or renaming files:

```bash
bash eval.sh RoboDojo cover_blocks robodojo-mem arx_x5 joint 0 0 1 opendm robodojo
```

## Evaluation

```bash
bash eval.sh RoboDojo cover_blocks DM05-MEM-Robodojo-Sim arx_x5 joint 0 0 1 opendm robodojo
```

The 10 arguments follow XPolicyLab: `bench task checkpoint robot action_type seed policy_gpu env_gpu policy_env eval_env`. `ckpt_name` can also be a checkpoint path; `MODEL_PATH` explicitly selects one. The adapter loads a complete run export, or the newest periodic checkpoint if no run export is present. Use the matching processor and `norm_stats.json` together.

Runtime defaults are in [deploy.yml](deploy.yml). Inference uses head, left-wrist and right-wrist RGB views, absolute joint actions, unclipped quantile normalization and FP32 action integration.

`OPENDM_DEPLOY_CONFIG` selects an alternative YAML file; relative paths resolve from this policy directory. Environment variables `MODEL_PATH`, `NORM_STATS_PATH`, `OPENDM_ACTION_STEPS`, `OPENDM_DIFFUSION_STEPS` and `OPENDM_DIFFUSION_NOISE_SEED` override the corresponding model path, statistics path, execution length, denoising steps and noise seed. Keep the defaults when comparing published metrics.

History is always enabled for this memory checkpoint. Model-specific keys in [deploy.yml](deploy.yml) are:

| Key | Default | Meaning / constraint |
| --- | --- | --- |
| `model_path` | `null` | Explicit checkpoint directory; otherwise use `ckpt_name`. Relative paths resolve from this policy directory. |
| `norm_stats_path` | `null` | Normalization JSON; defaults to `norm_stats.json` inside the selected checkpoint. Relative paths resolve from this policy directory. |
| `experiment_path` | `scripts/robodojo_dm05_history.py` | Python file defining `DM05Exp`; relative to this policy directory. It must define a history-enabled experiment. |
| `model_action_mode` | `absolute` | Absolute joint targets; other values are rejected for this checkpoint. |
| `action_chunk_size` | `50` | Number of actions predicted per chunk; keep aligned with the checkpoint. |
| `action_steps` | `25` | Actions executed before replanning; must be between 1 and `action_chunk_size`. |
| `history_image_key` | `images_1` | History camera: `images_1` is head, `images_2` left wrist, `images_3` right wrist. Use head for this checkpoint. |
| `history_slots` | `20` | Maximum number of prior images; must be positive. Missing history is padded. |
| `history_fps` | `1.0` | Requested history sampling rate; used only when `history_action_interval` is unset. |
| `runtime_fps` | `25.0` | Action frequency used to calculate history spacing; must match the environment. It does not set the simulator's control rate. |
| `history_action_interval` | `25` | Action steps between stored history frames; overrides `history_fps`. Set to `null` to derive it from `runtime_fps / history_fps`, rounded to an integer. |
| `history_tokens_per_slot` | `16` | Tokens per history image; must match the upstream model constant (16). |
| `diffusion_steps` | `10` | Number of action denoising iterations. |
| `diffusion_noise_seed` | `null` | Optional fixed noise seed; `null` uses normal stochastic inference. |
| `prompt` | `null` | Fallback instruction when an observation has no instruction; otherwise falls back to `task_name`, then a generic instruction. |
| `robot_type` | `null` | Robot label in the language prompt; `null` resolves to `Dual ARX5`. This does not select robot dimensions. |
| `control_mode` | `null` | Optional control-mode text in the language prompt; `null` uses the experiment default (omitted for this adapter). It does not change the joint-action interface. |
| `speed` | `"0.5"` | Speed text in the language prompt; does not rescale actions or set the simulator's control rate. |
| `add_state` | `true` | Include quantized state values in the language prompt. |
| `n_bins` | `256` | Number of bins for quantizing normalized state values. |
| `model_max_length` | `1536` | Maximum input token length; long task instructions may be shortened to fit. |
| `bf16` | `false` | Controls upstream loading/attention selection. The fixed `fp32_mixed` precision policy retains FP32 parameters/action computation and BF16 VLM autocast. |
| `llm_attn_implementation` | `sdpa` | VLM language attention backend. |
| `vision_attn_implementation` | `sdpa` | Vision attention backend. |
| `action_attn_implementation` | `sdpa` | Action-expert attention backend. |
| `liger_kernel` | `true` | Apply the upstream Liger kernels when CUDA is available. |

The effective history rate is `runtime_fps / history_action_interval` (25 / 25 = 1 Hz by default). History settings and prompt fields should match those used for training.
