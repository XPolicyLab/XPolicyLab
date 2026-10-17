# Rex_M1_preview

**Contributor:** Futian Lab & VisIncept | **Paper:** Not released | **arXiv:** Not released | **Original code:** See vendored `rex_m1/`.

`Rex_M1_preview` adapts the Rex-M1-preview policy to XPolicyLab/RoboDojo for training and evaluation. Integration scripts live at this directory level; the vendored implementation lives in `rex_m1/`. It supports RoboDojo simulation with Dual ARX5 (`arx_x5`) and `action_type=ee`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` for the manual setup, checkpoint layout, integrity hashes, and smoke checks. The recommended installer creates a conda environment with the policy dependencies:

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash install.sh
conda activate rex_m1_preview
```

Set `REX_M1_PREVIEW_CONDA_ENV` before running `install.sh` to use a different environment name.

## Data Processing

`process_data.sh` converts RoboDojo HDF5 episodes into the `rex_m1` JSON/MP4 training format, then precomputes the head-camera thumbnails the dataloader reads as history frames:

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash process_data.sh <bench_name> <task_names|all> <env_cfg_type> <action_type> [expert_data_num]

# Example: convert every arx_x5 simulation task
bash process_data.sh RoboDojo all arx_x5 ee
```

| Artifact | Path |
| --- | --- |
| Episode JSON | `rex_m1/data/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>/data/chunk-XXXX/episode_*.json` |
| Videos | the sibling `videos/` directory, three MP4s per episode (ego / wrist-left / wrist-right) |
| History thumbnails | the sibling `thumbs/` directory, one `.ego400s5.npy` per episode |

The training recipe is the checked-in `rex_m1/configs/data/robodojo_all_arx_x5.yaml`: the history settings plus normalization statistics computed by this script on the full current RoboDojo release. Converting `all` keeps that file. Any other task selection writes a new config with recomputed statistics; copy the `history_frames` through `ptp_steps` keys from the recipe into it, or the dataloader refuses to start. Optional overrides: `RAW_DATA_ROOT`, `OUTPUT_DIR`, `DATA_WORKERS`, `ACTION_LENGTH` (30), `BATCH_SIZE` (8), `HISTORY_RES` (400) and `THUMBS_STRIDE` (5), the last two matching `history_res` and `thumbs_stride` in the data config.

## Training

Training starts from the public [Xiaomi-Robotics-1-5B](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) weights. The repository ships `model_states.pt`, already in the format the training entry point loads:

```bash
cd XPolicyLab/policy/Rex_M1_preview
hf download XiaomiRobotics/Xiaomi-Robotics-1-5B model_states.pt --local-dir checkpoints/pretrained_ckpt
```

`train.sh` wraps the vendored launcher (`rex_m1/scripts/train.sh`, DeepSpeed via Lightning):

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [hydra_overrides...]

# Example: the reference recipe, on each of 4 nodes with 8 GPUs
WORLD_SIZE=4 RANK=<node_rank> MASTER_ADDR=<node0_ip> MASTER_PORT=29500 \
  bash train.sh RoboDojo all arx_x5 ee 42 0,1,2,3,4,5,6,7
```

The reference recipe is 60000 steps at a global batch of 256 (4 nodes x 8 GPUs x `batch_size` 8), AdamW (DeepSpeed FusedAdam, betas 0.9/0.95, weight decay 0.1) with a cosine schedule from a peak learning rate of 1e-5 down to 5e-7 after 500 warmup steps, bf16 mixed precision. With fewer GPUs, keep the global batch through `trainer.accumulate_grad_batches`. History frames: 20 head-camera frames, one every 25 steps, each pooled to 16 tokens; their vision path receives no gradient and runs under `no_grad`.

`bench_name`, `ckpt_name`, `env_cfg_type`, and `action_type` must match the `process_data.sh` call. Training writes `config.py` and `last.ckpt/` under `train_runs/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, and `train.sh` links `checkpoints/<same name>/` to it, so that name is what `eval.sh` takes as `ckpt_name`.

| Variable | Notes |
| --- | --- |
| `PRETRAINED_PATH` | Xiaomi-Robotics-1-5B `model_states.pt`; default `checkpoints/pretrained_ckpt/model_states.pt`. |
| `OUTPUT_DIR`, `DATA_CONFIG_NAME` | Converted dataset root and data config basename; must match `process_data.sh`. |
| `MAX_STEPS`, `SAVE_INTERVAL` | Step budget (60000) and checkpoint interval (20000). |
| `MAX_LENGTH` | Per-sample token budget of the collate; default `20000`. Longer samples are dropped. |
| `WANDB_MODE` | `offline` by default; `online` uploads after `wandb login`. |

## Model Assets

Place the released checkpoint under `checkpoints/<ckpt_name>/`, or set `model_dir` in `deploy.yml` to an existing checkpoint directory. The directory must contain `config.py` and `last.ckpt/`; `config.py` carries the history settings used at inference.

```text
policy/Rex_M1_preview/checkpoints/Rex_M1_preview/
├── config.py
└── last.ckpt/
```

See `INSTALLATION.md` for the complete checkpoint preparation and verification commands.

## Evaluation

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the released checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls Rex_M1_preview arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check; leave it unset or set it to `sim` for RoboDojo simulation. For split-machine deployment, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `action_type` | Must be `ee`. |
| `model_dir` | Optional checkpoint directory; takes precedence over `ckpt_name`. |
| `action_length` | Number of leading actions executed before replanning; `0` executes the full predicted chunk. |
| `vlm_processor_path` | Hugging Face repository ID or local path for the Qwen3-VL processor. |
| `image_factor` | Camera frames are resized to a multiple of this before tokenization. |
| `image_max_pixels` | Upper bound on pixels per current-view frame after resizing. |
| `default_prompt` | Instruction used when the environment supplies none. |

History parameters are read from the checkpoint's `config.py` so training and inference settings remain aligned.

## Acknowledgements

Rex-M1-preview builds on the [codebase](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) and
[pretrained weights](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) released by the
Xiaomi Robotics team. We sincerely thank the team for making these resources publicly available and
for their valuable contributions to the community.

## Notes

- **One policy server per evaluation.** This adapter keeps a per-environment history buffer keyed on
  `obs["env_idx"]` alone, so two concurrent evaluations pointed at the same server would interleave
  their histories without any error being raised.
