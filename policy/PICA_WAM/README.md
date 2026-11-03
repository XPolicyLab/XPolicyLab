# PICA_WAM

**Contributor:** Yude Zou | **Paper:** PICA-WAM: Visual Perturbation Invariance via Cross-Appearance Self-Alignment in World Action Models | **arXiv:** [2610.26813](https://arxiv.org/abs/2610.26813) | **Original code:** [YudeZou/PICA-WAM-release](https://github.com/YudeZou/PICA-WAM-release)

PICA-WAM makes a World Action Model invariant to visual perturbations through cross-appearance self-alignment: an EMA teacher reads the nominal observation, and the model aligns its action features on a perturbed view to the teacher's. This adapter releases the training and inference code for the RoboTwin 2.0 checkpoint, built on the InternW0-Δ backbone and trained on clean demonstrations only. `internw0_pica/` is adapted from [InternW0-Delta](https://github.com/InternRobotics/InternW0-Delta) `90801ba`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Linux, Python 3.11, CUDA 12.8:

```bash
cd policy/PICA_WAM
bash install.sh pica-wam
conda activate pica-wam
```

`install.sh` pins PyTorch 2.10.0 (cu128) and installs `internw0_pica/` in editable mode with its `[train,modelscope]` extras: Transformers 5.13.0 and DeepSpeed 0.19.2. It also installs XPolicyLab in editable mode, plus a conda-local gcc/g++ (for Triton) and FFmpeg (for `torchcodec`).

The released checkpoint was trained and evaluated without the optional `flash-linear-attention` / `causal-conv1d` kernels, so RynnBrain runs its PyTorch path. `install.sh` leaves those kernels out.

## Data Processing

Only `bench_name=RoboTwin`, `env_cfg_type=aloha_agilex`, `action_type=joint`. The training data is the official RoboTwin 2.0 clean demonstrations ([`aloha-agilex_clean_50`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0), 50 tasks × 50, 50 Hz).

```bash
cd policy/PICA_WAM
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [gpu_ids]

# Writes data/RoboTwin-clean50-aloha_agilex-joint/ and its text-embedding cache
bash process_data.sh RoboTwin clean50 aloha_agilex joint 0
```

The script runs two steps:

1. `internw0_pica/tools/prepare_robotwin_data.py` downloads these demonstrations in LeRobot v3.0 form into `data/robotwin2.0-lerobot-v3.0/` and checks the size and checksum of every file. It then rewrites the export as the **LeRobot v2.0** layout the vendored loader reads: one parquet per episode, `meta/episodes.jsonl` and `meta/tasks.jsonl`, with videos symlinked rather than re-encoded.
2. `tools/text_cache.py task=robotwin_pica` encodes every instruction with the local Wan2.2 UMT5 encoder from `download_assets.sh`, into `.cache/text_embed/<bench>-<ckpt>-<env>-<action>/`. The Wan directory must be named `Wan2.2-TI2V-5B`.

The keys match the official converters ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)): `observation.state` and `action` (14D, left arm + gripper, right arm + gripper), and `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` (480×640 video). No XPolicyLab trajectory is decoded here.

| Variable | Effect |
| --- | --- |
| `PICA_WAM_SOURCE_DIR` | v3.0 download directory |
| `PICA_WAM_DATASET_DIR` | v2.0 export directory; an existing complete export is reused |
| `PICA_WAM_TEXT_CACHE` | text-embedding cache directory |
| `PICA_WAM_SKIP_DOWNLOAD=1` | reuse an already verified download |
| `HF_ENDPOINT` | Hugging Face mirror |

## Training

Post-training starts from [`InternW0-Delta-Base`](https://huggingface.co/InternRobotics/InternW0-Delta-Base) (`checkpoints/pretrain.pt`, fetched by `download_assets.sh --train`) with the vendored `robotwin_pica` task.

The run lands in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, with `config.yaml`, `dataset_stats.json` and `checkpoints/weights/slot_*.pt` plus `weights_manifest.json`.

```bash
cd policy/PICA_WAM
bash download_assets.sh --train
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [num_gpus]

# One 8-GPU node; seed 42 is the released run's seed
bash train.sh RoboTwin clean50 aloha_agilex joint 42 0,1,2,3,4,5,6,7
```

| Variable | Effect |
| --- | --- |
| `PICA_WAM_GLOBAL_BATCH` | global batch (1024, the released run's), kept through gradient accumulation |
| `PICA_WAM_MICRO_BATCH` | per-GPU batch (32) |
| `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` | multi-node launch; run the same command on every node |
| `WAM_PRETRAIN_CHECKPOINT` | initialization weights |
| `PICA_WAM_DATASET_DIR`, `PICA_WAM_TEXT_CACHE` | as in data processing |
| `PICA_WAM_TRAIN_OVERRIDES` | extra Hydra overrides, space separated, e.g. `max_steps=100 save_every=50` |

## Evaluation

Run from a RoboTwin 2.0 checkout whose `XPolicyLab/` contains this adapter. Both scene settings (`demo_clean`, `demo_randomized`) use the same checkpoint, with 100 episodes per task, scene seeds from 100000 and unseen instructions.

```bash
cd /path/to/RoboTwin/XPolicyLab/policy/PICA_WAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Released weights: checkpoints/robotwin.pt (no matching run directory)
bash download_checkpoint.sh
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin click_bell robotwin aloha_agilex joint 0 0 0 pica-wam <robotwin_env>

# The training run above
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin click_bell clean50 aloha_agilex joint 42 0 0 pica-wam <robotwin_env>

# All 50 tasks, once per setting
cd /path/to/RoboTwin
for setting in demo_clean demo_randomized; do
  bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml \
    --policy-name PICA_WAM --ckpt-name robotwin --bench-name RoboTwin \
    --env-cfg-type aloha_agilex --policy-conda-env pica-wam \
    --eval-env-conda-env <robotwin_env> --task-config "$setting" \
    --test-num 100 --instruction-type unseen --seed 0 --num-workers 1
done
```

Inside a run directory the server loads the `latest` slot of `weights_manifest.json` (else the newest slot) together with the run's own `config.yaml`. Without a run directory it uses `checkpoints/robotwin.pt` and composes the vendored `robotwin_pica` task. Either way, normalization uses `internw0_pica/assets/stats/robotwin.json`, the training statistics.

Keep `--num-workers 1` (the default): the policy keeps per-environment state, and RoboTwin's multi-worker client does not tag `reset` / `get_action` with the environment.

## Model Assets

Weights are not in Git. Paths are relative to this directory:

| Artifact | Source | Local path |
| --- | --- | --- |
| Wan2.2 VAE, UMT5, tokenizer | `Wan-AI/Wan2.2-TI2V-5B` on ModelScope | `assets/Wan-AI/Wan2.2-TI2V-5B/` |
| RynnBrain | `Alibaba-DAMO-Academy/RynnBrain1.1-2B` on ModelScope | `assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B/` |
| PICA-WAM RoboTwin checkpoint | [xdzouyd/PICA-WAM](https://huggingface.co/xdzouyd/PICA-WAM), `internw0_delta/robotwin.pt` | `checkpoints/robotwin.pt` |
| Training initialization (training only) | [InternRobotics/InternW0-Delta-Base](https://huggingface.co/InternRobotics/InternW0-Delta-Base), `pretrain.pt` | `checkpoints/pretrain.pt` |

```bash
conda activate pica-wam
bash download_assets.sh            # add --train for pretrain.pt; or pass a local asset root
bash download_checkpoint.sh        # or a local file / HTTPS URL; SHA256 is checked
```

Wan2.2 downloads only the VAE, UMT5 encoder and tokenizer, because the checkpoint already contains the video expert. Review the upstream licenses before downloading.

## Configuration

Adapter-specific `deploy.yml` keys, resolved from this directory:

- Paths:
  - `checkpoint_path`, `base_model_dir`, `vlm_model_path`.
  - `wam_root`: the vendored source.
  - `dataset_stats_path`: `null` uses the training statistics.
  - `train_config_path`: `null` uses the run's `config.yaml`, else the vendored task.
- Inference contract (keep the defaults to match the reported result): `device`, `mixed_precision`, `action_horizon`, `replan_steps`, `num_inference_steps`, `action_hz`, `sigma_shift`, `rand_device`, `tiled`.
- Other keys: `timing_enabled`, `default_instruction` (used when an observation has no instruction), `allow_dummy_policy`.

`setup_eval_policy_server.sh` environment overrides:

| Variable | Override |
| --- | --- |
| `WAM_CHECKPOINT_PATH` | weight file, instead of `ckpt_name` resolution |
| `WAM_TRAIN_CONFIG_PATH` | training `config.yaml` to rebuild the model from |
| `WAM_DATASET_STATS_PATH` | z-score statistics JSON |
| `WAM_WAN22_PATH` | local Wan2.2 directory |
| `WAM_RYNNBRAIN_PATH` | local RynnBrain directory |
| `WAM_ALLOW_DUMMY_POLICY=true` | debug wiring only; skips real weights |

## Notes

- **Inference path.** It follows the paper's RoboTwin evaluation:
  - A 32-step action horizon, replanned after 10 steps, with 10 denoising steps.
  - Cameras are bilinearly resized into a 384×256 T-shaped canvas: head on top, wrists below.
  - State is z-scored with the training statistics.
  - Gripper commands are passed through unclipped.
  - The alignment head and EMA teacher are training-only, and inference never builds them.
- **Statefulness.** `eval_batch: true` keeps one session per `env_idx` (memory frames, pending actions, step counter) and replans each environment through the single-sample path. Batched observations must carry `env_idx`.
- **Naming.** `ckpt_name=robotwin` selects the release file, not a training run.
- **Reported result.** Paper, RoboTwin 2.0, 50 tasks × 100 episodes: 90.68% clean, 74.70% randomized.
