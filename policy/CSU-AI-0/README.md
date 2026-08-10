# CSU-AI-0

**Contributor:** CSU-AI Team | **Paper:** Spatial Forcing / OpenPI integration | **arXiv:** Not applicable | **Original code:** [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)

`CSU-AI-0` is a single-model Spatial Forcing policy for RoboDojo. It installs its own pinned OpenPI/Spatial Forcing source tree and uv environment under this policy directory, then loads a new Orbax checkpoint trained for `arx_x5` joint actions; the previous multi-expert QRouter submission is not part of this adapter.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

The adapter installs an independent runtime under `policy/CSU-AI-0/open_sf`. The installer sparsely fetches the required OpenPI/Spatial Forcing source from the official XPolicyLab repository at pinned revision `2dfd4ee7af7ecf8b3281847179f14d22c5c04a35`, creates `open_sf/.venv`, and installs the current XPolicyLab checkout into that environment:

```bash
cd XPolicyLab/policy/CSU-AI-0
bash install.sh
bash download_checkpoints.sh
```

The generated `open_sf/` source tree and `.venv` are ignored by Git and are not part of the submitted policy package. Set `CSU_AI_0_RUNTIME_REPO` or `CSU_AI_0_RUNTIME_REVISION` only when testing an explicit runtime mirror or revision. If `uv` is installed outside `PATH`, set `CSU_AI_0_UV_BIN=/absolute/path/to/uv`.

The download is resumable and verifies all 23 checkpoint files against `CSU-AI-0.sha256`. The expected local path is:

```text
policy/CSU-AI-0/checkpoints/CSU-AI-0
```

Public source: [ShaoRun/vla_ro/checkpoints/CSU-AI-0](https://huggingface.co/datasets/ShaoRun/vla_ro/tree/main/checkpoints/CSU-AI-0). Set `CSU_AI_0_HUGGINGFACE_REPO=<namespace>/<dataset>` to test a mirror without editing the script.

## Data Processing

The training configuration consumes the RoboDojo LeRobot v2.1 dataset and computes `state`/`actions` normalization statistics with the upstream Spatial Forcing implementation. The entry point follows the shared XPolicyLab positional arguments and forwards any trailing arguments to `compute_norm_stats.py`:

```bash
# Command template
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  [norm_stats_args...]

# Runnable example
cd XPolicyLab/policy/CSU-AI-0
HF_LEROBOT_HOME=/path/to/lerobot-data \
PI05SF_ASSETS_DIR=/path/to/output-assets \
bash process_data.sh RoboDojo CSU-AI-0 arx_x5 joint \
  --max-frames 8192 --num-workers 8
```

Omit `--max-frames` for full-dataset statistics. The generated asset directory must contain `RoboDojo_lerobot_v21_video/norm_stats.json` before training.

## Training

Training uses the upstream `pi05sf_jax_robodojo_v21_offcache` recipe. The wrapper maps `seed` and `gpu_id` to the upstream trainer and forwards trailing arguments to `train_align.py`. Prepare the Pi0.5 base checkpoint, VGGT weights, RoboDojo dataset, and Spatial Forcing feature cache, then run:

```bash
# Command template
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <gpu_id> [trainer_args...]

# Runnable example
cd XPolicyLab/policy/CSU-AI-0
HF_LEROBOT_HOME=/path/to/lerobot-data \
PI05_BASE_PATH=/path/to/pi05-base \
VGGT_WEIGHT_PATH=/path/to/VGGT-1B \
SF_CACHE_DIR=/path/to/spatial-forcing-cache \
EXP_NAME=csu_ai_0_reproduction \
bash train.sh RoboDojo CSU-AI-0 arx_x5 joint 0 0
```

`PI05_BASE_PATH` must contain `params/` and `assets/`; `VGGT_WEIGHT_PATH` must contain `model.pt`. The cache is read-only during training. If `EXP_NAME` is omitted, the wrapper derives a deterministic name from the shared arguments. Additional OpenPI configuration overrides may be appended after `gpu_id`.

## Evaluation

```bash
cd XPolicyLab/policy/CSU-AI-0

# Offline closed-loop wiring check.
EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo stack_bowls CSU-AI-0 arx_x5 joint 0 0 0 uv base

# Simulator evaluation.
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo stack_bowls CSU-AI-0 arx_x5 joint 0 0 0 uv RoboDojo
```

The full entry point is:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>
```

`ckpt_name` may be `CSU-AI-0`, an absolute checkpoint path, or another directory under `policy/CSU-AI-0/checkpoints/`. The policy-side environment argument is `uv`; an explicit Spatial Forcing/OpenPI project directory is also accepted.

## Model Assets

The public model repository contains one Orbax OCDBT checkpoint:

```text
CSU-AI-0/
├── _CHECKPOINT_METADATA
├── assets/RoboDojo_lerobot_v21_video/norm_stats.json
└── params/
```

Large checkpoint files are never committed to Git. Verify an existing local copy independently with:

```bash
python3 verify_checkpoints.py
```

## Configuration

- Policy: `CSU-AI-0`
- Model family: `Spatial_Forcing`
- Policy runtime: independent `policy/CSU-AI-0/open_sf/.venv`
- Benchmark: `RoboDojo`
- Environment configuration: `arx_x5`
- Action type: `joint`
- OpenPI train config: `pi05sf_jax_robodojo_v21_offcache`
- Norm-stat asset id: `RoboDojo_lerobot_v21_video`
- Checkpoint format: Orbax OCDBT, 23 files
- Protocol: WebSocket (`ws`)

## Notes

- `model.py` uses the standard XPolicyLab `ModelTemplate` contract and obtains robot action dimensions from `get_robot_action_dim_info`.
- Observation images arrive already decoded by the policy server and remain RGB; this adapter only normalizes shape and dtype.
- The checked-in checksum manifest covers every file in the published checkpoint.
