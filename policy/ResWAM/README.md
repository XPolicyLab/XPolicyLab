# ResWAM

**Contributor:** OPTML Group, Michigan State University | **Paper:** ResWAM: Learning What Changes, Not What Stays, for World-Action Models | **arXiv:** [2611.19559](https://arxiv.org/abs/2611.19559) | **Original code:** [OPTML-Group/ResWAM](https://github.com/OPTML-Group/ResWAM)

ResWAM is a 2.3B-parameter residual world-action model: a frozen RynnBrain-1.1-2B backbone with a 0.5B dual-branch DiT that jointly predicts action chunks and the DINOv3 feature residual of the future frame via conditional flow matching. This adapter serves the released RoboDojo checkpoint (one model for all tasks, `arx_x5`, absolute joint targets, 32-step prediction with the first 10 steps executed). `install.sh` clones the upstream repository at a pinned commit into `policy/ResWAM/ResWAM/`; nothing is vendored.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Clones ResWAM at commit `76a9cfe` and builds its Python 3.10 environment (PyTorch 2.6.0 / CUDA 12.4, transformers 5.2.0) with [uv](https://docs.astral.sh/uv/), then installs XPolicyLab into it:

```bash
cd XPolicyLab/policy/ResWAM
bash install.sh [venv_dir]          # default: policy/ResWAM/.venv
```

The resulting venv path is the `<policy_conda_env>` argument of `eval.sh` (a conda env name also works). RynnBrain-1.1-2B and DINOv3-ViT-B/16 are fetched from Hugging Face at pinned revisions on first use; set `HF_HOME` to share the cache.

## Model Assets

Released checkpoint (stage-2 step 328k, the model behind the paper and leaderboard numbers), downloaded into `checkpoints/ResWAM-RoboDojo/` so that `ckpt_name=ResWAM-RoboDojo` resolves:

```bash
bash download_checkpoint.sh policy/ResWAM/.venv/bin/python
```

## Data Processing

ResWAM trains on the official RoboDojo LeRobot export (`RoboDojo-Benchmark/RoboDojo` @ `35efbc7`, `data/RoboDojo_lerobot_v21_video`, 35 training tasks / 3,500 episodes) and reads its keys as-is; `process_data.sh` only writes ResWAM's dataset contract and normalization statistics on top of it:

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [policy_env]

# Example (data under $RESWAM_DATA_ROOT/robodojo, default policy/ResWAM/data/robodojo)
bash process_data.sh RoboDojo reswam arx_x5 joint policy/ResWAM/.venv
```

## Training

Two stages, as in the paper: stage 1 is warmup plus cosine decay to 200k steps; stage 2 resumes from the stage-1 step-200k state and trains to 400k at the floor learning rate. The released checkpoint is stage-2 step 328k.

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [policy_env]

# Example: both stages on 4 GPUs
bash train.sh RoboDojo reswam arx_x5 joint 0 0,1,2,3 policy/ResWAM/.venv
```

Runs are written to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/{robodojo,robodojo_stage2}/`. `RESWAM_STAGE=1|2|all` selects the stage(s). `<seed>` only names the run; the training seed stays at the paper value from `configs/robodojo.yaml`.

## Evaluation

```bash
cd XPolicyLab/policy/ResWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: released checkpoint on stack_bowls, seed 0
bash eval.sh RoboDojo stack_bowls ResWAM-RoboDojo arx_x5 joint 0 0 0 $(pwd)/.venv RoboDojo
```

The policy server needs about 7 GB of GPU memory (batch of 8 environments) next to the simulator.

## Configuration

`deploy.yml` keys beyond the standard set:

| Key | Default | Meaning |
| --- | --- | --- |
| `execute_horizon` | `10` | Steps executed from each 32-step chunk (paper and leaderboard protocol). |
| `base_seed` | `2609` | Base of the per-chunk flow-matching noise seeds, which also depend on the task, eval seed, episode and checkpoint SHA-256. |
| `checkpoint_path` | `null` | Explicit `model.pt` (or its directory); overrides `ckpt_name` resolution. Also settable through `RESWAM_CHECKPOINT_PATH`. |
| `checkpoint_sha256` | `null` | Precomputed checkpoint SHA-256, which skips hashing the 9.3 GB file at start-up. Also settable through `RESWAM_CHECKPOINT_SHA256`. |
| `codec_statistics_path` | `null` | Normalization statistics; defaults to `codec_statistics.json` next to `model.pt`. |

`RESWAM_REPO` points the adapter at another ResWAM checkout instead of `policy/ResWAM/ResWAM`.

## Notes

- Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. The model consumes the head and both wrist cameras plus the 14-D joint state.
