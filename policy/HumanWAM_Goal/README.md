# HumanWAM_Goal

**Contributor:** Open-HumanWAM Team (DAGroup, PKU) | **Paper:** [Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models](https://dagroup-pku.github.io/open-humanwam/) | **arXiv:** [2611.06817](https://arxiv.org/abs/2611.06817) | **Original code:** [DAGroup-PKU/HumanNet (`Open-HumanWAM`)](https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM/humanwam_goal) | **Weights:** [HumanWAM-Goal-RoboTwin](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-RoboTwin)

HumanWAM-Goal is a goal-image-conditioned world action model that jointly denoises future video and a 48-step action chunk from the current observation, short observation history (0.5 s and 2 s), language instruction, and one goal image. This adapter supports RoboTwin 2.0 Aloha-AgileX data conversion, FSDP2 post-training from [`HumanWAM-Goal-Base`](https://huggingface.co/DAGroup-PKU/HumanWAM-Goal-Base), and closed-loop evaluation (`action_type=joint`, 32-step replanning). Model, geometry, and training packages (`goalwam` and `goalwam_train`) are installed from `Open-HumanWAM` by `install.sh`; nothing is vendored here.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard) and [RoboTwin 2.0 Leaderboard](https://robotwin-platform.github.io/leaderboard).

## Installation

Place this XPolicyLab checkout inside the RoboTwin 2.0 workspace (`<RoboTwin>/XPolicyLab`) so XPolicyLab's `env_cfg` lookup resolves to RoboTwin's robot metadata:

```bash
cd policy/HumanWAM_Goal
bash install.sh
python download_checkpoint.py --ckpt-name HumanWAM-Goal-RoboTwin
```

`install.sh` clones `DAGroup-PKU/HumanNet` at the commit pinned in `release.json` into `OpenHumanWAM/` (override with `HWGOAL_REPO`), builds the `uv` virtual environment `env/` via `humanwam_goal/scripts/setup_environment.sh` (Python 3.13, PyTorch 2.10.0+cu128, FlashAttention-3; requires glibc ≥ 2.34), and installs `humanwam_goal/posttrain[convert]` plus XPolicyLab. `setup_environment.sh` also installs `nvidia-cublas-cu12==12.8.3.14` (`env/runtime/cublaslt-12.8.3.14/`), which the policy server preloads via `LD_PRELOAD` for deterministic BF16 inference (override with `HWGOAL_CUBLASLT`).

For post-training, also download the pretrained base checkpoint and training goal cache:

```bash
python download_checkpoint.py --ckpt-name HumanWAM-Goal-Base --with-train-goal-cache
```

## Data Processing

HumanWAM-Goal trains on LeRobot v3.0 datasets (`observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` at 30 FPS, and 49-D `observation.state` / `action` vectors in head-camera NWU coordinates with `command_openness` grippers — distinct from XPolicyLab's generic 14-D converter). `goalwam_train.convert` converts the 50 official [`TianxingChen/RoboTwin2.0`](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) `aloha-agilex_clean_50.zip` HDF5 archives directly into this layout without reading XPolicyLab trajectories:

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>

hf download TianxingChen/RoboTwin2.0 --repo-type dataset \
  --include "dataset/*/aloha-agilex_clean_50.zip" --local-dir /path/to/robotwin_raw
HWGOAL_RAW_ROBOTWIN=/path/to/robotwin_raw \
  bash process_data.sh RoboTwin HumanWAM-Goal-RoboTwin aloha_agilex joint
```

Set `HWGOAL_CLEAN_50` to link an already-converted `clean_50` directory instead. When `checkpoints/HumanWAM-Goal-Base` is present, `process_data.sh` also runs `goalwam_train.normalization` to generate `data/<setting>/robot.json` (`q01`/`q99` joint bounds over the 2,500 `clean_50` episodes at horizon 48, `[0, 1]` gripper openness, and shared EEF bounds from `HumanWAM-Goal-Base`), matching `checkpoints/HumanWAM-Goal-RoboTwin/robot.json` (`a9b20372b33f3a2618f763bf95f0b4b8da9225e3d038798b59075f1310b5e4c8`):

```bash
env/bin/python -m goalwam_train.normalization \
  --data data/RoboTwin-HumanWAM-Goal-RoboTwin-aloha_agilex-joint/clean_50 \
  --pretrained checkpoints/HumanWAM-Goal-Base \
  --output data/RoboTwin-HumanWAM-Goal-RoboTwin-aloha_agilex-joint/robot.json
```

When `assets/train_goal_cache` is present, `process_data.sh` verifies its manifest and links it to `data/<setting>/train_goal_cache`.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

bash train.sh RoboTwin HumanWAM-Goal-RoboTwin aloha_agilex joint 42 0,1,2,3,4,5,6,7
```

`train.sh` launches `goalwam_train.train` under `torchrun` (36,000 steps, global batch 256, micro-batch 16) from `checkpoints/HumanWAM-Goal-Base` (override with `HWGOAL_PRETRAINED`) and links the final safetensors checkpoint to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. Extra CLI flags after `<gpu_id>` are forwarded to `goalwam_train.train`.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env_conda_env>

# Single task (clean / randomized)
HWGOAL_TASK_CONFIG=demo_clean bash eval.sh RoboTwin open_microwave HumanWAM-Goal-RoboTwin \
  aloha_agilex joint 0 0 1 "$PWD/env" RoboTwin
HWGOAL_TASK_CONFIG=demo_randomized bash eval.sh RoboTwin open_microwave HumanWAM-Goal-RoboTwin \
  aloha_agilex joint 0 0 1 "$PWD/env" RoboTwin

# All 50 tasks via RoboTwin's multitask scheduler
cd /path/to/RoboTwin
for setting in demo_clean demo_randomized; do
  bash scripts/eval_policy.sh multitask --config env_cfg/eval/all_tasks.yml \
    --policy-name HumanWAM_Goal --ckpt-name HumanWAM-Goal-RoboTwin \
    --bench-name RoboTwin --env-cfg-type aloha_agilex \
    --policy-conda-env "$PWD/XPolicyLab/policy/HumanWAM_Goal/env" \
    --eval-env-conda-env RoboTwin --task-config "$setting" --num-workers 1
done
```

Simulator evaluation runs RoboTwin's `scripts/eval_policy_xpolicylab.py`:

| `HWGOAL_SEED_MODE` | Behavior |
| --- | --- |
| `fixed` (default) | Evaluates the 100 cached `(seed, instruction)` cases per task from `assets/goal_cache` (`HWGOAL_FIXED_CASES=start:stop` runs a slice). |
| `native` | Runs `eval_policy_xpolicylab.py`'s live expert check from seed `100000` against `assets/goal_cache`. |

Each simulator episode loads the pre-generated `224×288` head-camera goal image for `(task_config, task_name, seed, instruction)` from `assets/goal_cache` after verifying the step-0 observation, and aborts on a cache miss rather than skipping seeds.

`EVAL_ENV_TYPE=debug` runs XPolicyLab's offline debug loop with the current head camera frame as a placeholder goal and ends with `[MAIN] eval finished`.

## Model Assets

| Path | Content |
| --- | --- |
| `checkpoints/HumanWAM-Goal-RoboTwin/` | Post-trained RoboTwin 2.0 safetensors checkpoint (`transformer/`, `vae/`, `text_tokenizer/`, `processor.json`, `sampling.json`, `robot.json`, `export.json`) |
| `checkpoints/HumanWAM-Goal-Base/` | Pretrained initialization checkpoint for post-training |
| `assets/goal_cache/` | 10,000 pre-generated head-camera evaluation goals (50 tasks × 100 episodes × `demo_clean` / `demo_randomized`, generated from step-0 head RGB + instruction only) |
| `assets/train_goal_cache/` | 2,500 pre-generated head-camera training goals for `clean_50` |

## Configuration

Extra `deploy.yml` keys (relative paths resolve against `policy/HumanWAM_Goal/`): `ws_ping_interval_s`, `ws_ping_timeout_s`, `checkpoint_path`, `official_goal_cache`, `official_goal_cache_sha256`, `robotwin_profile`, `cublaslt_library`, `output`.

Environment overrides: `HWGOAL_TASK_CONFIG`, `HWGOAL_SEED_MODE`, `HWGOAL_FIXED_CASES`, `HWGOAL_CHECKPOINT`, `HWGOAL_PRETRAINED`, `HWGOAL_RAW_ROBOTWIN`, `HWGOAL_CLEAN_50`, `HWGOAL_TRAIN_GOAL_CACHE`, `HWGOAL_GOAL_CACHE`, `HWGOAL_GOAL_CACHE_SHA256`, `HWGOAL_PROFILE`, `HWGOAL_CUBLASLT`, `HWGOAL_OUTPUT`, `HWGOAL_REPO`, `HWGOAL_ENV`, `ROBOTWIN_ROOT`, `HWGOAL_CHECK_ONLY`.

## Notes

- Stateful history and per-episode goal conditioning require one environment per policy server (`eval_batch: false`, `--num-workers 1`).
- `env_cfg_type` accepts `aloha_agilex` (and wire-equivalent `arx_x5`).
- Reported RoboTwin 2.0 result (50 tasks × 100 episodes): **92.66%** Clean, **86.50%** Randomized (**89.58%** mean).
