# InternW0_delta

**Contributor:** Xingyu Miao | **Paper:** InternW0-Δ | **arXiv:** [2609.31394](https://arxiv.org/abs/2609.31394) | **Original code:** [InternRobotics/InternW0-Delta](https://github.com/InternRobotics/InternW0-Delta)

RoboDojo adapter for InternW0-Δ on `arx_x5` / `joint` (14D): action horizon 32, replan after 10 steps, 10 denoising steps. Evaluation uses the vendored `wam_runtime/`. Training and data processing call the full official checkout (`wam.datasets`, Hydra, Accelerate/DeepSpeed), pinned by `sync_upstream.sh`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Linux, Python 3.11, CUDA 12.8. From this directory:

```bash
cd policy/InternW0_delta
bash install.sh internw0-delta
conda activate internw0-delta
```

`install.sh` pins PyTorch 2.10.0 (cu128), Transformers 5.13.0, and a conda-local gcc/g++ so RynnBrain's FLA/Triton helper can compile on first forward. Evaluation-only installs `wam_runtime/`. Training also needs the official tree, installed with the `[train,modelscope]` extras:

```bash
bash sync_upstream.sh
bash install.sh internw0-delta
```

`INTERNW0_DELTA_ROOT` points `install.sh`, `process_data.sh`, and `train.sh` at an existing checkout instead of `policy/InternW0_delta/InternW0-Delta`.

## Data Processing

Only `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`. The second argument is `ckpt_name`, the same string later passed to `train.sh` and `eval.sh`. It is also the default raw-task folder: HDF5 is read from the parent workspace at `data/RoboDojo/<ckpt_name>/arx_x5/data/`.

```bash
cd policy/InternW0_delta
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# One task. Writes data/RoboDojo-stack_bowls-arx_x5-joint/
bash process_data.sh RoboDojo stack_bowls arx_x5 joint
```

The dataset is **LeRobot v2.1** (`meta/tasks.jsonl`, `meta/episodes.jsonl`, one parquet per episode). Keys match the official converter `scripts/transform_lerobot_v21_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)): `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}`, `observation.state`, `action`, 14D, at 480×640. `configs/data/robodojo.yaml` in the pinned upstream reads those keys. Decoding goes through `decode_image_bit`.

`expert_data_num` caps episodes per task. `raw_task_dirs` (or `INTERNW0_RAW_TASK_DIRS`) is a comma-separated list of task folders when `ckpt_name` is not itself a task name:

```bash
INTERNW0_RAW_TASK_DIRS=stack_bowls,place_bowl bash process_data.sh RoboDojo cotrain arx_x5 joint
```

The converter imports `lerobot.datasets.lerobot_dataset` from a LeRobot release that still writes codebase v2.1. That package is not in `internw0-delta` (upstream vendors its own reader), so set `INTERNW0_CONVERT_PYTHON` to a Python that has it. `ROBODOJO_DATA_ROOT` overrides the output directory. An already complete dataset is reused.

The wrapper then builds the official text cache (`tools/text_cache.py task=robodojo`) with the local Wan2.2 UMT5 encoder from `download_assets.sh`, under `.cache/internw0/text_embed/robodojo`. The Wan directory must be named `Wan2.2-TI2V-5B`. Set `INTERNW0_SKIP_TEXT_CACHE=1` to skip the cache.

## Training

Same four leading arguments as data processing. Artifacts go to `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/checkpoints/weights/slot_*.pt`, with `weights_manifest.json` and `checkpoints/state/latest/`.

```bash
cd policy/InternW0_delta
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

# Continues the stack_bowls dataset from the command above
bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

`gpu_id` may be comma-separated; `num_gpus` defaults to that count. A missing dataset triggers `process_data.sh`. Before training, run `download_assets.sh` and place [`InternW0-Delta-Base`](https://huggingface.co/InternRobotics/InternW0-Delta-Base) at `checkpoints/pretrain.pt` (`WAM_PRETRAIN_CHECKPOINT`). The wrapper points Hydra at the local Wan2.2 and RynnBrain trees (`model.redirect_common_files=false`); `DIFFSYNTH_SKIP_DOWNLOAD=true` makes a missing file fail. The video DiT is loaded from `pretrain.pt` (`skip_dit_load_from_pretrain=true`).

Extra Hydra overrides go in `INTERNW0_TRAIN_OVERRIDES`. Leave model-architecture keys alone; evaluation rebuilds the model from `config/eval_model.yaml`.

```bash
INTERNW0_TRAIN_OVERRIDES="max_steps=1000 save_every=500" \
  bash train.sh RoboDojo stack_bowls arx_x5 joint 0 0
```

## Evaluation

```bash
cd policy/InternW0_delta
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Released weights: checkpoints/robodojo.pt (no matching run directory)
bash eval.sh RoboDojo stack_bowls robodojo arx_x5 joint 0 0 0 internw0-delta <eval_env_conda_env>

# The training run above
bash eval.sh RoboDojo stack_bowls stack_bowls arx_x5 joint 0 0 0 internw0-delta <eval_env_conda_env>
```

`ckpt_name` is the short run name (`stack_bowls` → `checkpoints/RoboDojo-stack_bowls-arx_x5-joint-<seed>/`) or the full run-directory name. Inside that directory the server loads `checkpoints/weights/slot_*.pt` from `weights_manifest.json` (`latest`), else the newest slot. A run directory with no slots is an error. If no run directory exists, evaluation uses `checkpoints/robodojo.pt`. `WAM_CHECKPOINT_PATH` overrides both. Normalization stays `config/dataset_stats.json` (same JSON as upstream `assets/stats/robodojo.json`).

Offline wiring check, no weights:

```bash
EVAL_ENV_TYPE=debug WAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls robodojo arx_x5 joint 0 0 0 internw0-delta base
```

Leave `EVAL_ENV_TYPE` unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

Weights are not in Git. Three artifacts, paths relative to this directory:

| Artifact | Source | Local path |
| --- | --- | --- |
| Wan2.2 VAE, UMT5, tokenizer | `Wan-AI/Wan2.2-TI2V-5B` on ModelScope | `assets/Wan-AI/Wan2.2-TI2V-5B/` |
| RynnBrain | `Alibaba-DAMO-Academy/RynnBrain1.1-2B` on ModelScope | `assets/Alibaba-DAMO-Academy/RynnBrain1.1-2B/` |
| RoboDojo checkpoint `robodojo.pt` | [InternRobotics/InternW0-Delta-RoboDojo](https://huggingface.co/InternRobotics/InternW0-Delta-RoboDojo) | `checkpoints/robodojo.pt` |

```bash
conda activate internw0-delta
bash download_assets.sh                  # or: bash download_assets.sh /path/to/local/assets
bash download_checkpoint.sh              # or a local file / HTTPS URL; SHA256 is checked
```

Wan2.2 downloads only the VAE, UMT5 encoder, and tokenizer. The Video-DiT snapshot is unnecessary: `robodojo.pt` already contains that expert. Expected paths and hashes are in `config/artifacts.lock.json`. Review the upstream licenses before downloading.

## Configuration

Adapter-specific `deploy.yml` keys, resolved from this directory:

- Paths: `checkpoint_path`, `base_model_dir`, `vlm_model_path`, `dataset_stats_path`, `train_config_path`.
- Inference contract (keep the defaults to match the reported result): `device`, `mixed_precision`, `action_horizon`, `replan_steps`, `num_inference_steps`, `action_hz`, `text_cfg_scale`, `negative_prompt`, `rand_device`, `tiled`.
- `timing_enabled`, `default_instruction` (used when an observation has no instruction), `allow_dummy_policy`.

`setup_eval_policy_server.sh` environment overrides:

| Variable | Override |
| --- | --- |
| `WAM_CHECKPOINT_PATH` | weight file, instead of run-dir resolution |
| `WAM_DATASET_STATS_PATH` | z-score statistics JSON |
| `WAM_EVAL_CONFIG_PATH` | evaluation model configuration |
| `WAM_WAN22_PATH` | local Wan2.2 directory |
| `WAM_RYNNBRAIN_PATH` | local RynnBrain directory |
| `WAM_ALLOW_DUMMY_POLICY=true` | debug wiring only; skips real weights |

Evaluation uses the checkpoint z-score stats, RGB input with no training-time color jitter, discrete Action RoPE, physical-time RoPE off, and fan-in calibration off.

## WUJI 58-D hybrid (tianji_wuji)

The adapter also serves checkpoints trained for the WUJI dual-arm dexterous-hand
platform (`env_cfg_type=tianji_wuji`, registered as `arm_dim=[7, 7]`,
`ee_dim=[20, 20]`) in the 58-D hybrid layout defined by WUJI SimBench and
implemented once in `XPolicyLab.utils.wuji58`:

```text
[ 0:20]  left hand joints   absolute, finger-major
[20:29]  left wrist         xyz + rot6d; state absolute in the wrist frame,
                            action = camera-frame delta vs. the live wrist
[29:49]  right hand joints
[49:58]  right wrist
```

Run evaluation with `action_type=ee`; the observation keys consumed are
`left_ee_pose` / `right_ee_pose` (7, `[x y z qw qx qy qz]`) and
`left_ee_joint_state` / `right_ee_joint_state` (20), and the returned action
dicts carry the same four keys as absolute targets:

```bash
bash policy/InternW0_delta/eval.sh RoboDojo <task_name> <ckpt_name> tianji_wuji ee <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> <eval_env_conda_env>
```

Requirements and keys:

- A checkpoint post-trained on 58-D hybrid data plus its matching
  `dataset_stats.json` (58-wide `action` / `state` stats). The adapter checks the
  stats width at load and refuses a 14-D arx_x5 stats file for a `tianji_wuji` run.
  The bundled `config/dataset_stats.json` is the 14-D arx_x5 file.
- `wrist_frame` (`deploy.yml`, default `te2`): the fixed frame wrist poses are
  expressed in. `te2` is the WUJI SimBench head camera the public WUJI data was
  encoded with; `world` keeps world coordinates. It must match training.
- Each predicted chunk is decoded with the shared `decode_action_chunk58` default
  (`iterative`): element 0 against the observation it was predicted from, each
  later element against the pose decoded before it, matching the per-frame
  deltas the official converters write. With `allow_dummy_policy: true` the
  adapter returns a hold-position chunk.

The arx_x5 joint contract (`action_type=joint`, 14-D) is unchanged; any other
embodiment is rejected at construction.

## Notes

- Supported target is `RoboDojo` + `arx_x5` + `joint` only.
- `eval_batch: true`. The policy is stateful, so each `env_idx` has its own session (memory frames, pending actions, step counter) and replans through the single-sample path. GPU inference is one environment at a time. Batched observations must carry `env_idx`.
- Keep `ckpt_name` the same from `process_data.sh` through `train.sh` to `eval.sh`. `robodojo` is the released-file name, not a training run.
- Reference result: one single-environment run over 54 tasks and 6,300 episodes, 1,444 successes — 22.92% success rate, 30.35 mean score.
