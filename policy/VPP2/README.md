# VPP2

**Contributor:** [Haodong Yan](https://github.com/Haodong-Yan) | **Paper:** Video Prediction Policy 2: Predict Better, Act Better | **arXiv:** https://arxiv.org/abs/2610.10270 | **Original code:** https://github.com/roboterax/video-prediction-policy-2

`VPP2` adapts [Video Prediction Policy 2](https://robert-gyj.github.io/video-prediction-policy-2/) — a Wan2.1-I2V-14B video model jointly trained with a 2B action expert (Action2B) — to XPolicyLab/RoboDojo. It trains and evaluates the released joint Video + Action2B 100k policy on **RoboDojo / `arx_x5` / `ee` (absolute EE16)** only. The upstream implementation is not vendored: `install.sh` clones the official repository at a pinned commit into `upstream/` (git-ignored), and the scripts here are thin wrappers around `upstream/scripts/robodojo/`. LIBERO, LIBERO-OOD and LIBERO-PRO are supported only in the official repository.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Requires Linux and an NVIDIA driver compatible with the selected CUDA build. Run from a RoboDojo workspace with `env_cfg/` beside the XPolicyLab checkout:

```bash
cd XPolicyLab/policy/VPP2
bash install.sh [env_name] [eval|train]

# Example: evaluation-only environment, then download and verify the released bundle
bash install.sh vpp2
conda activate vpp2
bash download_checkpoints.sh
python launch_policy.py --dry-run
```

Use `bash install.sh vpp2 train` for a training environment (adds DeepSpeed; needs CUDA compiler tools). The installer creates a Python 3.10 env, clones the official code at `ed39864` into `upstream/`, and installs PyTorch 2.11 / CUDA 13.0 by default — override with `TORCH_VERSION`, `TORCHVISION_VERSION` and `TORCH_CUDA`. It refuses to run when `upstream/` already exists; move that directory to reinstall. To add training dependencies to an existing eval env:

```bash
python -m pip install -r upstream/requirements-train.txt -c upstream/environment-reference.txt
```

Install the simulator separately using the RoboDojo instructions.

## Data Processing

VPP2 trains on the published **LeRobot v3.0 EE16 export** `RoboDojo_ee_lerobot_v30_video` (pinned revision below). It does **not** follow the layout of the [official converters](../../README.md#official-lerobot-conversion): those are joint-space only, while this export stores absolute EE16 state/action `[x, y, z, qw, qx, qy, qz, gripper] × 2` (left, then right). `scripts/RoboDojo/download_robodojo_data.sh` does not fetch it, so download it with `hf download`. Joint14 data and raw HDF5 trajectories are not supported.

`process_data.sh` does not take the standard `<bench_name> <ckpt_name> ...` arguments. It runs two upstream stages: `convert` writes one native-frame EE16 Parquet and one T-shaped RGB video per episode plus `full_episode_metadata.csv`; `prepare` builds the fixed 3466/34 train/held-out split (`train.csv`, `val.csv`), copies the supplied normalization statistics to `dataset_stats.json`, and builds `critical_index/`:

```bash
bash process_data.sh convert --source <ee_lerobot_v30_dir> --output <media_root> [--workers N]
bash process_data.sh prepare --metadata <media_root>/full_episode_metadata.csv \
  --media-root <media_root> --output <prepared_dir>

# Example: full public dataset, from policy/VPP2 in the activated env
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision cefcfbbf2497103fe46b99039dbd381976fe4a42 \
  --include 'data/RoboDojo_ee_lerobot_v30_video/**' --local-dir /data/robodojo_download
export VPP2_SOURCE=/data/robodojo_download/data/RoboDojo_ee_lerobot_v30_video
export VPP2_MEDIA_ROOT="$PWD/data/robodojo_source"
export VPP2_PREPARED="$PWD/data/robodojo"
bash process_data.sh convert --source "$VPP2_SOURCE" --output "$VPP2_MEDIA_ROOT" --workers 4
bash process_data.sh prepare --metadata "$VPP2_MEDIA_ROOT/full_episode_metadata.csv" \
  --media-root "$VPP2_MEDIA_ROOT" --output "$VPP2_PREPARED"
```

Conversion needs `ffmpeg` with `libx264` and checks the released inventory (3500 episodes / 1,856,102 frames at 25 Hz), so keep the pinned dataset revision and use fresh output directories. Camera composition and a small conversion smoke test are described in the upstream [data guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/training.md#convert-the-public-dataset).

## Training

`train.sh` does not take the standard `<bench_name> <ckpt_name> ...` arguments either: it forwards `--dry-run`, `--allow-batch-change` and OmegaConf `key=value` overrides to the official 0–100k recipe, and writes to `VPP2_OUTPUT_DIR` rather than `checkpoints/<bench_name>-<ckpt_name>-...`. Evaluation reads an exported bundle (step 4 below). [`train_100k.yaml`](train_100k.yaml) inherits `upstream/configs/robodojo/train_100k.yaml` and only redirects paths to this directory.

| Recipe | Value |
| --- | --- |
| Initialization | Video from the released history-conditioned Video-10k; Action2B interpolated from the same backbone |
| Steps / global batch | 100,000 / 288 (`batch_size: 3` per GPU) |
| Learning rate | Video `5e-6`, action `1e-4`; 500-step warmup, cosine, low-LR tail from step 80,000 |

**1. Assets, text cache and Action2B initializer.** Text caching needs one GPU; initialization is a large CPU job.

```bash
bash download_checkpoints.sh --stage train
export VPP2_WAN_ROOT="$PWD/checkpoints/Wan2.1-I2V-14B-480P"
export VPP2_VIDEO_INIT="$PWD/checkpoints/initialization/robodojo_his10k.pt"
export VPP2_ACTION_INIT="$PWD/checkpoints/action2b_init.pt"
CUDA_VISIBLE_DEVICES=0 bash text_cache.sh --data "$VPP2_PREPARED" --wan "$VPP2_WAN_ROOT"
bash init_action.sh --video "$VPP2_VIDEO_INIT" --output "$VPP2_ACTION_INIT"
```

**2. Launcher topology.** Without overrides the launcher assumes the reference layout — `NNODES=12`, `NPROC_PER_NODE=8`, `REQUIRE_RDMA=1` — and stops if the visible GPU count differs from `NPROC_PER_NODE` or `/dev/infiniband` is missing. Training refuses a global batch other than 288 (`batch_size × gradient_accumulation_steps × world_size`) unless `--allow-batch-change` is passed. On any other layout, set the topology and raise `gradient_accumulation_steps` to keep 288; for multiple nodes, run the same command on every node with its own `NODE_RANK` plus shared `MASTER_ADDR` / `MASTER_PORT`, data paths and output directory.

```bash
bash train.sh [--dry-run] [--allow-batch-change] [key=value ...]

# Example: one node with 8 GPUs (8 × 3 × 12 = 288)
export NNODES=1 NPROC_PER_NODE=8 REQUIRE_RDMA=0
bash train.sh --dry-run gradient_accumulation_steps=12
```

`--dry-run` only prints the resolved world size, global batch and initializer paths. It does not check that the data or weights exist and does not allocate GPUs; the real launch checks the files and then loads the model.

**3. Preflight, then the formal run.** Run the 20-step probe in its own output directory, inspect `probe.json` and checkpoint loading, then start the formal run fresh:

```bash
bash train.sh gradient_accumulation_steps=12 output_dir="$PWD/runs/robodojo_joint2b_probe" \
  max_steps=20 eval_at_start=false eval_every=0 save_every=20 \
  preflight.enabled=true preflight.expected_steps=20

export VPP2_OUTPUT_DIR="$PWD/runs/robodojo_joint2b_100k"
bash train.sh gradient_accumulation_steps=12
# Resume full training state after an interruption:
bash train.sh gradient_accumulation_steps=12 resume="$VPP2_OUTPUT_DIR/checkpoints/state/step_090000"
```

Training refuses an output directory that already contains a run unless `resume=` is set. Held-out validation runs at start and every 1000 steps (written to `eval/heldout_action.jsonl`) with action shift 3; simulator evaluation uses shift 1. Pass absolute paths in overrides — relative ones resolve from `upstream/`.

**4. Export the evaluation bundle.** The exporter writes `action.pt`, `video.pt`, `dataset_stats.json` and `manifest.json`, refuses an existing destination, and defaults to step 100,000:

```bash
bash export.sh \
  --checkpoint "$VPP2_OUTPUT_DIR/checkpoints/weights/step_100000.pt" \
  --config "$VPP2_OUTPUT_DIR/config.yaml" \
  --stats "$VPP2_PREPARED/dataset_stats.json" \
  --output "$PWD/checkpoints/my_joint2b_s100000"
bash eval.sh RoboDojo stack_bowls my_joint2b_s100000 arx_x5 ee 1 0 1 vpp2 <eval_env_conda_env>
```

## Evaluation

```bash
cd XPolicyLab/policy/VPP2
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: released bundle on stack_bowls (policy on GPU 0, simulator on GPU 1)
bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 1 vpp2 <eval_env_conda_env>

# Offline wiring checks with real weights, no simulator
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 bash eval.sh RoboDojo stack_bowls joint2b_s100000 arx_x5 ee 1 0 0 vpp2 vpp2
```

`bench_name`, `env_cfg_type` and `action_type` must be `RoboDojo`, `arx_x5` and `ee`. `ckpt_name` is a bundle directory under `checkpoints/` (the release is `joint2b_s100000`) or a path to one; `VPP2_BUNDLE` overrides it. The released results use `seed=1`. The policy server is launched through `launch_policy.py`, which validates the bundle before loading weights. `EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator) and verifies action shapes, not task success; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

`download_checkpoints.sh` fetches weights into `checkpoints/` from public [Hugging Face](https://huggingface.co/Haodong082399/VPP2) (default) or [ModelScope](https://modelscope.cn/models/haodong123/VPP2). Both sources support anonymous downloads; no login or token is required. Every download mode includes the root `config.json` in `policy/VPP2/` and the shared Wan2.1 encoders:

```bash
bash download_checkpoints.sh [--stage eval|train|all] [--source huggingface|modelscope]

# Example: evaluation bundle (default), then the training initializer
bash download_checkpoints.sh
bash download_checkpoints.sh --stage train
bash download_checkpoints.sh --source modelscope
```

```text
checkpoints/
├── joint2b_s100000/                # --stage eval: paired evaluation bundle
│   ├── action.pt
│   ├── video.pt
│   ├── dataset_stats.json
│   └── manifest.json
├── initialization/
│   └── robodojo_his10k.pt          # --stage train: Video-10k initializer
└── Wan2.1-I2V-14B-480P/            # always: VAE, CLIP, UMT5, tokenizer
    ├── Wan2.1_VAE.pth
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    └── google/umt5-xxl/...
```

Keep the four bundle files together. `python launch_policy.py --dry-run` checks bundle file sizes against the manifest, the manifest step and the encoder inventory without allocating a GPU.

The robot-video pretrained Video models are also public, under `checkpoints_video/` in both repositories (about 65.6 GB each): `vpp2-video-stage1-49f.pth` (Stage 1, event-level, 49 frames) and `vpp2-video-stage2-17f.pth` (Stage 2, fixed horizon, 17 frames). They serve zero-shot video prediction, which is not part of this adapter — see the upstream [video prediction guide](https://github.com/roboterax/video-prediction-policy-2/blob/main/docs/video_prediction.md); its script is newer than the commit `install.sh` pins. `download_checkpoints.sh` does not fetch them, and the RoboDojo recipe above still starts from Video-10k:

```bash
# Writes <weights_dir>/checkpoints_video/vpp2-video-stage1-49f.pth
hf download Haodong082399/VPP2 --local-dir <weights_dir> \
  --include config.json 'checkpoints_video/vpp2-video-stage1-49f.pth'
```

## Configuration

`deploy.yml` records the released inference contract:

| Setting | Value |
| --- | --- |
| Action inference | 10 Euler steps, sigma shift 1, seed 1 |
| Predicted / executed actions | 32 / 24 |
| Video context | History 8, native stride 25, episode anchor |
| RGB input | Native views → T-shaped composition → 240 × 416 |
| Action representation | Absolute poses `[x, y, z, qw, qx, qy, qz]` and gripper, left then right |
| Normalization | Checkpoint z-score statistics, including grippers |
| Precision | `device: cuda`, `mixed_precision: bf16` |

Model-specific keys: `deployment_adapter: robodojo_ee16`, `action_dim: 16`, `action_horizon: 32`, `history_stride: 25` and `video_seed_offset: 1000003` fix the checkpoint contract and are validated at load time. `checkpoint_path`, `video_checkpoint_path`, `dataset_stats_path` and `wan_model_dir` default to the bundle and encoder directories above. `default_instruction` is used only when an observation carries no instruction. `max_retained_clients` (default 64) bounds the per-client histories kept by one server.

| Variable | Used by | Notes |
| --- | --- | --- |
| `VPP2_BUNDLE` | eval | Existing bundle directory; overrides `ckpt_name`. |
| `VPP2_WAN_ROOT` | eval, training | Wan2.1 encoder directory; defaults to `checkpoints/Wan2.1-I2V-14B-480P`. |
| `VPP2_NUM_INFERENCE_STEPS` / `VPP2_SIGMA_SHIFT` / `VPP2_REPLAN_STEPS` | eval | Diagnostic overrides of `num_inference_steps` / `sigma_shift` / `replan_steps`; they change the evaluation setting. |
| `VPP2_SOURCE` / `VPP2_MEDIA_ROOT` / `VPP2_PREPARED` | data, training | EE16 source, converted episodes, prepared split; `train_100k.yaml` reads the last two. |
| `VPP2_VIDEO_INIT` / `VPP2_ACTION_INIT` | training | Video-10k initializer and generated Action2B initializer. |
| `VPP2_OUTPUT_DIR` | training | Run directory; defaults to `runs/robodojo_joint2b_100k`. |
| `NNODES` / `NPROC_PER_NODE` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT` / `REQUIRE_RDMA` | training | Launcher topology; defaults `12` / `8` / `0` / `127.0.0.1` / `29500` / `1`. |
| `PYTHON_BIN` | data, training | Interpreter for the upstream scripts; defaults to `python`. |
| `TORCH_VERSION` / `TORCHVISION_VERSION` / `TORCH_CUDA` | install | PyTorch build; defaults `2.11.0` / `0.26.0` / `cu130`. |

## Notes

- `launch_policy.py` accepts only bundles whose manifest step is 100,000. A bundle exported at another step (`export.sh --step N`) does not pass its check.
- Keep `eval_batch: false`: vectorized batch methods raise `NotImplementedError`. Each simulator client sends its own client ID, so several independent clients can share one policy server without mixing observation histories.
- The reference evaluation GPU is one 96 GiB RTX PRO 6000; smaller devices are unverified.
