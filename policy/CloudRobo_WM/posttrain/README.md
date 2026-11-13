# CloudRobo-WM Ascend post-training

The model, reader, optimizer, metrics, launch runtime and configuration files are
the byte-identical sources used by the current 18-node training
job. Source identity is recorded in `data/training_source.json`; the original
sealed source manifest is retained as provenance in
`data/training_source_manifest.json`. The launch manifest for this policy-format
package is `FILES_SHA256.json`.

Use the matching **2509-tensor BF16  initialization** and its manifest.


## Installation

Post-training uses a separate Linux Ascend environment with matching PyTorch/`torch_npu` releases, a compatible CANN runtime, DeepSpeed 0.19.0 and a CPU decoder exposing PyAV (`av`). Install the framework and its device extension first, then install the included training source independently:

```bash
cd posttrain
/opt/crwm-training/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
cd ..
```

Set `CLOUDROBO_TRAIN_PYTHON` to this environment's interpreter when using the standard `train.sh` entrypoint. Training and inference use separate environments.

## Data preparation

Training consumes clean-only data of RobotWin3D's LeRobot **v2.1** export from [flex-pi/robotwin_3d](https://huggingface.co/datasets/flex-pi/robotwin_3d), with parquet, RGB videos, packed lossless depth and camera intrinsics. These keys and media differ from XPolicyLab's official LeRobot converters; their output cannot replace this export. XPolicyLab encoded image buffers never enter this native reader.

From the parent policy directory:

```bash
python3 -m pip install huggingface_hub
bash download_data.sh --output-dir /datasets/robotwin3d
bash process_data.sh RoboTwin crwm arx_x5 joint \
  --data-root /datasets/robotwin3d --bundle-root /datasets/crwm-bundle --index-only
bash process_data.sh RoboTwin crwm arx_x5 joint \
  --data-root /datasets/robotwin3d --bundle-root /datasets/crwm-bundle \
  --training-resources /datasets/crwm-training-resources/data --prepare-permissions
```

The included `data/clean_index.json.gz` and `data/train_stats.json` define the same 2,500 Clean episodes (50 tasks × 50 demonstrations). Their hashes and matching contract are recorded in `data/training_metadata_manifest.json`. The data preparation entry verifies them before writing the training index and statistics; it does not depend on a pre-existing ModelArts index.

`process_data.sh` uses `posttrain/.venv/bin/python` when available. Set `CLOUDROBO_DATA_PYTHON` to select another training interpreter.

The standard resource release has this layout:

```text
crwm-training-resources/
├── data/
│   ├── eef/manifest.json
│   ├── eef/episodes/
│   ├── prompt_clean/  # published manifest, indexes, texts and vectors/
│   └── instruction_overlay/
│       ├── metadata/selected_training_reader_rows.json
│       └── cache/
├── assets/dino.safetensors
├── assets/vae.safetensors
├── initialization_bf16.safetensors  # 2509-tensor  initialization
└── initialization_bf16.safetensors.manifest.json
```

The precheck stages only the data resources used by the selected episodes. It checks the original downloaded files, published index/statistics, selected sidecars, standard text caches and native reader coverage before publishing `crwm-bundle/READY.json`. `--index-only` does not publish READY. Use `--verify-only` on `process_data.sh` to validate an already prepared bundle without changing it. The complete data check requires the training environment; it does not load inference or training model weights.

The prepared bundle contains `data/indices/train_manifest_0000.json`, `data/indices/train_stats.json`, the data resources and a link at `data/robotwin_3d` to the original downloaded dataset. `--prepare-permissions` applies the supplied Ascend recipe's 0777 data contract to current-user-owned files. Existing different files are preserved and rejected.

## Training launch

Run one launcher per node from the parent policy directory using the standard `train.sh` entrypoint. Its sixth argument is named `npu_id`, which requires **16 distinct logical NPU IDs**. Other seeds, robot/action configurations and world sizes are rejected.

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <npu_id> \
  [--mode train|smoke|dry-run]
```

Configure the paths below and set each node's own `NODE_RANK`:

```bash
export CLOUDROBO_TRAIN_PYTHON=/opt/crwm-training/bin/python
export CLOUDROBO_WM_DATA_ROOT=/datasets/crwm-bundle/data
export CLOUDROBO_WM_ASSET_ROOT=/datasets/crwm-training-resources/assets
export CLOUDROBO_WM_INIT_WEIGHTS=/datasets/crwm-training-resources/initialization_bf16.safetensors
export CLOUDROBO_WM_MEDIA_REPLICA_RECEIPT=/datasets/robotwin3d/RAW_MEDIA_REPLICA_SHA256.json
export CLOUDROBO_WM_OUTPUT_ROOT=/mnt/sfs/cloudrobo-wm/runs/check-fp32-001
export CLOUDROBO_WM_COMM_DTYPE=fp32
export CLOUDROBO_WM_NPU_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

# Manual distributed launch: use the correct node index, 0..17.
export NNODES=18
export NODE_RANK=0
export MASTER_ADDR='<rank-zero-address>'
export MASTER_PORT=29500

bash train.sh RoboTwin crwm arx_x5 joint 42 "$CLOUDROBO_WM_NPU_DEVICES" --mode dry-run
bash train.sh RoboTwin crwm arx_x5 joint 42 "$CLOUDROBO_WM_NPU_DEVICES" --mode smoke

export CLOUDROBO_WM_SMOKE_REPORT=/mnt/sfs/cloudrobo-wm/runs/check-fp32-001/SMOKE_REPORT.json
export CLOUDROBO_WM_OUTPUT_ROOT=/mnt/sfs/cloudrobo-wm/runs/train-fp32-001
bash train.sh RoboTwin crwm arx_x5 joint 42 "$CLOUDROBO_WM_NPU_DEVICES"
```

Use a reachable rank-zero address. All nodes must share the same data, assets and output paths. The included recipe uses 18 nodes, 16 logical ranks per node, batch 12 per rank (global batch 3456), seed 42 and 5,542 updates. Training configuration files are under `configs/`.

`dry-run` inspects the command without loading weights or starting training. `smoke` saves step 3, starts new processes and resumes through step 6. `train` requires the matching successful `SMOKE_REPORT.json`. Use a fresh output directory for each check or formal run. FP32 and BF16 communication have separate configurations and matching check reports. Set `CLOUDROBO_WM_COMM_DTYPE=bf16` to select the latter.

## ModelArts

Configure the distributed job for the recipe above and mount the same data, assets and shared output directory on all nodes. Set the `CLOUDROBO_WM_*` environment variables above and use this policy directory as the working directory.

Startup command for the save/resume check:

```bash
bash posttrain/SMOKE_RESUME.sh
```

Startup command for a separate formal job, with its matching report and a new output root:

```bash
bash posttrain/FULL_TRAIN.sh train
```

For a Python startup-file job, select `posttrain/runtime/launch.py` with argument `smoke`, `train` or `dry-run`. The launcher reads the platform node count, node index and master address from `VC_WORKER_*`/`VC_TASK_INDEX` or `MA_NUM_HOSTS`/`MA_TASK_INDEX`/`MA_MASTER_IP`. Remove stale manual topology values when using the platform variables.

Outputs require an unlinked shared POSIX/SFS directory supporting atomic rename, outside the code/data bundle. An OBS-linked directory or ModelArts OBS output channel is unsuitable for the checkpoint directory.

## Recovery and logs

`CLOUDROBO_WM_RESUME` points to a complete saved state directory. A tensor-only export is not a full recovery checkpoint. Complete states and separate BF16 exports are saved every 1000 updates and at the final update. New exports require their inference compatibility check before evaluation.

Global training scalars are in `metrics.jsonl` and `GLOBAL_STATUS.json`, using packed FP32 reduction across all 288 ranks. Per-rank `STATUS.json` keeps local losses for diagnostics. Joint/Gripper and EEF diagnostics retain their existing global scale; they are not multiplied or divided by world size again. The model exports 2509 BF16 tensors; native full recovery state and tensor-only inference exports remain separate.
