# KinRT for RoboDojo

**Contributor:** Tianhang Yang, Yanze Zheng, Junjie Wang, Wei-Bin Kou, Ruotong Li, Yujiu Yang | **Paper:** Route by Kinematics, Act by Observation | **arXiv:** https://arxiv.org/abs/2607.26807 | **Original code:** https://github.com/gleeacast/KinRT

This adapter applies KinRT to RoboDojo's dual-ARX-X5 environment: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, and `action_type=joint`. The default model is the delivered **Full35 checkpoint at 60,000 training steps**, trained with full parameter fine-tuning on 35 tasks, 3,500 episodes, and 1,859,602 frames. Its configuration is `kinrt_full_robodojo`; its dataset and normalization key is `RoboDojo_lerobot_v30_video`.

KinRT source remains in a separate checkout; this directory contains the XPolicyLab integration. This checkpoint has passed artifact checks; full inference and closed-loop evaluation have not been run for this submission update.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Use Linux with CUDA 12, a compatible NVIDIA driver, and [uv](https://docs.astral.sh/uv/) for model execution. The installer creates a Python 3.11 policy environment. A GPU with at least 24 GB VRAM is recommended for a standalone inference check; simulator memory is additional. Use sibling checkouts:

```text
<workspace>/
  RoboDojo/
    XPolicyLab/policy/KinRT/
  KinRT_RoboDojo/
    policy/pi05/
  LeRobot_KinRT_Full35/
```

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
bash install.sh "$KINRT_OPENPI_ROOT"
```

The path is optional with this sibling layout. Installation pins KinRT to `590d52802cde804cdc2d0ccb672c1a3a90d76f91` and runs `uv sync --frozen --no-default-groups` without editing its `pyproject.toml` or `uv.lock`. It then installs the adapter dependencies into the KinRT OpenPI environment.

LeRobot is checked out separately at `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` (v0.4.4, dataset format v3.0). An environment-local `kinrt_full35_lerobot.pth` gives this checkout's `src/` import precedence, reproducing the delivered run's `PYTHONPATH` source overlay while preserving the older locked dependencies. A successful install verifies imports and the selected LeRobot version; it does not validate model loading or GPU inference.

The delivery notes document a training-time `data_loader.py` fix for LeRobot task tables, but do not include its original patch. Installation applies a reconstructed compatibility fix that maps the DataFrame's `task_index` column to its prompt index. It also adapts the upstream router-label generator to read v3 episodes packed into or split across Parquet files, filtering by episode and sorting by global frame index without changing clustering or feature calculations. The converter finalizes v3 dataset writers. These changes are explicit; the source commit alone does not contain every modification used for the delivered training run.

From the adapter directory, `python prepare_full35_source.py "$KINRT_OPENPI_ROOT" --check` validates the known source without changing it; the same command without `--check` applies the fixes. `--revert` restores only this helper's exact changes. Both source files are checked before either is modified; unknown source edits are rejected. The installer performs the check and apply steps automatically.

The model repository is private. Authenticate an account with access using `hf auth login`, or provide `HF_TOKEN` through your environment. Evaluators need their own authorized access. No credential is stored in this repository.

## Data Processing

The **four router classes are distinct from the 35 task instructions**. `router_labels_k4_full35.npy` contains one kinematic routing label per global dataset frame; `dataset_meta/tasks.parquet` contains the 35 task indices and their language instructions. Training uses offline labels to supervise observation-conditioned top-1 routing. Inference uses observations and the task instruction, and does not require the `.npy` file.

### Original Full35 Dataset

Inference only needs the checkpoint and its packaged normalization statistics. Training reproduction additionally requires the **original complete LeRobot v3.0 dataset**, including data, videos, and episode metadata. The model delivery includes only `info.json`, `stats.json`, and `tasks.parquet`; those files do not constitute a training dataset.

Place the original dataset under `${HF_LEROBOT_HOME}/RoboDojo_lerobot_v30_video`, retaining its original frame and episode order, then prepare the verified training assets:

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
export HF_LEROBOT_HOME=<workspace>/cache/huggingface/lerobot
export KINRT_ROBODOJO_REPO_ID=RoboDojo_lerobot_v30_video

bash download_checkpoint.sh --assets-only

"$KINRT_OPENPI_ROOT/.venv/bin/python" full35_assets.py prepare-training \
  --artifact-root "$PWD/checkpoints/KinRT-RoboDojo-Full35-60k" \
  --dataset-root "$HF_LEROBOT_HOME/$KINRT_ROBODOJO_REPO_ID" \
  --openpi-root "$KINRT_OPENPI_ROOT"
```

Preparation validates the published assets and existing dataset metadata, then installs:

```text
<dataset>/meta/router_labels_k4_full35/router_labels.npy
<KinRT policy/pi05>/assets/kinrt_full_robodojo/RoboDojo_lerobot_v30_video/norm_stats.json
```

It refuses conflicting existing files. The labels must remain aligned with the original 1,859,602 global frame indices. Metadata and class counts alone cannot prove frame alignment, so do not attach these labels to a reconverted, reordered, filtered, or independently assembled dataset. Preserve the supplied normalization statistics for this checkpoint.

### Custom Datasets

`process_data.sh` remains available for new experiments using source demonstrations at `RoboDojo/data/RoboDojo/<task>/arx_x5/data/episode_*.hdf5`. Use a distinct dataset ID and regenerate its labels and normalization statistics:

```bash
export KINRT_ROBODOJO_REPO_ID=RoboDojo-KinRT-custom-arx_x5-joint
export KINRT_LEROBOT_METADATA_FPS=25
bash process_data.sh RoboDojo multitask arx_x5 joint \
  stack_bowls,insert_key,hang_mugs
bash generate_router_labels.sh
bash compute_norm_stats.sh kinrt_full_robodojo
```

The converter's default dataset ID is `RoboDojo-KinRT-arx_x5-joint`, separate from the original Full35 dataset. The label and normalization wrappers default to the original Full35 ID, so retain the explicit custom `KINRT_ROBODOJO_REPO_ID` for all commands. Existing output is protected: `KINRT_OVERWRITE_DATASET=1`, `KINRT_OVERWRITE_ROUTER_LABELS=1`, and `KINRT_OVERWRITE_NORM_STATS=1` opt into replacing the corresponding custom artifacts. Use those flags only for intentional regeneration of a separately named experiment.

The converter's metadata FPS defaults to 50 for historical adapter compatibility; the command above explicitly selects 25 FPS, matching the original Full35 metadata. Conversion targets the next recorded joint state, repeating the current state on an episode's last frame. A custom conversion does not reproduce the delivered dataset merely by using the same task names.

## Training

The delivered run used full parameter fine-tuning, 60,000 steps, global batch 256, seed 0, 8 DataLoader workers, and 8 A800 80 GB GPUs with FSDP across all 8 GPUs. It saved every 5,000 steps; the delivery retains checkpoints 50,000 and 60,000. Reproduce those settings after preparing the original dataset:

```bash
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
export HF_LEROBOT_HOME=<workspace>/cache/huggingface/lerobot
export KINRT_ROBODOJO_REPO_ID=RoboDojo_lerobot_v30_video
export OPENPI_BASE_CHECKPOINT=<path-to-pi05-base-params>
export OPENPI_TRAIN_CONFIG_NAME=kinrt_full_robodojo
export OPENPI_NUM_TRAIN_STEPS=60000
export OPENPI_BATCH_SIZE=256
export OPENPI_NUM_WORKERS=8
export OPENPI_SAVE_INTERVAL=5000
export OPENPI_FSDP_DEVICES=8
export OPENPI_WANDB_ENABLED=0

bash train.sh RoboDojo full35_full_k4_b256_s0_60k arx_x5 joint 0 0,1,2,3,4,5,6,7
```

The wrapper defaults to the Full35 configuration, dataset key, and budget shown above, and validates the Full35 assets before training. Checkpoints produced by this wrapper use the XPolicyLab layout:

```text
checkpoints/RoboDojo-full35_full_k4_b256_s0_60k-arx_x5-joint-0/<step>/
```

This differs from the downloaded delivery layout described below. Set `KINRT_RESUME=1` to resume a matching existing wrapper run. Changing GPU count, batch size, training configuration, or dataset creates a different experiment.

KinRT uses four experts, top-1 dense routing, supervised routing coefficient `0.05`, balanced-sampling alpha `0.5`, and action horizon `50`. The retained `router_sampling_mix_beta` field is not consumed by the loader. LoRA remains an optional source configuration, but the delivered 60k checkpoint requires `kinrt_full_robodojo`. The delivery includes the launch script, dependency lockfiles, source commit identifiers, and training log for provenance; its documented local source fix was reconstructed during integration.

## Evaluation

Download the 60k parameters and checkpoint assets before model execution:

```bash
cd <workspace>/RoboDojo/XPolicyLab/policy/KinRT
bash download_checkpoint.sh
export KINRT_OPENPI_ROOT=<workspace>/KinRT_RoboDojo/policy/pi05
export KINRT_CHECKPOINT_PATH="$PWD/checkpoints/KinRT-RoboDojo-Full35-60k/checkpoints/60000"
export KINRT_TRAIN_CONFIG_NAME=kinrt_full_robodojo
export KINRT_REPO_ID=RoboDojo_lerobot_v30_video
export KINRT_CHECKPOINT_NUM=60000
```

First load the checkpoint and predict one action chunk without a simulator:

```bash
PYTHONPATH=<workspace>/RoboDojo:"$KINRT_OPENPI_ROOT/src" \
  "$KINRT_OPENPI_ROOT/.venv/bin/python" offline_smoke.py \
  --checkpoint-root "$KINRT_CHECKPOINT_PATH" \
  --checkpoint-step 60000 \
  --repo-id RoboDojo_lerobot_v30_video \
  --train-config-name kinrt_full_robodojo
```

The default test uses deterministic synthetic RGB images, a 14-D state, and an instruction; it requires a finite `50 x 14` action output. Pass `--dataset-root <original-LeRobot-directory>` for a real dataset sample. Neither synthetic nor single-sample inference measures task success.

Then run the transport and schema check through the standard policy server:

```bash
EVAL_ENV_TYPE=debug \
  bash eval.sh RoboDojo stack_bowls full35_full_k4_b256_s0_60k arx_x5 joint 0 \
  0 0 "$KINRT_OPENPI_ROOT" <robodojo_conda_env>
```

Repeat with `DEBUG_OBS_ENCODED=1` to exercise server-side image decoding. For simulator evaluation, omit `EVAL_ENV_TYPE=debug`:

```bash
bash eval.sh RoboDojo stack_bowls full35_full_k4_b256_s0_60k arx_x5 joint 0 \
  0 0 "$KINRT_OPENPI_ROOT" <robodojo_conda_env>
```

Replace `stack_bowls` with the desired supported RoboDojo task. This is a per-task invocation of the shared 35-task model, not a complete 35-task evaluation. Keep the default action chunk size of 50 when comparing that policy setting; smaller `KINRT_ACTION_CHUNK_SIZE` values change replanning frequency.

**These inference, debug, and simulator commands have not yet been validated with the delivered 60k checkpoint.** Local verification was limited to artifact integrity, checkpoint metadata, and sampled weight decoding on an 8 GB Windows GPU machine. Full checkpoint restoration and task success remain unverified.

## Model Assets

The default download is pinned to [Gleez/kinrt-robodojo-full35-a800-60k](https://huggingface.co/Gleez/kinrt-robodojo-full35-a800-60k/tree/9460d07a9c7677ef3c72ece08df1f34eba7e45c7), revision `9460d07a9c7677ef3c72ece08df1f34eba7e45c7`.

```bash
bash download_checkpoint.sh [DESTINATION] [--assets-only | --include-training-state]
```

The default destination is `policy/KinRT/checkpoints/KinRT-RoboDojo-Full35-60k/`, resolved relative to the adapter directory. A normal download includes 60k model parameters, checkpoint metadata and normalization, and all delivery files. It excludes the 50k checkpoint and optimizer/training state. The two optional modes are mutually exclusive: `--assets-only` downloads delivery files and the checkpoint normalization without model weights; `--include-training-state` additionally retrieves the 60k training state.

The downloader prefers the installed policy environment's `.venv/bin/python`, falling back to `python` when it is absent. `KINRT_PYTHON_BIN` overrides that choice. The chosen Python must provide the download and verification dependencies. Authenticate with `hf auth login` or `HF_TOKEN` before accessing the private repository.

Verification can also be rerun explicitly with flags matching the selected download:

```bash
"$KINRT_OPENPI_ROOT/.venv/bin/python" full35_assets.py verify \
  --artifact-root "$PWD/checkpoints/KinRT-RoboDojo-Full35-60k"
```

The checkpoint manifest covers 1,057 files, including 704 training-state files omitted by a default inference download. Verification accounts for the selected download scope and the delivery manifest's original path prefix.

| Asset | Verified value |
| --- | --- |
| Final checkpoint | `checkpoints/60000/`; step 60000 finalized in the training log |
| Training dataset | LeRobot v3.0; 35 tasks / 3,500 episodes / 1,859,602 frames; 25 FPS |
| Router labels | `int32`, shape `(1859602,)`, classes `0..3` |
| Router class counts | `[213759, 453930, 587902, 604011]` |
| Task instructions | `delivery/dataset_meta/tasks.parquet`; task indices `0..34`, prompt text in the pandas index |
| Delivery normalization | `delivery/norm_stats.json` |
| Inference normalization | `checkpoints/60000/assets/RoboDojo_lerobot_v30_video/norm_stats.json` |

Verified SHA-256 digests:

```text
router_labels_k4_full35.npy
3a7f5265006a32a892dada84c38130a424a38cc4ab185a5e5ea31ad215e64cd7
delivery/norm_stats.json
51d09fc068d7af5fce5fb87a95d3e72960c060b986dd2733cac144d5f008e661
checkpoints/60000/assets/RoboDojo_lerobot_v30_video/norm_stats.json
b3d1307c5f8b1c0334fe8377cb5aef422067b22204fb1529fb5c8289a98d2c68
delivery/dataset_meta/tasks.parquet
39e934f1eea211fc471be6a32cfd920fd5bc5a8c06a01c819cfc7e68b6990f65
```

The two normalization files have different formatting but identical parsed JSON values. The [public Full35 router-label release](https://github.com/gleeacast/KinRT/releases/download/robodojo-full35-router-labels-v1/router_labels_k4_full35.npy) was downloaded independently and has the same bytes and digest as the delivery copy. All 14 entries in `delivery/delivery_files.sha256` passed local checks. The final checkpoint's 1,057 manifest paths all exist in the pinned remote revision; that inventory check is not a substitute for downloading and hashing the weights needed on the execution host.

The adapter's `--assets-only` downloader was exercised against the pinned private snapshot and downloaded 17 small files successfully. All 12 portable asset-helper tests passed (`python -m unittest discover -s policy/KinRT/tests` from the XPolicyLab root). Additional local checks covered six training-wrapper cases, nine source compatibility cases using the pinned upstream files, and two converter lifecycle cases. These checks do not execute the model.

## Configuration

### Observation And Action Interface

| RoboDojo field | KinRT input |
| --- | --- |
| `cam_head` | `cam_high` |
| `cam_left_wrist` | `cam_left_wrist` |
| `cam_right_wrist` | `cam_right_wrist` |
| left arm, left gripper, right arm, right gripper | 14-D state/action vector |

Images remain RGB throughout conversion, training, and evaluation. The policy server decodes runtime image buffers before `model.py` receives them. The model predicts 50-step action chunks.

### Variables

| Variable | Purpose |
| --- | --- |
| `KINRT_OPENPI_ROOT` | KinRT `policy/pi05` checkout and installed policy environment. |
| `KINRT_SOURCE_REPO` / `KINRT_SOURCE_REV` | Source repository and pinned KinRT revision used by installation. |
| `KINRT_LEROBOT_ROOT` | Optional clean LeRobot checkout at the pinned v0.4.4 commit; defaults to the `LeRobot_KinRT_Full35` sibling. |
| `KINRT_ALLOW_UNPINNED_SOURCE` | Development opt-out from the source revision check. |
| `KINRT_PYTHON_BIN` | Python executable override for policy wrapper commands and asset downloads. |
| `HF_TOKEN` | Hugging Face credential supplied through the environment; private repository access is required. |
| `KINRT_HF_REPO_ID` / `KINRT_HF_REVISION` | Model repository/revision overrides; defaults are the pinned Full35 delivery. |
| `HF_LEROBOT_HOME` | Parent directory of local training datasets. |
| `KINRT_ROBODOJO_REPO_ID` | Training dataset ID; defaults to `RoboDojo_lerobot_v30_video` in `train.sh`. Custom conversion uses a separate ID. |
| `KINRT_ROBODOJO_SOURCE_DIR` | Optional source embodiment directory for custom HDF5 conversion. |
| `KINRT_LEROBOT_METADATA_FPS` | Metadata FPS for custom conversion; published Full35 metadata uses 25. |
| `KINRT_OVERWRITE_DATASET` / `KINRT_OVERWRITE_ROUTER_LABELS` / `KINRT_OVERWRITE_NORM_STATS` | Set the relevant flag to `1` only for intentional custom dataset/label/statistics regeneration. |
| `KINRT_ROBODOJO_ROUTER_LABELS_PATH` | Explicit per-frame router-label path. |
| `OPENPI_TRAIN_CONFIG_NAME` | Training configuration; Full35 default is `kinrt_full_robodojo`. |
| `OPENPI_BASE_CHECKPOINT` | Pi 0.5 base parameter path for starting training. |
| `OPENPI_NUM_TRAIN_STEPS` / `OPENPI_BATCH_SIZE` | Full35 defaults: 60000 steps and global batch 256. |
| `OPENPI_NUM_WORKERS` / `OPENPI_SAVE_INTERVAL` | Defaults: 8 DataLoader workers and saves every 5000 steps. |
| `OPENPI_FSDP_DEVICES` | FSDP device count; defaults to the supplied GPU count, 8 in the delivered run. |
| `OPENPI_WANDB_ENABLED` | Set to `0` to disable Weights & Biases. |
| `KINRT_RESUME` | Set to `1` to resume an existing training run. |
| `KINRT_TRAIN_CONFIG_NAME` / `KINRT_REPO_ID` | Evaluation configuration and normalization key: `kinrt_full_robodojo` / `RoboDojo_lerobot_v30_video`. |
| `KINRT_CHECKPOINT_PATH` | Evaluation checkpoint; default `checkpoints/KinRT-RoboDojo-Full35-60k/checkpoints/60000` relative to this adapter. |
| `KINRT_CHECKPOINT_NUM` | Preferred checkpoint step; default 60000. |
| `KINRT_ACTION_CHUNK_SIZE` | Actions executed per inference call; default 50. |
| `KINRT_EXTRA_PYTHONPATH` | Optional dependency path for isolated testing. |

## Limitations

- Full model inference, debug transport, and simulator evaluation of this 60k checkpoint are pending. The local Windows host has 8 GB VRAM and cannot validate this Linux JAX CUDA execution path.
- The repository is private. Leaderboard evaluators must obtain access before downloading; the integration does not change repository visibility.
- The full original training dataset is not included. Delivered router labels require its original frame ordering, and the metadata alone cannot reconstruct that ordering or the demonstrations.
- The documented training-time DataFrame compatibility patch was not delivered; installation contains a reconstructed fix, not the exact original patch. Delivered lockfiles also record a training-time package-mirror change.
- There is no matched 60k Pi 0.5 baseline in this delivery. Earlier 10k single-task metrics and baseline results do not establish Full35 performance.
- The adapter targets joint control for dual ARX-X5 robots. Simulator evaluation requires the complete RoboDojo Isaac Sim environment and assets.
- Batched simulation evaluation uses sequential model inference within each batch to bound accelerator memory use. The 60k model has not yet been exercised through that path.
