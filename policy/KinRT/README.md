# KinRT for RoboDojo

**Contributor:** Tianhang Yang, Yanze Zheng, Junjie Wang, Wei-Bin Kou, Ruotong Li, Yujiu Yang | **Paper:** Route by Kinematics, Act by Observation | **arXiv:** https://arxiv.org/abs/2607.26807 | **Original code:** https://github.com/gleeacast/KinRT

This adapter applies KinRT to RoboDojo's dual-ARX-X5 environment: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, and `action_type=joint`. The default model is the delivered **Full35 checkpoint at 60,000 training steps**, trained with full parameter fine-tuning on 35 tasks, 3,500 episodes, and 1,859,602 frames. Its configuration is `kinrt_full_robodojo`; its dataset and normalization key is `RoboDojo_lerobot_v30_video`.

KinRT source remains in a separate checkout; this directory contains the XPolicyLab integration. Complete GPU restoration, all-prompt WebSocket checks, and standard raw/encoded debug loops passed. The ongoing seed-0 simulator evaluation has completed `stack_bowls` at **22/25 (88%)** and `stack_bowls_random` at **4/25 (16%)**; these are two configurations, not a complete Full35 result.

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

The path is optional with this sibling layout. Installation pins KinRT to `590d52802cde804cdc2d0ccb672c1a3a90d76f91` and runs `uv sync --frozen --no-default-groups`. `KINRT_PYPI_MIRROR` selects `pypi` (default), `tencent`, or `original`; a guarded helper temporarily changes only lockfile mirror URLs after TOML semantic validation, preserving versions and hashes and restoring the original `uv.lock` afterward. `pyproject.toml` remains unchanged. The installer then adds the adapter dependencies.

LeRobot is checked out separately at `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` (v0.4.4, dataset format v3.0). An environment-local `kinrt_full35_lerobot.pth` gives this checkout's `src/` import precedence, reproducing the delivered run's `PYTHONPATH` source overlay while preserving the older locked dependencies. A successful install verifies the actual KinRT adapter import path, the selected LeRobot version, and OpenCV import; it does not validate checkpoint loading or GPU inference. The installer explicitly adds `pytest==9.0.3`, which upstream model modules import even when development dependency groups are disabled, and reinstalls headless OpenCV after removing the GUI wheel's shared files.

Installation and GPU execution have also passed in a newly created Linux policy environment with the pinned source overlays: JAX/JAXlib `0.5.0`, Flax `0.10.2`, Orbax `0.11.1`, PyTorch `2.6.0`, and NumPy `1.26.4`. Checks covered dependency synchronization, tokenizer download and digest verification, source preparation, original-lockfile restoration, the actual KinRT adapter path, LeRobot v3.0 source selection, and OpenCV `4.11.0`. Full checkpoint restoration, finite synthetic inference, and both standard raw/encoded debug loops passed in this new environment. Its simulator pair replay remains in progress; the simulator installation is still a separate prerequisite. See the [fresh runtime evidence](evidence/fresh_runtime_checks.json).

The delivery notes document a training-time `data_loader.py` fix for LeRobot task tables, but do not include its original patch. Installation applies a reconstructed compatibility fix that maps the DataFrame's `task_index` column to its prompt index. It also adapts the upstream router-label generator to read v3 episodes packed into or split across Parquet files, filtering by episode and sorting by global frame index without changing clustering or feature calculations. The converter finalizes v3 dataset writers. These changes are explicit; the source commit alone does not contain every modification used for the delivered training run.

From the adapter directory, `python prepare_full35_source.py "$KINRT_OPENPI_ROOT" --check` validates the known source without changing it; the same command without `--check` applies the fixes. `--revert` restores only this helper's exact changes. Both source files are checked before either is modified; unknown source edits are rejected. The installer performs the check and apply steps automatically.

The model repository is public and ungated. The default checkpoint download does not require an HF account, login, or token. Anonymous access was verified using a new cache with implicit authentication disabled; changing visibility did not change the pinned model revision.

The [reproduction guide](REPRODUCING.md) gives the complete evaluation prerequisites and commands, including public checkpoint download, first-use tokenizer caching, the RoboDojo material-path helper, and the single-environment setting used when the policy and simulator share a 24 GiB GPU. The installer also provisions the separate PaliGemma tokenizer through `prepare_tokenizer.py`, verifies SHA-256 `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`, and caches it under `OPENPI_DATA_HOME` (default `~/.cache/openpi`). Keep that cache setting for evaluation; a verified local tokenizer file can be supplied as described in the guide.

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

For the measured seed-0 simulator protocol, follow [REPRODUCING.md](REPRODUCING.md). It includes `prepare_robodojo.py`, which is a separate explicit preparation step and is not invoked by the policy installer. On a shared 24 GiB GPU, use one simulator environment and `XLA_PYTHON_CLIENT_ALLOCATOR=platform` as documented there.

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

### Verified Execution Scope

The complete inference checkpoint was restored and exercised on Ubuntu 22.04 with an RTX 3090 24 GiB GPU and NVIDIA driver `580.95.05`. Inputs were synthetic; the WebSocket checks substituted each of the 35 delivered task instructions and validated action shape and finite values.

| Check | Observed result |
| --- | --- |
| Complete checkpoint restoration | 13.469 s |
| First synthetic inference, including first-call compilation | 13.279 s; finite `50 x 14` action chunk |
| Real WebSocket server, all 35 task instructions | 35/35 calls returned finite `50 x 14` chunks |
| Warmed WebSocket inference, calls 2-35 | Mean 0.257 s per chunk in this run |
| Encoded RGB observations, batch environment indices `[2, 7]` | Two valid `50 x 14` chunks in 0.525 s |
| Standard `eval.sh` debug, raw and encoded observations | Both modes finished; 10 default episodes per mode, batch size 10 |
| Peak observed device memory during smoke/RPC checks | 17,886 MiB, including a 1,195 MiB idle/display baseline |

These timings describe this smoke test, including RPC transport where applicable; they are not a throughput benchmark or a task-success measurement. The batch check exercised the adapter's sequential per-environment inference through the actual server. Evaluation launchers default to `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`; both can be overridden for the host's memory budget.

Both standard `EVAL_ENV_TYPE=debug` runs reached `[MAIN] eval finished`, recorded in `logs/debug_encoded_0.log` and `logs/debug_encoded_1.log`; the runner also wrote `debug.complete`. Each mode completed 10 synthetic debug episodes with batch size 10. These checks validate the standard client/server lifecycle and action interface, not manipulation success.

The separate fresh policy environment repeated full checkpoint restoration and synthetic inference successfully, followed by both standard raw/encoded debug modes. Its `50 x 14` synthetic action array exactly matched the original environment's array (`numpy.array_equal=true`, maximum absolute difference `0.0`); this comparison covers one identical input, not all possible trajectories. Both new debug logs reached `[MAIN] eval finished` at line 461. The [fresh runtime evidence](evidence/fresh_runtime_checks.json) retains log and action-file digests and distinguishes these completed checks from the ongoing simulator replay.

### Simulator Results So Far

| Configuration | Seed | Native episodes | Successes | Success rate |
| --- | --- | --- | --- | --- |
| `stack_bowls` | 0 | 25 | 22 | 88% |
| `stack_bowls_random` | 0 | 25 | 4 | 16% |

The [reference evidence](evidence/seed0_reference_pair.json) records all 50 episode outcomes, run IDs, source-result SHA-256 values, model and simulator revisions, and the execution order. It identifies these measurements as the original compatible-environment run; it is not evidence of a fresh-install replay. Success percentages come from the boolean episode outcomes, not the simulator's separate partial-credit score.

These measurements use corrected material paths, one simulator environment, joint control, and action chunks of 50. A single fresh policy server ran the 25 `stack_bowls` episodes first, then the 25 `stack_bowls_random` episodes, without prior synthetic inference or a server restart. The policy RNG advances on every inference and is not reset by the adapter's episode reset, so two separate `eval.sh` invocations have a different RNG history. Follow the single-server sequence in [REPRODUCING.md](REPRODUCING.md) for the reported protocol.

An earlier run with incorrect tabletop materials was stopped and excluded. The two completed configurations are the fixed/random pair of one canonical task. They do not establish a Full35 average or guarantee identical outcomes on other hardware, simulator versions, or rendering settings.

The delivered metadata confirms training on **35 tasks, 3,500 episodes, and 1,859,602 frames**. The number 34 refers only to currently available simulator tasks: the spelling task's simulation code, configuration, and layouts are missing. The available evaluation scope expands to 46 configurations and 1,700 native episodes: 22 single configurations with 50 episodes each, plus 12 fixed/random pairs with 25 episodes per configuration. That is the planned scope, not the completed count. The fresh environment's 25+25 pair replay is in progress. Planned continuation over the other 44 configurations / 1,650 episodes starts a new policy server and records that RNG boundary; it is not one uninterrupted server run with the pair.

## Model Assets

The default download is pinned to [Gleez/kinrt-robodojo-full35-a800-60k](https://huggingface.co/Gleez/kinrt-robodojo-full35-a800-60k/tree/9460d07a9c7677ef3c72ece08df1f34eba7e45c7), revision `9460d07a9c7677ef3c72ece08df1f34eba7e45c7`.

```bash
bash download_checkpoint.sh [DESTINATION] [--assets-only | --include-training-state]
```

The default destination is `policy/KinRT/checkpoints/KinRT-RoboDojo-Full35-60k/`, resolved relative to the adapter directory. A normal download includes 60k model parameters, checkpoint metadata and normalization, and all delivery files. It excludes the 50k checkpoint and optimizer/training state. The two optional modes are mutually exclusive: `--assets-only` downloads delivery files and the checkpoint normalization without model weights; `--include-training-state` additionally retrieves the 60k training state.

The downloader prefers the installed policy environment's `.venv/bin/python`, falling back to `python` when it is absent. `KINRT_PYTHON_BIN` overrides that choice. The chosen Python must provide the download and verification dependencies. No authentication is required for the default public repository.

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

The two normalization files have different formatting but identical parsed JSON values. The [public Full35 router-label release](https://github.com/gleeacast/KinRT/releases/download/robodojo-full35-router-labels-v1/router_labels_k4_full35.npy) was downloaded independently and has the same bytes and digest as the delivery copy. On the Linux execution host, all **353 inference-checkpoint files** and all **14 delivery-manifest entries** were downloaded from the pinned revision and passed SHA-256 verification before model restoration. The final checkpoint's full 1,057 manifest paths exist remotely; the additional 704 training-state files are outside this inference download and were not needed for the execution checks.

The adapter's `--assets-only` downloader was exercised anonymously against the pinned public snapshot using a new HF cache with credential variables absent and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. It downloaded 17 files and verified all 14 delivery entries plus the checkpoint normalization. An anonymous request also verified the normalization digest, and an HTTP HEAD request for an actual model-weight shard returned 200. The anonymous check did not redownload all model weights; the full 353-file inference checkpoint had already passed verification and GPU execution.

All 12 portable asset-helper tests passed. Additional checks covered six training-wrapper cases, nine source compatibility cases using pinned upstream files, and two converter lifecycle cases. All 12 installer-helper tests and all 17 simulator-preparation tests passed on Linux, including real signal cleanup and symlink cases. These helper tests are separate from model execution.

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
| `UV_BIN` / `KINRT_PYPI_MIRROR` | Optional uv executable; lockfile download mirror `pypi` (default), `tencent`, or `original`. |
| `KINRT_ALLOW_UNPINNED_SOURCE` | Development opt-out from the source revision check. |
| `KINRT_PYTHON_BIN` | Python executable override for policy wrapper commands and asset downloads. |
| `HF_TOKEN` | Optional Hugging Face credential; not required for the default public repository. |
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
| `XLA_PYTHON_CLIENT_PREALLOCATE` / `XLA_PYTHON_CLIENT_MEM_FRACTION` | Evaluation defaults: `false` / `0.8`; adjust for the available GPU memory. |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | Use `platform` for the documented shared-GPU simulator reproduction. |
| `OPENPI_DATA_HOME` | Persistent OpenPI download cache, including the PaliGemma tokenizer. |
| `KINRT_EXTRA_PYTHONPATH` | Optional dependency path for isolated testing. |

## Limitations

- Complete GPU restoration, synthetic inference, all-prompt WebSocket checks, and standard raw/encoded debug loops passed on the RTX 3090 host. The reported simulator results cover only the seed-0 fixed/random stacking-bowls pair; the remaining suite is not a completed result.
- Installation, full checkpoint restoration, synthetic inference, and both standard debug modes passed in the new Linux policy environment. Its 25+25 simulator replay remains in progress. The installer does not provision RoboDojo or Isaac Sim.
- The model repository is public and ungated; anonymous download has passed. The complete original training dataset is separate from these inference assets.
- The full original training dataset is not included. Delivered router labels require its original frame ordering, and the metadata alone cannot reconstruct that ordering or the demonstrations.
- The documented training-time DataFrame compatibility patch was not delivered; installation contains a reconstructed fix, not the exact original patch. Delivered lockfiles also record a training-time package-mirror change.
- There is no matched 60k Pi 0.5 baseline in this delivery. Earlier 10k single-task metrics and baseline results do not establish Full35 performance.
- The adapter targets joint control for dual ARX-X5 robots. Simulator evaluation requires the complete RoboDojo Isaac Sim environment and assets.
- The available simulator checkout covers 34 of the 35 training tasks; the spelling task lacks code/configuration/layouts. No complete 35-task success rate, multi-seed result, or exact cross-hardware reproduction is claimed.
- Batch inference processes environments sequentially to bound accelerator memory use. An encoded two-environment WebSocket batch passed, but batched simulator task success has not been measured.
