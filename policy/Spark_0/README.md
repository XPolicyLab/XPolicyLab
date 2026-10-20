# Spark_0

**Contributor:** Tianxing Chen | **Paper:** Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation | **arXiv:** https://arxiv.org/abs/2610.22342 | **Original code:** https://github.com/XsparkAI/Spark-0 | **Project page:** https://xsparkai.com/spark-0

`Spark_0` adapts Spark-0, a spatially aligned vision-tactile-language-action (VTLA) policy for bimanual dexterous hands, to XPolicyLab. Touch plays two roles in one pretraining run: image-space contact maps and a frozen Depth-Anything-3 teacher supervise the VLM's visual tokens on *where* contact occurs (train-only, removed at inference), and a tactile encoder conditions the action expert on *how* contact evolves. One checkpoint therefore serves both vision-only and tactile-conditioned deployment. The model code is a vendored, patched copy of StarVLA in `source_starvla/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md).

## Installation

Spark-0 lives under `XPolicyLab/policy/Spark_0/` and imports `XPolicyLab.*`; keep `deploy.yml` `policy_name` equal to the directory name. The scripts assume the standard XPolicyLab workspace, with trajectories and robot configs beside the checkout:

```text
<workspace>/
├── data/<bench_name>/<task>/<env_cfg_type>/data/*.hdf5   # process_data.sh input
├── env_cfg/<env_cfg_type>.yml                             # robot dims for eval
└── XPolicyLab/policy/Spark_0/                             # this adapter
```

`env_cfg_type` is `tianji_marvin_wuji` for the Tianji + Wuji Hand 2 robot everywhere (data directory, `process_data.sh`, `train.sh`, eval); `scripts/finetune.sh` and `scripts/precompute_stats.sh` default to it (`ENV_CFG_TYPE` overrides). A standalone clone outside `XPolicyLab/policy/` cannot install XPolicyLab through `install.sh` and resolves `data/` to the wrong place; set `XPL_DATA_ROOT` and `pip install -e <XPolicyLab>` by hand if you must work that way.

```bash
cd XPolicyLab/policy/Spark_0
conda create -n spark_0 python=3.10 -y && conda activate spark_0
bash install.sh   # torch 2.6, flash-attn, vendored StarVLA, and XPolicyLab itself
```

`install.sh` pins Pillow because training and evaluation both resize through PIL's default kernel; keep the pin so the two paths stay pixel-identical.

## Model Assets

Weights are expected under `source_starvla/playground/Pretrained_models/`:

```text
source_starvla/playground/Pretrained_models/
├── Qwen3-VL-4B-Instruct/   # base VLM
└── DA3-LARGE/              # Depth-Anything-3 teacher, train-only: bash scripts/download_da3_large.sh
```

Download these weights, plus the Spark-0 pretraining checkpoint, with:

```bash
bash scripts/download_pretrained_ckpt.sh
```

`scripts/finetune.sh` checks the DA3 teacher before launch (`scripts/check_da3_teacher.sh`); set `SKIP_DA3_CHECK=1` to skip. The teacher is never loaded at inference.

## Data Processing

Training reads **LeRobot v3** datasets in a Spark-specific layout, **not** the keys of the official converters ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)): state and action are end-effector poses plus Wuji hand joints, cameras are declared through `meta/modality.json`, and tactile streams are extra columns. Official exports cannot be fed to this trainer directly.

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: <workspace>/data/SparkArena/sft_spark_0/tianji_marvin_wuji/data/*.hdf5
#          -> data/SparkArena-sft_spark_0-tianji_marvin_wuji-ee/
bash process_data.sh SparkArena sft_spark_0 tianji_marvin_wuji ee
```

`process_data.sh` runs `tools/transform_lerobot_v30_format_tianji_ee.py`. It reads `<workspace>/data/<bench_name>/<task>/<env_cfg_type>/data/*.hdf5` (`XPL_DATA_ROOT` overrides the `data/` location), where `<task>` is each entry of `raw_task_dirs` (comma-separated, default `ckpt_name`), and writes the LeRobot v3 dataset to `data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/`, the path `train.sh` derives from the same four arguments. `action_type` must be `ee`. The converter requires `vision/cam_head/extrinsics` in every episode and exports tactile columns when the episodes carry them. Other sources:

| Tool | Purpose |
| --- | --- |
| `tools/batch_retarget_camera_aligned.py`, `scripts/run_retarget_batch.sh` | Human / other-robot datasets → retargeted Spark HDF5 (see below and [tools/README.md](tools/README.md)) |
| `tools/transform_lerobot_v30_format_tianji_ee.py` | Spark HDF5 → LeRobot v3 with tactile, contact heatmap and MANO columns |
| `tools/repack_lerobot_row_groups.py <dataset_dir>` | Repack exports written with one parquet row group per file, for fast random frame reads |

### Cross-embodiment retargeting

Pretraining data is not recorded on Tianji + Wuji, so every source is first retargeted into the same 54-d layout (Tianji EE poses + Wuji Hand 2 joints) and then exported to LeRobot v3. Retargeting has two layers:

1. **Arm / EE.** Human wrists are re-expressed once into the Spark head-camera mount frame (`camera_alignment.py`). The initial camera window is placed at the world origin with the mount orientation in `spark_camera_alignment.json` (RPY 30/0/0); no translation is applied. A fixed per-hand orientation offset matches the initial palm to `target_ee_orientation_wxyz`. Training reads EE poses in the head-camera frame, so that world placement cancels and does not change model inputs. Robot datasets keep their own base frame.
2. **Hand.** 5×5 finger keypoints are morphology-normalized and solved to Wuji Hand 2 joints by IK (`retargeting/wuji2.py`). Every source writes the same 20-d Wuji Hand 2 order (Isaac order, EE = `Link7`).

| Source | Hand input | Hand path |
| --- | --- | --- |
| `human_touch`, `openaoe`, `egodex`, `egoverse` | Human hand keypoints / MANO | Camera alignment + Wuji IK |
| `fourier_actionnet`, `agibot_world_beta` | Fourier / Inspire hand joints | Robot-hand FK → keypoints → Wuji IK |
| `egovla_bench` | Inspire 12-d joints | Direct joint mapping, no IK |

AgiBot World Beta keeps the flange translation as the EE position and right-multiplies the built-in `INSPIRE_MOUNT_ROTATION` constant in `tools/data_alignment/datasets/agibot_world/reader.py` for EE orientation and the Inspire FK parent. See [tools/README.md](tools/README.md#source-robot-datasets).

For a bench whose hand already has a joint-level counterpart on Wuji, a direct joint mapping replaces the keypoint IK. This is implemented for the EgoVLA bench: `datasets/egovla_bench/hand_mapping.json` copies each of the 12 Inspire joints to its matching Wuji joint by index and sign. The 8 Wuji joints Inspire lacks (four MCP abductions, four DIPs) are set to 0, and no clipping is applied. The inverse, `HandCodec.wuji20_to_inspire12`, maps Wuji predictions back to Inspire. It is not yet wired into `model.py`, so evaluating on the EgoVLA bench still needs that action mapping on the deploy side.

URDFs (Wuji Hand 2, Inspire, Fourier) and MANO models are not bundled; place them under `tools/data_alignment/assets/` or pass the paths listed in [tools/README.md](tools/README.md#required-assets).

| Stream | Packed shape | Source column |
| --- | --- | --- |
| State / action | `(58,)` after pose conversion; 54-d on disk | `[L_ee_pose(7), L_hand(20), R_ee_pose(7), R_hand(20)]`, poses `[x, y, z, qw, qx, qy, qz]` |
| Cameras | RGB, resized to `obs_image_size` | `video.cam_high`, `video.cam_left_wrist`, `video.cam_right_wrist` |
| Region force | `[hands, regions, history, 1]` | `tactile.region_pressure_vector.{left,right}_ee` |
| 3D taxel cloud | `[hands, taxels, T, 4]` (xyz + pressure) | `taxel_pressure_3D.{left,right}_ee` or converter `3d_left` / `3d_right` |
| Contact heatmap | `[H, W]` | `tactile.2d` (train-only, for `tactile_align`) |

Tactile columns are packed only when a tactile stem is on. Stored image bits are read with `decode_image_bit` and written with `encode_image_bit` from `XPolicyLab.utils.process_data` (RGB end to end); never add a `BGR2RGB` swap after decoding.

## Training

`train.sh` follows the standard signature plus a required recipe — a YAML stem under `model_configs/` or a path (also settable via `SPARK_0_TRAIN_CONFIG`). Normalization statistics must be precomputed first (CPU only, once per data or stats-relevant config change):

```bash
bash scripts/precompute_stats.sh <recipe>
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> <recipe> [omegaconf_overrides...]

# Example: finetune on four GPUs
bash scripts/precompute_stats.sh sft_spark_0
bash train.sh SparkArena sft_spark_0 tianji_marvin_wuji ee 0 0,1,2,3 sft_spark_0
```

`scripts/finetune.sh <recipe> [gpus] [overrides...]` is a shortcut that names the run after the recipe (`checkpoints/SparkArena-<recipe>-tianji_marvin_wuji-ee-<seed>/`; override with `BENCH_NAME` / `CKPT_NAME` / `ENV_CFG_TYPE` / `SEED`, batch size with `PER_DEVICE_BS`):

```bash
bash scripts/finetune.sh sft_spark_0                                   # GPUs 0-7
bash scripts/finetune.sh sft_spark_0 0,1 trainer.max_train_steps=20000
```

| Recipe | Purpose | Schedule |
| --- | --- | --- |
| `pretrain_spark_0` | Visual + tactile pretraining, `sample_without_replacement: true`, DA3 teacher at 336 px | 500k steps, 20k warmup, save every 50k |
| `sft_spark_0` | Finetune from the pretrain checkpoint, `use_pretrained_stats: true`, `rtc: default`, DA3 teacher at 392 px | 80k steps, 8k warmup, save every 10k |

Both recipes enable DA3 alignment, flexible views, action queries, `tactile_fusion: default`, and `tactile_align: default`, and leave `data_mixture: null` (the positional dataset path is used and must contain `meta/modality.json`). To start a new run, copy a recipe, set `data_mixture` and `trainer.pretrained_checkpoint`, and launch it by stem.

- **Statistics.** `train.sh` sets `SPARK_0_REQUIRE_PRECOMPUTED_STATS=1`; every rank reads content-addressed caches `<dataset>/meta/stats_xpolicy*__<hash>.json` and a missing cache fails the launch with its path. Changing data, pose formats, `state_history_frames`, `action_horizon`, `exclude_episodes`, or the embodiment produces a new hash.
- **Warm start vs. resume.** `trainer.pretrained_checkpoint` loads weights with a fresh step counter; `trainer.resume_from` also continues the step counter. Optimizer moments are not saved.
- **Multi-node.** Run the same command on every node with a unique `SPARK_0_MACHINE_RANK` and the same `SPARK_0_NUM_MACHINES`, `SPARK_0_MAIN_PROCESS_IP` (rank-0's reachable address), `SPARK_0_MAIN_PROCESS_PORT`. Set `SPARK_0_GPUS_PER_NODE` to turn a GPU-count mismatch into a launch error instead of a hang.
- **DeepSpeed.** `SPARK_0_ACCELERATE_CONFIG` takes an accelerate YAML: `source_starvla/starVLA/config/deepseeds/zero2.yaml` or `accelerate_zero3.yaml` (not the raw `zero3.yaml`).
- **NaN fuse.** Training aborts every rank on a non-finite loss or gradient before the optimizer step; reload the last saved checkpoint.
- **Monitoring.** `eval_train/fixed_batch_mse` is the fit on a fixed training batch, not a held-out metric. Read `tactile_gate` / `tactile_patch_gate` as on/off switches and `tactile_extra_temb_rel_norm` as the force-path amplitude; whether touch helps needs a paired eval of one checkpoint with and without tactile.

## Evaluation

```bash
cd XPolicyLab/policy/Spark_0
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the SFT run on GPUs 0 (policy) and 1 (environment).
# <ckpt_name> is the training CKPT_NAME; the run dir
# checkpoints/SparkArena-sft_spark_0-<env tag>-ee-0 is resolved for any env tag.
bash eval.sh SparkArena <task_name> sft_spark_0 tianji_marvin_wuji ee 0 0 1 spark_0 <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check, `sim` (or unset) a simulator, `real` the robot client. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

`ckpt_name` may be the short recipe name, the full run-directory name, or a path; `SPARK_0_CKPT_PATH=/path/to/pytorch_model.pt` pins an explicit file. Eval replays the checkpoint's saved config (pose formats, normalization, cameras, state history, tactile layout), so keep `auto` in `deploy.yml`.

Spark-0 specifics of the `Model` contract:

- `get_action` returns **one action row per call**. The model re-infers once per chunk and serves the other rows from a cache, which keeps the state-history frames consecutive as in training.
- **The policy server is single-tenant.** Per-episode buffers are keyed by `env_idx` only: run one server per eval client, or give each client a disjoint `env_idx` range.
- With `FAST_OBS_UPDATE = True` (top of `deploy.py`), the client sends a full observation on a fixed period from the local model config (sync: `execute_horizon`; RTC: `action_horizon - fire_at_remaining`) and keeps the state-history ring buffer itself.
- **Tactile input.** The bench publishes the current history window under `obs["tactile"]` in raw sensor units: `region_pressure` `[hands, regions, history]` (oldest → current) and/or `taxel_pressure_3d` `[hands, taxels, T, 4]` (xyz in metres in the training frame, plus pressure). A stream the bench omits falls back to the learned `no_tactile` token, so the same checkpoint runs vision-only.

## Configuration

A run is one recipe YAML plus shared stems that `train.sh` merges and freezes with the checkpoint (do not set the stems' `enable` flags or `run_id` in a recipe):

```text
model_configs/<recipe>.yaml
├─ embodiment: tianji_marvin_wuji  -> model_configs/embodiments/     (robot keys, dims, camera catalog)
├─ execute_horizon: null | N       deploy-time
├─ rtc: null | <stem>              -> model_configs/rtc/             deploy-time
├─ tactile_fusion: null | <stem>   -> model_configs/tactile_fusion/  (encoder + action-expert fusion)
└─ tactile_align: null | <stem>    -> model_configs/tactile_align/   (train-only contact alignment)
```

| Switch | Key | Spark-0 recipes |
| --- | --- | --- |
| Tactile encoder + fusion | `tactile_fusion` | `default` (320-taxel full-hand glove); `fingertip_only` for 80-taxel fingertip exports; `null` = off, identical to a vision-only model |
| Contact-heatmap alignment | `tactile_align` | `default` (head camera); `null` = off |
| DA3 geometry distillation | `framework.spatial_alignment.enable` | `true`, `align_layer_index: 25` (must be odd) |
| VLM action queries | `framework.action_query` | 64 queries, `kv_mode: query_plus_visual` |
| Shortcut mask | `framework.action_model.shortcut_mask_prob` | `0.1`: hides the VLM from the action expert on some tactile rows so touch is not ignored |
| Flexible camera sets | `datasets.vla_data.flexible_views` | `true`: head-only, one-wrist, and two-wrist datasets co-train |
| State history | `datasets.vla_data.state_history_frames` | `3` |
| Pose formats | `datasets.vla_data.{state,action}_pose_format` | `cam_abs_xyz_rot6d` / `delta_xyz_rot6d` |
| Pose normalization | `datasets.vla_data.pose_normalization` | `hold_centered_perrow` |
| Video decoder | `datasets.vla_data.video_backend` | `pyav` |

For a vision-only run, set both tactile stems to `null` and `shortcut_mask_prob: 0`. The dataloader asserts that each dataset's tactile layout matches the selected stem. In `data_mixture`, weights are sampling probabilities; delete a member instead of setting `weight: 0` (which raises).

Deploy-time knobs are read from the **live** model config, found via `SPARK_0_MODEL_CONFIG`, then `model_configs/<ckpt_name>.yaml`, then the recipe stem recorded in the checkpoint: `execute_horizon: null` re-plans once per 50-row chunk, `N` every `N` rows; `rtc: default` prefetches the next chunk asynchronously (re-plans every 16 steps: `action_horizon` 50 minus `fire_at_remaining` 34) and overrides `execute_horizon`.

`deploy.yml` keys beyond the standard set:

| Key | Default | Purpose |
| --- | --- | --- |
| `Spark_0_server_port` | `5694` | Internal Spark_0 server behind the XPolicyLab policy server (`port: 6000`) |
| `unnorm_key`, `image_size`, `include_state`, `state_history_frames`, `ee_in_camera_frame`, `max_infer_batch` | `auto` | Replayed from the checkpoint |
| `flexible_views` / `required_views` | `auto` | Missing wrists are fine if training used `flexible_views` |
| `ee_camera_pose`, `ee_camera_extrinsics_semantics`, `ee_camera_axes` | `auto` | Observation camera convention |
| `tactile` | `auto` | `false` disables the glove to A/B the same checkpoint |
| `tactile_fast` / `tactile_slow` | `auto` | `false` drops the region-force / 3D-taxel stream |

Environment variables:

| Variable | Notes |
| --- | --- |
| `SPARK_0_DATA_ROOT` / `SPARK_0_CKPT_ROOT` | Data / checkpoint roots |
| `SPARK_0_TRAIN_CONFIG` | Recipe when `train.sh` gets no 7th argument |
| `SPARK_0_RESUME_FROM` | Override `trainer.resume_from` |
| `SPARK_0_ACCELERATE_CONFIG` | Accelerate YAML for DeepSpeed ZeRO-2 / ZeRO-3 |
| `SPARK_0_NUM_MACHINES`, `SPARK_0_MACHINE_RANK`, `SPARK_0_MAIN_PROCESS_IP`, `SPARK_0_MAIN_PROCESS_PORT`, `SPARK_0_GPUS_PER_NODE` | Multi-node launch |
| `NCCL_TIMEOUT` | Collective watchdog in seconds (default 600) |
| `SPARK_0_REQUIRE_PRECOMPUTED_STATS` | `1` (default) requires stats caches; `0` computes them on rank 0 at launch |
| `SPARK_0_STATS_MODE` | `auto` / `memory` / `streaming` (streaming keeps RAM near 1 GB) |
| `SPARK_0_TACTILE_STATS_READ_WORKERS` | Read threads for tactile statistics (default 32) |
| `SPARK_0_CKPT_PATH` | Explicit eval checkpoint file |
| `SPARK_0_MODEL_CONFIG` | Live model config for the eval server / client |
| `SPARK_0_CONFIG_OVERRIDE` | Eval-only inference overrides, e.g. `num_inference_timesteps`, `rtc.*` |
| `SPARK_0_WANDB_ENTITY` / `SPARK_0_WANDB_PROJECT` / `WANDB_MODE` | Weights & Biases (offline by default) |
| `SKIP_DA3_CHECK` | `1` skips the DA3 teacher check |
| `XPL_DATA_ROOT` | `process_data.sh` input `data/` directory (default: `<workspace>/data`, beside `XPolicyLab/`) |
| `SPARK_WORKSPACE_ROOT` | Root for relative paths in retarget run configs (default: this directory) |

## Notes

- **Vendored upstream.** `source_starvla/` tracks [starVLA/starVLA](https://github.com/starVLA/starVLA) at commit `c00b8e1` (2026-07-20) plus local patches: tactile encoder and fusion, contact alignment, DA3 alignment, action queries, flexible views, RTC, precomputed / streaming statistics, mixture sampling fixes, ZeRO-3 safety, and the NaN fuse. Diff against that commit when re-syncing.
- Data, checkpoints, and eval logs land in git-ignored `data/`, `checkpoints/`, and `eval_logs/`.

## File Structure

| Path | Purpose |
| --- | --- |
| `install.sh` / `process_data.sh` / `train.sh` / `eval.sh` | Install, convert data, train, same-machine eval |
| `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` | Split-machine policy server / environment client |
| `deploy.yml` / `deploy.py` / `model.py` | Deployment config, eval loop, XPolicyLab model adapter |
| `model_configs/` | Recipes plus `embodiments/`, `rtc/`, `tactile_fusion/`, `tactile_align/` stems |
| `scripts/` | `finetune.sh`, `precompute_stats.sh`, DA3 check / download, retarget batch and its configs |
| `tools/action_codec.py`, `tools/ee_delta_action.py`, `tools/relative_pose.py` | Env-space ↔ model-space action decoding and pose math |
| `tools/runtime_config.py` | Resolves `auto` knobs from the checkpoint config so train and eval match the saved run |
| `tools/data_alignment/` | Dataset readers, camera alignment, Wuji / MANO retargeting, Spark HDF5 and LeRobot writers |
| `tools/*.py` (others) | Retarget / LeRobot / repack command-line entry points |
| `source_starvla/` | Vendored StarVLA plus the XPolicyLab data registry |
| `../env_cfg/` | Tianji Marvin + Wuji robot, camera, and scene configs |

## Citation

```bibtex
@article{chen2026spark0,
  title={Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation},
  author={Chen, Tianxing and Lin, Wenwei and Lyu, Chuqiao and Xiang, Peicheng and Tang, Junyuan and Nian, Tian and Cai, Zijian and Ye, Kehe and Yu, Chenze and Zhu, Wenxuan and Chen, Eric J. and Qin, Yan and Chen, Yue and Wu, Tongyu and Gong, Junhao and Lei, Kai-Chong and Zhang, Jiahao and Su, Kailun and Liang, Qiwei and Xu, Renjing and Luo, Ping and Ding, Wenbo},
  journal={arXiv preprint arXiv:2610.22342},
  year={2026},
  url={https://arxiv.org/abs/2610.22342}
}
```

Built on [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), [StarVLA](https://github.com/starVLA/starVLA), and [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo). Contact: Tianxing Chen (chentianxing2002@gmail.com).
