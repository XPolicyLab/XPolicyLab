# Xiaomi_Robotics_1

**Contributor:** Xiaomi Corporation | **Paper:** Not released | **arXiv:** Not released | **Original code:** See vendored `xiaomi_robotics_1/`.

`Xiaomi_Robotics_1` is the adapter for Xiaomi's MiBot model: it drives a trained checkpoint in-process through a Qwen3-VL-4B-Instruct processor and converts the model's relative (delta) action chunks into absolute RoboDojo end-effector actions. It reproduces the official `mibot/server/deploy.py` + `runtime/server.py` + `runtime/client.py` pipeline without the socket hop. Integration scripts live at this directory level; the vendored upstream implementation lives in `xiaomi_robotics_1/xr1/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash install.sh
conda activate <policy_env>  # e.g. mibot (override the name via MIBOT_CONDA_ENV)
```

The installer creates the `mibot` conda environment with PyTorch 2.8, Flash Attention, and the other core dependencies. Read `INSTALLATION.md` for the manual install equivalent, checkpoint preparation, and smoke checks.

## Data Processing

`process_data.sh` converts RoboDojo HDF5 episodes into the xr1 JSON/MP4 training format and emits a ready-to-use Hydra data config with recomputed normalization statistics:

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash process_data.sh <bench_name> <task_names|all> <env_cfg_type> <action_type> [expert_data_num]

# Example: convert every arx_x5 simulation task
bash process_data.sh RoboDojo all arx_x5 ee

# Example: a 50-episode-per-task subset of two tasks
bash process_data.sh RoboDojo stack_bowls,load_washer arx_x5 ee 50
```

`bench_name` is `RoboDojo` (simulation, `arx_x5` only) or `RoboDojo_real` (real robots: `piper_x`, `piper`, `arx_x5`; each task belongs to exactly one of them, and only tasks carrying the requested robot are converted). Here `ckpt_name` doubles as the task filter — a comma-separated task list or `all` — and it must be repeated verbatim in `train.sh` so both steps agree on the dataset. `expert_data_num` caps the episodes taken from each task.

Outputs, all anchored on the script's own location:

| Artifact | Path |
| --- | --- |
| Episode JSON | `xiaomi_robotics_1/xr1/data/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>/data/chunk-XXXX/episode_*.json` |
| Videos | the sibling `videos/` directory, three MP4s per episode (ego / wrist-left / wrist-right) |
| Data config | `xiaomi_robotics_1/xr1/configs/data/<bench_lowercased>_<ckpt_name>_<env_cfg_type>.yaml` |

The generated config carries the `paths`, `mean`/`std` of the packed relative actions `(action_length, 60)`, and `q01`/`q99` of the packed state `(1, 60)`. Video paths inside the JSON are absolute, so training does not depend on the launch directory; moving the dataset means reconverting or rewriting those JSONs. Reruns skip episodes already converted — pass `EXTRA_ARGS=--overwrite` to force, or `EXTRA_ARGS=--stats-only` to recompute the statistics and the config alone. Optional overrides: `RAW_DATA_ROOT`, `OUTPUT_DIR`, `DATA_WORKERS`, `ACTION_LENGTH` (30), `BATCH_SIZE` (16).

Skip this step if you are evaluating a checkpoint you already have. `docs/data_format.md` in the vendored `xr1/` documents the JSON schema if you want to plug in your own converter.

## Model Assets

Download the inference checkpoint from the [RoboDojo official dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo); only the `ckpt/RoboDojo/Xiaomi_Robotics_1/` folder is needed:

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
mkdir -p checkpoints

hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "ckpt/RoboDojo/Xiaomi_Robotics_1/*" \
  --local-dir checkpoints/Xiaomi_Robotics_1 \
  --local-dir-use-symlinks False
```

At evaluation time the checkpoint is resolved from the `model_dir` field in `deploy.yml` (absolute path) or, when unset, from `checkpoints/<ckpt_name>/`.

Training instead starts from the released [Xiaomi-Robotics-1-5B](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) weights, converted once into the single-file format the training entrypoint loads:

```bash
hf download XiaomiRobotics/Xiaomi-Robotics-1-5B --local-dir hf_pretrain

python -u xiaomi_robotics_1/xr1/tools/weight_convert.py \
  --model_path hf_pretrain \
  --output_dir checkpoints/pretrained_ckpt \
  --output_filename model_states.pt
```

## Training

`train.sh` fine-tunes the released Xiaomi-Robotics-1-5B weights on a dataset produced by `process_data.sh`. It wraps the vendored launcher (`xiaomi_robotics_1/xr1/scripts/train.sh`, DeepSpeed via Lightning) and takes care of the XPolicyLab naming so evaluation finds the result without extra arguments.

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [hydra_overrides...]

# Example: 4-GPU run over every arx_x5 simulation task
bash train.sh RoboDojo all arx_x5 ee 0 0,1,2,3

# Example: raise the step budget and the gradient accumulation
MAX_STEPS=60000 bash train.sh RoboDojo all arx_x5 ee 0 0,1,2,3 \
  trainer.accumulate_grad_batches=2
```

The first six arguments follow the shared convention; anything after them is forwarded verbatim to Hydra. `bench_name`, `ckpt_name`, `env_cfg_type`, and `action_type` must match the `process_data.sh` call — they resolve the converted dataset and the generated data config. `action_type` must be `ee` (`ALLOW_NON_EE_ACTION=1` overrides the check; see Notes).

The converted pretrained weights from [Model Assets](#model-assets) must exist at `checkpoints/pretrained_ckpt/model_states.pt`, or `PRETRAINED_PATH` must point at them.

Training writes to `train_runs/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/project_xr1/<same_name>/`, holding `config.py` and `last.ckpt/` — exactly the layout `model.py` loads. `train.sh` symlinks `checkpoints/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/` to it, so that directory name is what you pass to `eval.sh` as `ckpt_name`. An existing non-symlink at that path aborts the run rather than being replaced.

Multi-GPU is driven by `gpu_id`; multi-node reads `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, and `MASTER_PORT` (run the same command on every node). The effective batch is `batch_size` from the data config × GPUs × `trainer.accumulate_grad_batches`, so lower `BATCH_SIZE` at conversion time (or override `data.params.train_datasets.batch_size`) if a step does not fit in memory. `MAX_LENGTH` (default 20000) bounds the token budget of one sample; samples exceeding it are dropped by the collate.

Logging goes through W&B. `train.sh` defaults to `WANDB_MODE=offline` because the training entrypoint always constructs a `WandbLogger` and an unauthenticated online run would block on the login prompt. Run `wandb login` and pass `WANDB_MODE=online` to upload.

| Variable | Notes |
| --- | --- |
| `PRETRAINED_PATH` | Converted release weights; default `checkpoints/pretrained_ckpt/model_states.pt`. |
| `OUTPUT_DIR`, `DATA_CONFIG_NAME` | Converted dataset root and data config basename; must match `process_data.sh`. |
| `RUN_ROOT`, `PROJECT` | Training output root (default `train_runs/<ckpt_setting>`) and project segment of the artifact path (default `xr1`). |
| `MAX_STEPS`, `SAVE_INTERVAL` | Step budget (30000) and checkpoint interval (10000). |
| `ASYNC_TRAIN` | Random action-prefix conditioning for asynchronous inference; default `true`. |
| `MAX_LENGTH` | Per-sample token budget of the collate; default `20000`. |
| `WANDB_MODE` | `offline` by default; `online` uploads after `wandb login`. |
| `ALLOW_NON_EE_ACTION` | Set to `1` to train with `action_type != ee` anyway. |

## Evaluation

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the downloaded official checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls Xiaomi_Robotics_1 arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the run trained above; ckpt_name is the directory train.sh linked
bash eval.sh RoboDojo stack_bowls RoboDojo-all-arx_x5-ee-0 arx_x5 ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## WUJI 58-D hybrid (tianji_wuji)

The adapter also drives the WUJI dual-arm dexterous-hand platform (`env_cfg_type: tianji_wuji`, two 7-DOF arms each carrying a 20-DOF Wuji hand, registered as `arm_dim [7, 7]` / `ee_dim [20, 20]`). The run is selected automatically when the robot registry reports two 20-DOF hands and `action_type` is `ee`; nothing changes for `arx_x5`.

In this mode the 60-slot state and action space carries the WUJI SimBench `wrist58` vector from `XPolicyLab.utils.wuji58` in slots `[0:58]`, with slots `[58:60]` zero and masked. State and action share the layout:

| Slots | Content | State | Action |
| --- | --- | --- | --- |
| `[0:20]` | left hand joints (finger-major) | absolute | absolute target |
| `[20:29]` | left wrist `xyz + rot6d` | absolute, in the wrist frame | delta vs. the wrist observed at the chunk start |
| `[29:49]` | right hand joints | absolute | absolute target |
| `[49:58]` | right wrist `xyz + rot6d` | absolute, in the wrist frame | delta vs. the wrist observed at the chunk start |

Wrist deltas are spatial deltas in the wrist frame (`dp = p_target - p_current`, `dR = R_target @ R_currentᵀ`, rot6d = first two rows), the WUJI SimBench `wrist_rel_te2` convention; the MiBot end-effector reframe does not apply. The wrist frame is `te2` (the WUJI SimBench head camera) by default or `world`, set by `wrist_frame` in `deploy.yml` and by `WRIST_FRAME` at conversion time — the two must match. The adapter restores absolute `left_ee_pose` / `right_ee_pose` (7-D, wxyz) and the 20-D `*_ee_joint_state` hand targets through the shared `decode_action_chunk58` with `reference="chunk_start"`: every chunk element is decoded against the observation the chunk was predicted from, because `chunk_actions` formed the training deltas that way (XR-1's native convention). This differs from the per-frame deltas the official converters write, so a WUJI SimBench or converter-made dataset is not interchangeable with this adapter's JSON export without re-running `process_data.sh`.

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
# conversion: the layout is auto-detected from the 20-wide hand joint state
bash process_data.sh RoboDojo all tianji_wuji ee
# training: lifts freq_excluded_dims (slots 17-19 are left-wrist dims here)
bash train.sh RoboDojo all tianji_wuji ee 0 0,1,2,3
# evaluation
bash eval.sh RoboDojo <task_name> RoboDojo-all-tianji_wuji-ee-0 tianji_wuji ee 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

Conversion reads the standard trajectory keys `state/{left,right}_ee_poses` (T, 7), `state/{left,right}_ee_joint_states` (T, 20) and the matching `action/` groups (simulation data without recorded `action/*_ee_poses` emulates them by the usual next-state shift) and writes `proprios.wuji58_state` / `actions.wuji58_target` (T, 58) stamped with `"layout": "wuji58"`; the loader forms the chunk-relative wrist deltas at sample time (`wuji58_layout.training_sample`). Statistics follow the arx_x5 rules — wrist deltas: per-(step, dim) gaussian over moving frames; hand targets: per-(step, dim) gaussian over every frame; state: 1st/99th percentiles — with the two padding slots written as identity / `0/0`. The robot must be registered in both `XPolicyLab/utils/robot/_robot_info.json` (read by `train.sh`) and the parent `env_cfg/robot/_robot_info.json` (read at runtime). `tests/test_wuji58_layout.py` pins the packing, the chunk decode round-trip and the statistics shapes on CPU.

## Configuration

`deploy.yml` keys to check before evaluation:

| Key | Meaning |
| --- | --- |
| `action_type` | Must be `ee`. `joint` is rejected at startup — see Notes. |
| `model_dir` | Checkpoint dir holding `config.py` and `last.ckpt/`. Overrides `ckpt_name`. |
| `ckpt_name` | Used when `model_dir` is unset: resolves to `checkpoints/<ckpt_name>/`. Nested layouts are searched for `config.py`. |
| `action_length` | Leading steps of each chunk to execute; `0` executes the whole chunk, whose length comes from the checkpoint. |
| `image_factor`, `image_max_pixels` | Image preprocessing, matching `mibot.utils.io.resize_image`. Defaults `32` / `160000`. |
| `vlm_processor_path` | HuggingFace repo id or local path; default `Qwen/Qwen3-VL-4B-Instruct`. |
| `default_prompt` | Instruction used when the observation carries none. |
| `wrist_frame` | WUJI 58-D hybrid only: `te2` (default) or `world`; must match the `WRIST_FRAME` the dataset was converted with. Ignored for `arx_x5`. |

Script environment variables, all optional:

| Variable | Used by | Notes |
| --- | --- | --- |
| `MIBOT_CONDA_ENV` | `install.sh` | Conda env name; default `mibot`. |
| `RAW_DATA_ROOT` | `process_data.sh` | HDF5 root holding `<task>/<env_cfg_type>/data/*.hdf5`; default is the nearest `data/<bench_name>` above `xr1/`. |
| `OUTPUT_DIR` | `process_data.sh`, `train.sh` | Converted dataset root; default `xiaomi_robotics_1/xr1/data/<data_setting>`. |
| `DATA_CONFIG_NAME` | `train.sh` | Data config basename under `configs/data`; default is the name `process_data.sh` generates. |
| `DATA_WORKERS` | `process_data.sh` | Conversion worker processes; default `os.cpu_count() // 2`. |
| `ACTION_LENGTH`, `BATCH_SIZE` | `process_data.sh` | Written into the generated data config; defaults `30` / `16`. |
| `WRIST_FRAME` | `process_data.sh` | WUJI 58-D hybrid only: wrist frame, `te2` (default) or `world`. |
| `EXTRA_ARGS` | `process_data.sh` | Flags forwarded to `scripts/process_data.py`, e.g. `--overwrite`, `--stats-only`. |
| `PRETRAINED_PATH` | `train.sh` | Converted release weights; default `checkpoints/pretrained_ckpt/model_states.pt`. |
| `RUN_ROOT`, `PROJECT` | `train.sh` | Training output root and project segment of the artifact path. |
| `MAX_STEPS`, `SAVE_INTERVAL`, `ASYNC_TRAIN`, `MAX_LENGTH` | `train.sh` | Training knobs; see the Training section. |
| `WANDB_MODE` | `train.sh` | `offline` by default. |
| `ALLOW_NON_EE_ACTION` | `train.sh` | Set to `1` to train with `action_type != ee`. |

## Notes

- **Only `action_type=ee` is supported.** The packed 60-dim action carries end-effector slots only (`ACTION_PARTS` in `mibot/utils/io.py` has no arm-joint entry), so joint targets cannot be recovered from the model output. Passing `joint` raises at startup rather than emitting wrong actions.
- The input state is always joint-space. State and action use different 60-dim layouts, so they must not be confused:
  - state (`compose_state`): `[0:7]` left_arm_joint, `[7:8]` left_gripper, `[8:15]` right_arm_joint, `[15:16]` right_gripper.
  - action (`ACTION_PARTS`): `[0:3]` left_ee_pos, `[3:6]` left_ee_aa, `[6:7]` left_gripper, `[8:11]` right_ee_pos, `[11:14]` right_ee_aa, `[14:15]` right_gripper, `[16:17]` waist, `[17:20]` base_vel.
- Every action slot is a **relative delta** with respect to the observed pose, with translation and rotation expressed in the current end-effector frame. The adapter restores absolute targets and maps end-effector rotations between the MiBot and simulator frames.
- The inference prompt carries only the vision and instruction turns, exactly as `mibot/server/runtime/client.py` builds them. The training-time `Robot state: <state>` / `<a_i>…<score>` turns are deliberately omitted: at inference the VLM is called without `state_embeds`, so a `<state>` token would fail inside `Qwen3VLModel.forward`. Proprioception reaches the model through the `state` tensor and the DiT state projector instead.
