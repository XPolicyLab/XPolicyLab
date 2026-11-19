# FlexPi

**Contributor:** Ge Yan et al. (UW / AI2) | **Paper:** Flex-π: A Multi-Stream World-Action Model with Compute Flexibility | **arXiv:** [2608.10860](https://arxiv.org/abs/2608.10860) | **Original code:** https://github.com/geyan21/flex-pi

`FlexPi` adapts Flex-π — a 6B world-action model that co-denoises future RGB, 3D pointmaps, DINOv3 semantics and actions inside a Mixture-of-Transformers — to XPolicyLab. One checkpoint deploys across every observed/generated stream combination, selected at runtime; full joint generation takes about 1.0 s per 32-step chunk here (eager bf16, 10 denoising steps, RTX 5090 or A100). The adapter serves RoboDojo (`arx_x5`) and RoboDojo_real (`arx_x5`, `piper`, `piper_x`) with `joint` or `ee` actions. The upstream implementation is vendored at `FlexPi/` (`src/`, `configs/`, `experiments/robotwin/`, `scripts/`); `tools/` holds the offline replay and diagnostics, `tests/` the state-bridge tests.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/FlexPi
bash install.sh [env_name] [--train] [--trt]
conda activate flexpi
```

Follows flex-pi `docs/INSTALL.md` §1a against the vendored tree and adds an editable XPolicyLab install (required: the policy server imports `XPolicyLab.policy.FlexPi.model`). `--train` adds `cuda-nvcc` (DeepSpeed shells out to `nvcc --version` at import); `--trt` adds the TensorRT extra.

## Model Assets

`download_checkpoint.sh` fetches a checkpoint package into `checkpoints/<ckpt_name>/` and the three base-weight files evaluation reads — the Wan2.2 VAE and the umt5-xxl encoder and tokenizer, ~13 GB from ModelScope — into `FlexPi/checkpoints/`:

```bash
bash download_checkpoint.sh <hf_repo_id> [ckpt_name] [revision]
```

The script reads a package at the top of the repo. The released RoboDojo_real checkpoints sit one folder per robot instead; [RoboDojo_real checkpoints](#robodojo_real-checkpoints) gives their download commands.

- `DIFFSYNTH_MODEL_BASE_PATH` moves the base weights; the download script and the server both read it.
- The released RoboDojo_real configs set `model.skip_dit_load_from_pretrain: true`, so those three files are the whole base set. A checkpoint whose `config.yaml` keeps `model.skip_dit_load_from_pretrain: false` also loads the Wan2.2-TI2V-5B DiT and the ActionDiT init file at boot, before its own weights replace them. The config names the ActionDiT file relative to `FlexPi/`, so it must sit at `FlexPi/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` whatever `DIFFSYNTH_MODEL_BASE_PATH` says; `FlexPi/scripts/preprocess_action_dit_backbone.py` builds it (flex-pi `docs/INSTALL.md` §2.1).
- The server fetches DINOv3 (`timm/vit_base_patch16_dinov3.lvd1689m`) from the Hugging Face hub at boot; an offline host needs it cached.
- Serving takes about 16 GB of GPU memory with the text encoder on the CPU (`offload_text_encoder: true`, the default).

## Data Processing

Two paths, one per checkpoint layout.

**14-D, official export.** Uses the **official** XPolicyLab converter, `scripts/transform_lerobot_v21_format.py` ([official LeRobot conversion](../../README.md#official-lerobot-conversion)) — no custom converter. LeRobot v2.1; its keys are exactly what flex-pi consumes: `observation.state`, `action`, and `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`. The camera names match flex-pi's own dataset names verbatim, and 14 = `arm[6,6] + ee[1,1]`, so neither keys nor dimensions need remapping.

```bash
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [max_episode] [raw_task_pattern]

# Example: convert every arx_x5 task at the training resolution
bash process_data.sh RoboDojo robodojo_all arx_x5 joint 200 '*'
```

Conversion is pinned to 240×320 (`FLEXPI_CONVERT_RESOLUTION`), matching `configs/data/robodojo.yaml` `raw_shape`. The result is linked to `data/<dataset_id>/lerobot`. `dataset_stats.json` is written by flex-pi's `RobotVideoDataset` on the first training run, so there is no separate stats step. The official converter reads only `vision/<cam>/colors`, so this export carries no depth.

**34-D (EEF + joints).** The 34-D layout is `[L pos3, L rot6d6, R pos3, R rot6d6, L grip, R grip, L joint6, pad, R joint6, pad]` (rot6d = the first two rows of R; pads 26/33 always 0). The official converters read joint keys only; the 34-D dataset builders (LeRobot v2.1, images decoded only through `decode_image_bit`) are released with the training code.

Keys differ from the official converter only in the 34-D `observation.state` / `action` (plus depth streams in the sim set and a per-frame `embodiment_index` in the real one); the camera names are the official ones. RoboDojo_real checkpoints train on one of two variants of the real corpus: the raw-frame one (each robot's EE pose as recorded, six defective episodes dropped) or the relabeled one, which rewrites every EE rotation as `R @ P[robot]` — `arx_x5` Ry(+90°), `piper` I, `piper_x` Rz(+90°), the table in `state_bridge._RELABEL_ROT` — so the three robots share one tool frame (the YAM `grasp_site` axes); positions, grippers and joints are unchanged.

<!-- TODO(submitter): ship or link the episode drop list and the relabel script with the training release. -->

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [num_gpus]

# Example: 8-GPU run
bash train.sh RoboDojo robodojo_all arx_x5 joint 0 0,1,2,3,4,5,6,7
```

A thin wrapper over flex-pi's own hydra entry (`accelerate launch scripts/train.py`) for the 14-D official export, pinned to `task=robodojo_unified_flex_3cam_384_1e-4`. It overrides only dataset paths, `output_dir`, seed, and the action/state dims (read from `env_cfg_type` via `utils/get_action_dim.sh`, not hardcoded). Checkpoints land in `checkpoints/<bench>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`, which is what `eval.sh` resolves from `ckpt_name`. `train.sh` has not been run end to end. Checkpoints in the 34-D layout are not trained through `train.sh`.

The RoboDojo_real checkpoints are submitted eval-only: their 34-D trainer and configs are not part of this adapter.

<!-- TODO(submitter): the training release timeline for the 34-D trainer and configs. -->

The two RoboDojo configs live in the vendored tree (copies of their RoboTwin siblings):

| File | Difference from the RoboTwin version |
|---|---|
| `FlexPi/configs/data/robodojo.yaml` | dataset / stats / text-cache paths; no `shape_meta.depth` |
| `FlexPi/configs/task/robodojo_unified_flex_3cam_384_1e-4.yaml` | `override /data: robodojo`, `model.enable_pointmap: false` |

The composite layout is reused as-is (`tshape_robotwin_384x320_uniform`): it is a geometric slot layout and is embodiment-agnostic despite the name.

## Evaluation

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: RoboDojo sim
bash eval.sh RoboDojo stack_bowls RoboDojo-robodojo_all-arx_x5-joint-0 arx_x5 joint 0 0 0 flexpi robodojo
```

RoboDojo_real runs one policy server per robot, each loading its own robot's checkpoint folder: `bench_name RoboDojo_real`, `env_cfg_type` `arx_x5`, `piper` or `piper_x`. The real-robot client is not part of the open-source release; [RoboDojo_real checkpoints](#robodojo_real-checkpoints) gives the launch line, what the client is assumed to send, and a pre-flight.

XPolicyLab owns the rollout loop and replanning cadence, so the adapter bypasses the upstream `step()` / `pending_actions` queue and calls `_infer_action_chunk()` directly, truncating to `replan_steps`.

`ckpt_name` resolves as documented in the XPolicyLab README; additionally both flex-pi checkpoint shapes are accepted:

```
run dir     <ckpt_dir>/config.yaml + <ckpt_dir>/checkpoints/weights/step_*.pt
packaged    <ckpt_dir>/{config.yaml, dataset_stats.json, step_*.pt}
```

The latest `step_*.pt` is selected by version sort. `dataset_stats.json` is expected next to `config.yaml`; override either with `FLEXPI_CHECKPOINT_PATH` / `FLEXPI_DATASET_STATS_PATH`. At eval time upstream reloads the model architecture and processor from the nearest `config.yaml` above the checkpoint's **resolved** path, and the launcher prints which one (`[SERVER] trained config:`); with none it falls back to the task preset, which can drift from training. A symlinked `step_*.pt` therefore makes upstream read the `config.yaml` beside the link's target, so the launcher refuses one whose target does not resolve to `<ckpt_dir>/config.yaml`; hardlink or copy the weights instead.

Offline wiring check, no simulator and no weights:

```bash
EVAL_ENV_TYPE=debug FLEXPI_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 flexpi flexpi
EVAL_ENV_TYPE=debug FLEXPI_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo_real insert_tubes demo arx_x5 joint 0 0 0 flexpi flexpi
```

`tools/openloop_replay.sh` replays 34-D (or 32-D YAM) LeRobot episodes through `Model` exactly as the server drives it, and reports the input parity and the open-loop error against the recorded actions.

## Configuration

Every `deploy.yml` key is documented inline. The ones worth checking before a run: `replan_steps`, `num_inference_steps`, `action_horizon`, `obs_image_hw`, `use_depth`, `depth_input_scale`, `eef_frame`, `check_stats_coverage`, `action_dim_mask`, `hold`, `sim_task`, and the six `infer_joint_*` / `infer_present_*` stream flags (`null` = the checkpoint's own choice). Per-run overrides go in `FLEXPI_EXTRA_OVERRIDES` (space-separated `key=value`, appended after the launcher's own).

| | |
|---|---|
| `action_type` | `joint` or `ee`, set by `--action-type` on **both** server and client (RoboDojo's `robodojo.sh` defaults it to `ee`). `joint` executes the checkpoint's joint half as 14D targets. `ee` executes its EEF half as `{left,right}_ee_pose` targets `[x,y,z,qw,qx,qy,qz]` in the frame the environment reports, plus grippers; the environment turns them into joints with its IK. It needs the 32D layout below. RoboDojo sim holds an arm whose IK fails without saying so, so `deploy.py` counts IK failures per episode into `_timing.jsonl`. |
| Checkpoint layout | Native 14D (`joint` only), or flex-pi's YAM 32D layout (34-D configs patched as below) bridged by `state_bridge.py` (`state_bridge: auto`). The bridge permutes joints/grippers and fills the EEF proprio block from `{left,right}_ee_pose` in the frame `eef_frame` names; on the way out it gathers the 14 joint/gripper slots (`joint`), or sends the EEF block back through the same frame (`ee`). Grippers are clipped to [0, 1]. |
| `eef_frame` | The frame the checkpoint's EEF block was trained in. `world_link6` passes the observed pose through: RoboDojo sim checkpoints in sim, raw-frame RoboDojo_real checkpoints on the real robots. `relabeled` applies the per-robot tool-frame table: relabeled-corpus RoboDojo_real checkpoints. `arm_base_grasp` is the real-robot YAM lineage. Getting it wrong fails silently, so it must match the checkpoint. |
| `env_cfg_type` | Any 3-camera robot with `arm_dim [6,6]`, `ee_dim [1,1]` (14-D), read through `get_robot_action_dim_info`: `arx_x5` in sim, `arx_x5` / `piper` / `piper_x` on RoboDojo_real. `eef_frame: relabeled` has a constant for those three only and refuses any other at init. |
| Cameras | Exactly three: `cam_head`, `cam_left_wrist`, `cam_right_wrist`. The vendored per-cam pipeline is fixed at 3 cameras; a missing one raises at the adapter boundary. |
| Batched eval | `get_action_batch` is implemented and `deploy.yml` ships `eval_batch: true`; batching runs one forward per env. |

| Variable | Notes |
|---|---|
| `FLEXPI_EXTRA_OVERRIDES` | Extra `deploy.yml` overrides for one run, e.g. `"eef_frame=relabeled num_inference_steps=4"`. |
| `FLEXPI_CKPT_ROOT` / `FLEXPI_CKPT_SETTING` | Override the checkpoint directory search. |
| `FLEXPI_CHECKPOINT_PATH` | Point directly at a `.pt`, skipping resolution. |
| `FLEXPI_DATASET_STATS_PATH` | Override `dataset_stats.json`. |
| `FLEXPI_ALLOW_DUMMY_POLICY` | `true` → skip checkpoint load, return zero actions (debug wiring only). Under `joint` on a robot the zeros are targets, not a hold; `hold: true` is the no-motion pass. |
| `FLEXPI_DATASET_ID` | Reuse a differently named converted dataset. |
| `FLEXPI_TASK_CONFIG` | Override the training task preset. |
| `FLEXPI_CONVERT_RESOLUTION` | Conversion resolution, default `240x320`. |
| `DIFFSYNTH_MODEL_BASE_PATH` | Base-weights directory; defaults to `FlexPi/checkpoints`. |
| `FLEXPI_DUMP_OBS` / `FLEXPI_DUMP_ACTIONS` / `FLEXPI_DUMP_STREAMS` | Debug dumps of the model inputs, the action chunks and the predicted streams, one directory each. |
| `FLEXPI_REPLAY_GT` / `FLEXPI_REPLAY_GT_ROOT` | Serve recorded 34-D dataset actions instead of the model (with `allow_dummy_policy`), to check the controller path; the root is required. |

### Depth / pointmap stream

`deploy.yml` ships `use_depth: false` with `infer_present_pointmap: false` / `infer_joint_pointmap: false`: RoboDojo observations carry no depth by default, and neither does RoboDojo_real.

1. **RoboDojo does not emit depth or intrinsics by default.** `env_cfg/arx_x5.yml` sets `observation.vision.{depth, approximate_depth, intrinsic_matrix, extrinsic_matrix}` all to `false`. At least one depth key plus `intrinsic_matrix` must be enabled before the adapter can see them. RoboDojo can emit `depth` (float32 metres) and `approximate_depth` (uint16 mm, pre-scaled); the adapter accepts either and picks the scale itself.
2. **The official LeRobot export carries no depth**, so a checkpoint trained through `process_data.sh` has no pointmap stream to feed. The 34-D sim dataset with depth does; its builder is released with the training code.

To use the stream: turn depth and `intrinsic_matrix` on in the RoboDojo env_cfg, set `use_depth: true`, and set both pointmap flags to `null` (the checkpoint's choice) or `true`.

**`depth_input_scale` matters and fails silently if wrong.** The upstream pointmap builder assumes uint16 **millimetres** and casts with `.to(torch.int32).clamp_(0, 65535).to(torch.uint16)`; its encoder then divides by 1000. RoboDojo's `depth` key is float32 **metres**, so passing it through unscaled floors every pixel to 0 — no exception, no warning, just a zero depth map and a dead pointmap stream. The default `1000.0` handles that; `approximate_depth` is pinned to `1.0`. Depth and `K` stay on one grid: with `obs_train_parity` both are resampled together onto the processor's depth shape, otherwise both are passed at native resolution.

## RoboDojo_real checkpoints

Three checkpoints, one per robot, each fine-tuned from [`flex-pi/flexpi-pretrain-preview`](https://huggingface.co/flex-pi/flexpi-pretrain-preview) on its robot's six RoboDojo_real tasks. The model reads the three RGB cameras and the joint, gripper and end-effector state of both arms, and returns 32-step chunks of absolute joint and gripper targets. It reads no depth (pointmap stream off), uses 34-D relative actions and frame-pooled q01/q99 normalization. `deploy.yml` ships the settings these checkpoints were checked with, so the plain launch line below serves them correctly.

### Download

Hugging Face [`flex-pi/flexpi-robodojo-real`](https://huggingface.co/flex-pi/flexpi-robodojo-real) (private; read access is granted per account), one folder per robot, each a complete package in the packaged layout above:

| folder (`env_cfg_type`) | weights | sha256 of the weights | sha256 of `dataset_stats.json` | trained on |
|---|---|---|---|---|
| `arx_x5/` | `step_103840.pt` | `76d550d0e2c6dca77943f2d463dfb7a0fde8f31be44a31b344fb6e9a165e6d78` | `b485348493742015e3df25abcf9e93e41d956c045bad10e8b94e0fc4e8cb1647` | 599 episodes, 664,412 frames |
| `piper/` | `step_083600.pt` | `ede12798d845cbbf526c4ed8af344bf3e08fc2b92f0665e35e0f062d75a9119d` | `7a769693984869eafc911e331b93cd8663036c07da38e3eef4eac81fd12a3398` | 595 episodes, 534,792 frames |
| `piper_x/` | `step_063520.pt` | `ac986f0aa590f31a596f37e6cc98184bc9667b37da522df27a52d38330ca95d0` | `e82bedf0de999b025b2b9fe188af8a97b2364fb9442bde13d57f67630a9367c5` | 600 episodes, 406,433 frames |

```bash
cd XPolicyLab/policy/FlexPi
conda activate flexpi
export HF_TOKEN=<token with read access>
for robot in arx_x5 piper piper_x; do
  huggingface-cli download flex-pi/flexpi-robodojo-real --include "$robot/*" --local-dir checkpoints/flexpi-robodojo-real
done
sha256sum checkpoints/flexpi-robodojo-real/*/step_*.pt checkpoints/flexpi-robodojo-real/*/dataset_stats.json
```

- `step_N.pt` (bf16, 12,082,213,237 B) holds every fine-tuned weight.
- `config.yaml` is the same file in all three folders (sha256 `e31d5453153db4a5fecf588cddf4b50384925b9949ec32370925916147875142`): the run's config, cleaned of training-only entries, through `tools/patch_34d_native_config.py`, which writes `raw_shape: [32, 34]` into the action/state shape_meta and prepends `AlignTo34DByWidth` (unpatched, `Model` init raises "No state bridge for checkpoint state dim 34 and robot dim 14"), plus `skip_dit_load_from_pretrain: true`.
- `dataset_stats.json` is per robot; the three files differ. Its `coverage` names the corpus and the robot, which the server checks at boot.
- `--local-dir` writes regular files. The Hugging Face cache layout (symlinked files) makes upstream read the `config.yaml` beside the link target, which the launcher refuses (see the checkpoint layouts above).

The base files (Wan2.2 VAE, umt5-xxl encoder and tokenizer, 12.8 GB from ModelScope) and DINOv3 (343 MB from the Hugging Face hub), once:

```bash
cd XPolicyLab/policy/FlexPi
conda activate flexpi
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/FlexPi/checkpoints"
python - <<'EOF'
from flexpi.models.helpers.io import ModelConfig
for repo, pattern in [
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/"),
]:
    ModelConfig(model_id=repo, origin_file_pattern=pattern).download_if_necessary()
EOF
python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)"
```

| file under `$DIFFSYNTH_MODEL_BASE_PATH` | bytes | sha256 |
|---|---|---|
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors` | 1,409,401,152 | `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996` |
| `DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors` | 11,361,845,432 | `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` |
| `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json` | 6,623 | `7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d` |
| `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model` | 4,548,313 | `e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458` |
| `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json` | 16,837,417 | `6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b` |
| `Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json` | 61,728 | `ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05` |

A base file missing at boot is fetched from ModelScope without a prompt; `DIFFSYNTH_SKIP_DOWNLOAD=true` makes it an error instead. The converted VAE and T5 files are not on Hugging Face, so a host without ModelScope access needs them copied in. With the DINOv3 cache filled, the server also starts with `HF_HUB_OFFLINE=1`.

### env_cfg

The server reads `<benchmark root>/env_cfg/<env_cfg_type>.yml` and the robot its `config.robot` names in `env_cfg/robot/_robot_info.json`, which must give `arm_dim [6, 6]` and `ee_dim [1, 1]`; nothing else in `env_cfg` is read. Open-source RoboDojo ships only `arx_x5.yml` (robot `dual_x5`, the same dims), so `piper` and `piper_x` take the station's own files; a missing one stops the server at start with `FileNotFoundError`.

### Launch

One policy server per robot, started from a shell at the conda base environment:

```bash
cd XPolicyLab/policy/FlexPi
bash setup_eval_policy_server.sh RoboDojo_real <task_label> checkpoints/flexpi-robodojo-real/<robot> \
    <robot> joint 0 <gpu_id> flexpi <port> 0.0.0.0 2>&1 | tee flexpi_server_<robot>.log
```

- `task_label` and the seed (`0`) only label the run: one server serves all six tasks of its robot, the instruction in each observation names the task, and the diffusion noise is fresh on every call.
- `env_cfg_type` (`<robot>`) selects the robot's tool-frame constant and must be the folder's robot; the server refuses another one at boot (below). `action_type` is `joint`, the same as the client's.
- `0.0.0.0` accepts a client on another machine; the default is `localhost`. The port opens once the model is loaded, 1-3 minutes after start.
- To pin every setting that decides the served model whatever `deploy.yml` says, export this before the launch. It is what `deploy.yml` ships, and a launch with it returns the same chunks bit for bit as one without:

  ```bash
  export FLEXPI_EXTRA_OVERRIDES="eef_frame=relabeled use_depth=false infer_present_pointmap=false \
  infer_joint_pointmap=false num_inference_steps=10 replan_steps=32 action_dim_mask=false obs_image_hw=null \
  use_per_cam=true obs_train_parity=true state_bridge=auto diffusion_seed=null sigma_shift=null \
  offload_text_encoder=true log_action_chunk=true torch_compile=false compile_encoders=false glue_cache=false \
  encoder_cuda_graph=false joint_loop_cuda_graph=false quantization=null attn_backend=sdpa solver=euler"
  ```

- Every other `FLEXPI_*` variable stays unset: `FLEXPI_CHECKPOINT_PATH`, `FLEXPI_DATASET_STATS_PATH` and `FLEXPI_CKPT_SETTING` / `FLEXPI_CKPT_ROOT` replace the folder argument, `FLEXPI_ALLOW_DUMMY_POLICY` serves zeros and `FLEXPI_REPLAY_GT` recorded actions. `FLEXPI_DUMP_OBS=<dir>`, for the pre-flight start only, writes one `.npz` per `get_action`.
- The launcher activates the conda environment itself, under `set -u`. When `flexpi` is already active in the calling shell, conda first runs its deactivate hooks, and one that reads an unset variable can stop the launcher right after its `[SERVER]` lines.

The station client reads `XPolicyLab/policy/FlexPi/deploy.yml` and runs FlexPi's episode loop, `deploy.py`, so the client host needs `policy/FlexPi` too (without `checkpoints/`; the loop needs only numpy). The loop resets the policy at each episode start, runs all 32 steps of a chunk in order, sends an observation after every step, and stops the client with exit status 2 if its `action_type` differs from the server's:

```bash
cd XPolicyLab/policy/FlexPi
EVAL_ENV_TYPE=real bash setup_eval_env_client.sh RoboDojo_real <task_name> flexpi-robodojo-real/<robot> <robot> joint 0 \
    <env_gpu_id> <eval_env_conda_env> "action_type=joint,base_cfg=<station base_cfg>" <port> <policy_server_ip>
```

### Observation, every control tick

| key | shape | content |
|---|---|---|
| `state.left_arm_joint_state`, `state.right_arm_joint_state` | (6,) | joint angles in rad, in the recorded order and sign |
| `state.left_ee_joint_state`, `state.right_ee_joint_state` | (1,) | gripper reading on the recorded scale: about 0.9 open, about 0 closed; Piper and Piper X readings reach about -0.04 when closed |
| `state.left_ee_pose`, `state.right_ee_pose` | (7,) | `[x, y, z, qw, qx, qy, qz]`, metres and a scalar-first quaternion, each arm in its own recorded frame (below). Needed under `joint` too: the model state includes it |
| `vision.cam_head.color`, `vision.cam_left_wrist.color`, `vision.cam_right_wrist.color` | (480, 640, 3) uint8 RGB, or an `encode_image_bit` JPEG | all three cameras |
| `instruction` | str | the task's training sentence ([Instructions](#instructions)) |

- EE pose: the link6 (flange) point of each arm in that arm's own frame, as `state/*_ee_poses` of the RoboDojo_real HDF5 stores it; there is no shared world frame. `arx_x5`: the origin is the arm's link6 position at zero joints, axes parallel to the arm base (x forward, y left, z up). `piper`, `piper_x`: the arm's base frame (x forward, y left, z up).
- Send the pose as recorded: the server applies its own constant rotation per robot ([The relabeled frame](#the-relabeled-frame)), so a pose converted on the client reaches the model 90° or 180° off. The quaternion is normalized, so its sign does not matter.
- Before the model sees the state, every value except the rotation is clamped to the checkpoint's trained 1st-99th percentile range, so out-of-range readings and convention errors raise nothing; the home-pose reading of the pre-flight is where they show.
- Left and right are the robot's own: the left arm appears in the left half of the head image, and `cam_left_wrist` is the left arm's wrist camera.
- Images: raw arrays pass through unchanged, encoded frames are decoded with `decode_image_bit`. A non-uint8 array is cast to uint8, so a float image in [0, 1] arrives black; another size or aspect ratio is resized differently from training. Training used brightness, contrast and saturation jitter (0.3 / 0.4 / 0.5) and no hue jitter.
- Timing: the latest frames and state, with no added delay. In the recordings each image trails the state by about 2 frames, and the model was trained on that pairing.
- Instruction: the sentence alone; the server wraps it in the training template. Bytes are decoded. A missing instruction becomes `complete the task`; the server warns once per instruction that is not one of its robot's six sentences.

### Actions

- Each `get_action` returns 32 steps; step k is the target for control tick t+1+k, where t is the tick of the observation it was computed from. Execution is synchronous: all 32 steps in order, one per tick, then the next chunk is computed from the current observation while the robot holds its last target.
- `joint` returns `left_arm_joint_state`, `right_arm_joint_state` (6), absolute joint targets in rad (the measured joints plus the predicted change), and `left_ee_joint_state`, `right_ee_joint_state` (1), the predicted gripper reading, clipped to [0, 1].
- `ee` decodes the same network output into link6-flange poses in the recorded per-arm frame (`qw >= 0`); it has been checked offline only.
- Rate: one step per control tick at the recording rate, 30 Hz on `arx_x5` and `piper`, 25 Hz on `piper_x` (113 of its 600 files record 25 Hz, the rest no rate; its preview videos are 25 fps). The model has no rate input: at another rate it replays the learned motions faster or slower.

### The relabeled frame

`eef_frame: relabeled` re-expresses each observed `*_ee_pose` rotation in the corpus's shared tool frame, `R_model = R_obs @ P[env_cfg_type]`, for the proprio state and the relative-action anchor in both action types; in `ee` mode the returned pose is mapped back, `R_obs = R_model @ P^T`. Positions, joints and grippers are untouched. At the home pose the model frame is z forward (approach), y left (finger opening), x down on all three robots.

| robot | recorded tool axes at the home pose | P (`state_bridge._RELABEL_ROT`) |
|---|---|---|
| `arx_x5` | x forward (approach), y left, z up | Ry(+90°) `[[0,0,1],[0,1,0],[-1,0,0]]` |
| `piper` | x down, y left, z forward (approach) | identity |
| `piper_x` | x right, y down, z forward (approach) | Rz(+90°) `[[0,-1,0],[1,0,0],[0,0,1]]` |

XPolicyLab's `_EEF_REFRAME_P_ROBODOJO_REAL` (`policy/Xiaomi_Robotics_1`, `policy/Rex_M1_preview`) is this table times Rz(180°), so it turns every robot, `piper` included. This adapter's `P[piper]` is the identity, so a piper-only test cannot show a wrong frame here; `arx_x5` and `piper_x` do. GT replay under `action_type: ee` reads the replay root's frame from its `meta/info.json` and refuses a mismatched `eef_frame`.

### Pre-flight, per robot

1. Boot lines. A correct start logs these, with others in between:

   ```text
   [SERVER] checkpoint_path: .../checkpoints/flexpi-robodojo-real/<robot>/step_N.pt
   [SERVER] dataset_stats_path: .../checkpoints/flexpi-robodojo-real/<robot>/dataset_stats.json
   [FlexPi] stats coverage robodojo_realworld_34d_relabeled_<robot>: eef_frame=relabeled env_cfg_type=<robot> match
   [FlexPi] Loaded model + processor from trained config: .../<robot>/config.yaml
   [FlexPi] action_dim_mask: disabled (action_dim_mask=false)
   [FlexPi] flex infer overrides: joint_video=None joint_dino=None joint_pointmap=False present_video=None present_dino=None present_pointmap=False
   [FlexPi] ready: action_type=joint env_cfg_type=<robot> replan_steps=32 action_horizon=32 use_depth=False needs_depth=False state_bridge=yam32 eef_frame=relabeled diffusion_seed=None
   ```

   Neither `No config.yaml found next to checkpoint` nor `Shape mismatch` appears. Another robot's `env_cfg_type`, or an `eef_frame` other than `relabeled`, stops the server here with a `ValueError` naming the stats' `coverage`.

2. Home-pose reading, without motion. With all joints at 0 and both grippers open, the pose the recordings start from, one observation reads about:

   | robot | arm | `ee_pose` position (m) | `ee_pose` quaternion (w, x, y, z) | gripper |
   |---|---|---|---|---|
   | `arx_x5` | left | (0.000, 0.000, 0.001) | (1.000, -0.008, 0.000, -0.001) | 0.909 |
   | `arx_x5` | right | (0.000, 0.000, 0.001) | (1.000, 0.010, 0.001, 0.000) | 0.891 |
   | `piper` | left | (0.056, 0.000, 0.213) | (0.733, 0.000, 0.680, 0.000) | 0.927 |
   | `piper` | right | (0.057, 0.000, 0.212) | (0.731, -0.002, 0.682, 0.002) | 0.906 |
   | `piper_x` | left | (0.098, 0.000, 0.211) | (0.516, -0.483, 0.484, -0.516) | 0.912 |
   | `piper_x` | right | (0.098, 0.000, 0.212) | (0.516, -0.483, 0.482, -0.517) | 0.913 |

   A match is within about 5 mm and 0.03 per quaternion component, up to an overall sign. Typical mismatches: an `arx_x5` position near (0.095, 0.000, 0.158) is relative to the arm base instead of the recorded origin, and still inside the trained range; an `arx_x5` quaternion other than about (1, 0, 0, 0) means a rotation applied before sending or reordered components; `[qx, qy, qz, qw]` order is only about 5° off at the `piper_x` home, so the numbers show it and the angle does not; an open gripper not near 0.9 is another gripper scale. Without a client-side print, the first dump file of step 3 holds the same values as the model receives them:

   ```bash
   python -c "import numpy as np; s = np.load('<dir>/obs_00000.npz')['state']; print(s[0:3], s[3:9], s[9:12], s[12:18], s[18:20])"
   ```

   This prints the left position, left rot6d, right position, right rot6d and the grippers; at home each rot6d reads about (0, 0, 1, 0, 1, 0) on `arx_x5` and (0.07, 0, 1.0, 0, 1, 0) on `piper` and `piper_x`, within about 0.05.

3. Dry run at the home pose, on a server started with `FLEXPI_DUMP_OBS=<dir>` and `FLEXPI_EXTRA_OVERRIDES="hold=true"`: the model runs and logs every call as usual, but each chunk sent is the state measured at the episode's first call, so the arms stay put.
   - `[FlexPi] Encoding prompt (episode N): "A video recorded from a robot's point of view executing the following instruction: <sentence>"` carries exactly the task's sentence, at the first call of each episode; no `WARNING: instruction` line follows.
   - The first call prints `[FlexPi bridge]` lines: each arm's `ee pos (relabeled)` at home as in the table, `state grip [18:20]` about 0.9 for both grippers, and `action[0] joints - state joints` a few hundredths of a rad or less.
   - Every call logs a `[FlexPi chunk ...]` summary of what the model asked for (largest joint change, gripper range).

4. Colour order, from the same dump:

   ```bash
   python -c "import numpy as np; from PIL import Image; a = np.load('<dir>/obs_00000.npz')['head_camera.rgb']; print(a.shape); Image.fromarray(a).save('head.png')"
   ```

   The frame is (480, 640, 3) and `head.png` shows the scene's colours (a red object that reads blue means swapped channels; the server cannot tell). The scored trials then run on a server started without `FLEXPI_DUMP_OBS` and `hold`.

### Safety and timing

- The server clips grippers to [0, 1] and applies no joint position or velocity limits; the station's limits and emergency stop act as for any policy. In replays of recorded episodes the joint targets stayed within about 6 mrad of the trained joint range, and per-tick target changes reached 0.14 rad against 0.13 rad in the recordings of the same windows.
- In 48 recorded clips per robot the model moved an arm the demonstration kept still in 5 arm-windows (3 on `piper_x`, 1 each on `piper` and `arx_x5`), by up to 1.5 rad within one chunk: watch the idle arm in the first trials.
- Per `get_action` (10 steps, bf16): about 1.0 s on an A100 or an RTX 5090, 1.4 s on an RTX 6000 Ada. The arms pause for it between chunks; RoboDojo's step cap counts control steps, so this changes wall time, not scores.
- The text encoder runs on the CPU. Each distinct instruction is encoded once per server and kept across resets: the first episode of each task pays it (25-27 s on 8 CPU threads of an A100 node, a few seconds on a workstation CPU), and XPolicyLab's websocket client allows 120 s per request by default.
- About 16 GB of GPU memory per server, with the 11 GB umt5-xxl encoder in host RAM; three servers on one host need about 3 x 16 GB.

### Instructions

The model receives `A video recorded from a robot's point of view executing the following instruction: <instruction>`, with these sentences exactly (`robodojo_real_tasks.py`; also the instructions on RoboDojo's real-task pages):

| robot | task | instruction |
|---|---|---|
| `arx_x5` | cover_blocks | Cover the three blocks from left to right, then uncover them again in the order of red, green, and blue. |
| `arx_x5` | insert_tubes | Pick up the test tubes on the table and insert them into the test tube rack. |
| `arx_x5` | make_bread | Pick up the two slices of bread from the bowl and place them into the toaster, then place the two small bowls on the plate. |
| `arx_x5` | make_food | Take the cutting board out and place it on the table. Then place the steak and the knife on the cutting board, put the vegetables and shrimp into the pot, place the pot on the stove, and finally cover it with the lid. |
| `arx_x5` | pack_and_pour_fruit | Place all the fruits into the blue bowl, then pour the fruits from the blue bowl into the large white bowl. |
| `arx_x5` | store_in_safe | Place all the objects into the safe, then close the safe. |
| `piper` | fill_pen_holder | Pick up the pen holder and place all the pens into it. |
| `piper` | insert_charger | Insert the charger plug into the power strip, then connect the charging cable to the plug. |
| `piper` | put_objects_into_basket | Place all the objects on the table into the basket. |
| `piper` | stack_and_cover_blocks | Stack the blocks on the table, then cover them with the cup. |
| `piper` | stack_bowls | Stack the bowls on the table. |
| `piper` | stand_up_bottles | Stand the bottle upright. |
| `piper_x` | cap_pen | Pick up the pen and the pen cap, then put the cap on the pen. |
| `piper_x` | classify_objects | Classify the two types of objects on the table and place them into the two baskets. |
| `piper_x` | disassemble_LEGO | Take apart the assembled LEGO structure and place all the pieces into the bowl. |
| `piper_x` | hang_mugs | Hang the mugs on the mug rack. |
| `piper_x` | pack_objects_into_backpack | Place all the objects on the table into the backpack. |
| `piper_x` | sweep_blocks | Pick up the broom, hand it over to the right hand, then use the dustpan to sweep the blocks. |

### Validation (offline, in-sample)

- Inputs: the adapter's state built from the raw recorded values equals the training state on every training frame; its image inputs are identical to the training inputs.
- Output decoding: the `joint` and `ee` paths return the recorded targets within 1.2e-7 rad and 3.2e-8 m.
- Serving: through the launcher and websocket, this adapter's chunks for the `arx_x5` and `piper_x` packages (`diffusion_seed=0`) are bit-identical to the in-process replay of the same checkpoints; for `arx_x5` also with the pinned override line, and from an environment built by `install.sh` alone.
- Replay: 24 recorded windows per robot (4 per task) on an A100 in bf16 at 10 steps, each 32-step chunk compared with the recorded future; "hold" repeats the current state.

| robot | joint MAE (rad) | hold | gripper MAE | hold | EE position (mm) | hold | EE rotation (°) | hold |
|---|---|---|---|---|---|---|---|---|
| `arx_x5` | 0.0068 | 0.0735 | 0.0020 | 0.0171 | 4.9 | 37.6 | 0.59 | 8.5 |
| `piper` | 0.0066 | 0.0790 | 0.0034 | 0.0510 | 4.1 | 36.5 | 0.59 | 9.1 |
| `piper_x` | 0.0156 | 0.0849 | 0.0066 | 0.0689 | 7.7 | 45.7 | 1.59 | 10.2 |

Every recording was used for training, so these windows are in-sample; behaviour on the real robots is untested.

## Notes

- **Vendored edits.** `model.py` translates the XPolicyLab observation/action contract onto the dict shape the vendored RoboTwin deploy entry point reads. Besides trailing-whitespace cleanup, upstream code carries two edits in `experiments/robotwin/flexpi_policy/deploy_policy.py`, both no-ops for the stock RoboTwin path — re-apply them on a re-vendor:
  1. `_denormalize_action` takes an optional `anchor_state`, which `_infer_action_chunk` passes, so relative-action checkpoints (`Yam32DRelativeAction`) decode against the real current state instead of a zero placeholder. Configs without `action_state_transforms` are unaffected.
  2. `_infer_action_chunk` builds depth and intrinsics only when `infer_present_pointmap` is not `false`, so a flex checkpoint can run with the pointmap stream dropped on an observation that carries no depth.
- **`deploy.py`** follows `policy/demo_policy/deploy.py` and adds three things around its loop, none of which touches the actions: per-batch wall-clock timing and IK-failure counts in `_timing.jsonl`, and a check that the server's and the client's `action_type` agree, which stops the client on a mismatch (it would file results under the wrong action type).
- **Verified.**
  - The debug loop above passes, plain and with `DEBUG_OBS_ENCODED=1`.
  - `tests/test_state_bridge.py` covers the state bridge. `tests/test_robodojo_real_frames.py` replays 12 episodes per robot from the raw RoboDojo_real values against both corpora: state exact; `ee` output within 2.9e-8 (position) / 2.1e-8 (quaternion); joints within 1.2e-7. It skips unless `ROBODOJO_REAL_LOWDIM_ROOT`, `ROBODOJO_REAL_RAW_ROOT` and `ROBODOJO_REAL_RELABELED_ROOT` point at the data.
  - `tests/test_model_guards.py` covers instruction decoding, the stats coverage check and the prompt cache.
  - Real RoboDojo checkpoints have been served through this adapter in RoboDojo simulation.
  - The RoboDojo_real checkpoints, through the launcher and websocket: [Validation](#validation-offline-in-sample). The boot refusals (another robot's `env_cfg_type`, `eef_frame: world_link6`), the cached prompt encode (bit-identical to a fresh one) and `hold: true` (every step sent equals the measured state) were checked the same way.
  - A YAM real-robot checkpoint served through this adapter's websocket server reproduces its in-process `Model` actions exactly, and matches the YAM rig's own deploy stack within sampling noise on recorded episodes.
- **Known limitations.**
  - `action_type=ee` needs a 32D-layout checkpoint (a native-14D one has no EEF block and is rejected at init). Near the X5 wrist singularity (q5 = ±π/2), RoboDojo sim's IK switches branch or 2π-wraps a joint and the arm swings; read the `ik` counts in `_timing.jsonl` next to any `ee` success rate.
  - The RoboDojo_real observation contract is unpublished. The settings above assume the live observation matches the recorded HDF5, including each robot's EE convention.
  - The `enable_pointmap: false` × `flex_joint.p_*_pointmap: 0.5` combination in the 14-D task preset has not been trained here. It should be a no-op (no pointmap stream exists to drop).
  - The `robodojo_*` configs live only in the vendored copy; a re-vendor from flex-pi must keep them.
- **Vendored-tree audit.** The remaining `Image.open` / `np.frombuffer` hits are vendored LeRobot file loading and flex-pi's own FFV1 depth codec. None of them read XPolicyLab trajectories, and none are on the RoboDojo eval path.
