# LingBot_VA

**Contributor:** RoboDojo Team | **Paper:** LingBot-VA technical report | **arXiv:** TBD | **Original code:** See vendored `lingbot_va/`.

`LingBot_VA` is the XPolicyLab/RoboDojo adapter for the corresponding policy. It keeps integration-facing scripts at this directory level and leaves the original or vendored implementation in the nested source tree when present.

<details>
<summary>File Structure</summary>

| Path | Purpose |
|---|---|
| `README.md` | Supplemental documentation or environment metadata. |
| `install.sh` | Installs the policy-side runtime and editable dependencies. |
| `process_data.sh` | Runs the full latent pipeline (30-dim actions, `action_config`, Wan2.2 VAE latents, `empty_emb.pt`). |
| `process_data.py` | Latent-extraction worker invoked by `process_data.sh`. |
| `train.sh` | Launches the XPolicyLab training wrapper for this policy. |
| `eval.sh` | Runs a same-machine policy server plus RoboDojo environment client evaluation. |
| `setup_eval_policy_server.sh` | Starts only the policy server for distributed/debug evaluation. |
| `setup_eval_env_client.sh` | Starts only the RoboDojo environment client and connects to a policy server. |
| `deploy.py` | Policy wrapper used by the XPolicyLab model server. |
| `model.py` | Model adapter loaded by `deploy.py` or the policy server. |
| `deploy.yml` | Runtime configuration and default checkpoint/model parameters. |
| `lingbot_va/` | Vendored upstream code, policy-specific assets, or helper scripts. |
| `visualization/` | Vendored upstream code, policy-specific assets, or helper scripts. |

</details>

## Installation

What it does: installs or activates the policy-side runtime so the XPolicyLab server can import the adapter and upstream model code.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `policy_env` | Name of the conda environment used by the policy runtime. |

```bash
cd XPolicyLab/policy/LingBot_VA
# Example: install dependencies for the LingBot_VA policy adapter.
bash install.sh
# Example: activate the environment used later as <policy_conda_env>.
conda activate <policy_env>  # e.g. lingbot-va
```

## Demo Data Processing

What it does: LingBot-VA trains on **precomputed Wan2.2 VAE latents**, not on raw LeRobot parquet/video. `process_data.sh` takes a standard RoboDojo LeRobot v2.1 dataset (parquet + per-camera mp4) and runs every upstream step to produce a training-ready dataset: it maps actions into the 30-dim layout, adds an `action_config` segment to each `meta/episodes.jsonl` line, encodes Wan2.2 VAE video latents into a `latents/` tree, and writes `empty_emb.pt` at the dataset root. The base model (`LINGBOT_VA_BASE_MODEL_PATH`) supplies the VAE and text encoder, so it must be set before running.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Data/run identifier. Use a different value for ablations. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `expert_data_num` | Optional. Number of episodes to process; omit or `0` for all. |

Environment variables:

| Variable | Notes |
|---|---|
| `LEROBOT_DATA_ROOT` | Parent directory that holds LeRobot dataset folders. |
| `LEROBOT_DATASET_REPO_ID` | Source dataset folder name under `LEROBOT_DATA_ROOT`. |
| `LINGBOT_VA_SOURCE_DATASET` | Full path to the source LeRobot dataset; overrides the two variables above. |
| `LINGBOT_VA_DATASET_PATH` | Output path for the prepared latent dataset (defaults to `data/<tag>`). |
| `LINGBOT_VA_BASE_MODEL_PATH` | **Required:** `lingbot-va-base` weights (provides VAE + text encoder). |
| `LINGBOT_VA_PROCESS_GPU` | GPU id used for encoding (default `0`). |
| `LINGBOT_VA_TARGET_FPS` | Target sampling fps for latents (default `10`). |

```bash
cd XPolicyLab/policy/LingBot_VA
export LEROBOT_DATA_ROOT=<parent_dir_of_lerobot_datasets>
export LEROBOT_DATASET_REPO_ID=<source_dataset_folder_name>
export LINGBOT_VA_BASE_MODEL_PATH=<path_to_lingbot-va-base>
export LINGBOT_VA_DATASET_PATH=<output_dataset_path>

# Build the latent dataset (add a trailing episode count to process a subset).
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type>
```

The source dataset must be LeRobot **v2.1** (the trainer loader expects v2.1). Actions are mapped into the 30-dim layout below; dimensions absent from the source (e.g. end-effector poses in joint-only data) are zero-padded.

### Dataset layout

A prepared dataset mirrors `videos/` under `latents/` and adds `empty_emb.pt` at the root:

```
your_dataset/
├── videos/
│   └── chunk-000/observation.images.cam_high/episode_000000.mp4
├── latents/
│   └── chunk-000/observation.images.cam_high/episode_000000_0_450.pth
├── empty_emb.pt
└── meta/
    └── episodes.jsonl
```

- Latents are written for **each** camera in the config's `obs_cam_keys` (`cam_high`, `cam_left_wrist`, `cam_right_wrist` for RoboDojo). The trainer skips any segment whose latent files are missing for a camera.
- Latent files are named `episode_{index}_{start_frame}_{end_frame}.pth`, matching the `action_config` segments in `episodes.jsonl` (e.g. segment `0–450` → `episode_000000_0_450.pth`).
- `empty_emb.pt` is the Wan2.2 text embedding of an empty string, used when classifier-free guidance drops language conditioning. Generate it with the same text encoder used for latent extraction.

### `action_config` in `meta/episodes.jsonl`

Each episode line must carry an `action_config` list describing temporal segments and their language descriptions:

```json
{
  "episode_index": 0,
  "tasks": ["task description"],
  "length": 450,
  "action_config": [
    {"start_frame": 0, "end_frame": 450, "action_text": "Description of the action in this segment."}
  ]
}
```

`process_data.sh` writes one segment per episode (`start_frame=0`, `end_frame=length`, `action_text` = the episode task). For multiple sub-tasks per episode, use one entry (and one latent file per camera) per segment.

### Action and video specifications

- **Actions**: 30-dimensional, laid out as left/right arm EEF (7+7), left/right arm joints (7+7), left/right gripper (1+1). Map your robot's dimensions into this layout and pad missing dimensions with `0`.
- **Videos**: resize to ~256×256 pixels and downsample to 5–15 fps before VAE encoding (matches `height=256`, `width=256` in `va_robotwin30_train_cfg`).

Each latent `.pth` is a dict with the fields documented in `lingbot_va/README.md` § Custom Dataset Preparation (`latent`, `latent_num_frames/height/width`, `video_num_frames/height/width`, `text_emb`, `text`, `frame_ids`, `start_frame`, `end_frame`, `fps`, `ori_fps`).

## Model Training

What it does: starts FSDP post-training through the upstream trainer and writes checkpoints under this adapter directory. Requires a latent-prepared dataset from [Demo Data Processing](#demo-data-processing) and a base model linked at `.merged_ckpt`.

Parameters used by the command:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `ckpt_name` | Training run identifier, for example `cotrain`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Random seed. |
| `gpu_id` | GPU id or comma-separated GPU ids. The transformer is sharded with FSDP; a multi-GPU list is recommended, as a single GPU is typically insufficient to load and train it. |

Environment variables:

| Variable | Notes |
|---|---|
| `LINGBOT_VA_DATASET_PATH` | Path to the prepared latent dataset (or set `LEROBOT_DATA_ROOT` + `LEROBOT_DATASET_REPO_ID`). |
| `LINGBOT_VA_BASE_MODEL_PATH` | Path to the `lingbot-va-base` weights linked into `.merged_ckpt`. |

```bash
cd XPolicyLab/policy/LingBot_VA
conda activate <policy_env>

export LINGBOT_VA_DATASET_PATH=<path_to_prepared_dataset>
export LINGBOT_VA_BASE_MODEL_PATH=<path_to_lingbot-va-base>
ln -sfn "${LINGBOT_VA_BASE_MODEL_PATH}" .merged_ckpt

# Template: train on a GPU list (FSDP).
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on eight GPUs.
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7
```

Before training, ensure the base model's `transformer/config.json` has `"attn_mode": "flex"` (switch to `"torch"` or `"flashattn"` before inference/eval; see upstream README § Important: `attn_mode` Configuration).

The usual checkpoint directory is `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`. During evaluation, `ckpt_name` may be the short run name from training (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory.

## Deployment and Evaluation

What it does: serves the policy through XPolicyLab and connects it to a RoboDojo evaluation client. Use `eval.sh` for a same-machine run, or split server/client scripts for debugging and multi-machine evaluation.

Parameters used by `eval.sh`:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

```bash
cd XPolicyLab/policy/LingBot_VA
# Template: run same-machine policy server and RoboDojo environment client.
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls.
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

Parameters used by the split server/client flow:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |
| `policy_server_port` | Port exposed by the policy server, for example `5000`. |
| `policy_server_host` | Server bind host, for example `0.0.0.0` on the policy machine. |
| `policy_server_ip` | IP or hostname that the environment client uses to reach the policy server. |
| `additional_info` | Comma-separated runtime overrides passed to the eval client, for example `ckpt_name=...,action_type=joint`. |

```bash
cd XPolicyLab/policy/LingBot_VA
# Terminal 1 on the policy machine: start the policy server.
bash setup_eval_policy_server.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_conda_env> <policy_server_port> <policy_server_host>

# Example: bind the policy server to all interfaces on port 5000.
bash setup_eval_policy_server.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 <policy_conda_env> 5000 0.0.0.0

# Terminal 2 on the environment machine: connect RoboDojo to the policy server.
bash setup_eval_env_client.sh \
  <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> <additional_info> \
  <policy_server_port> <policy_server_ip>

# Example: connect to a policy server reachable at <policy_server_ip>:5000.
bash setup_eval_env_client.sh \
  RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 \
  0 <eval_env_conda_env> "ckpt_name=RoboDojo-cotrain-arx_x5-joint-0,action_type=joint" \
  5000 <policy_server_ip>
```

Set `EVAL_ENV_TYPE=debug` for offline shape/IO checks when the adapter supports it; leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation.

## Important Parameters

Common parameter meanings used across the commands above:

| Parameter | Description |
|---|---|
| `bench_name` | Benchmark or dataset family, usually `RoboDojo`. |
| `task_name` | RoboDojo simulation task to evaluate, for example `stack_bowls`. |
| `ckpt_name` | Checkpoint/run directory name, usually under `checkpoints/`. |
| `env_cfg_type` | Robot/environment configuration, for example `arx_x5`. |
| `action_type` | Action representation, for example `joint`. |
| `seed` | Evaluation seed. |
| `policy_gpu_id` | GPU used by the policy server. |
| `env_gpu_id` | GPU used by the RoboDojo simulation client. |
| `policy_conda_env` | Conda environment for the policy server. |
| `eval_env_conda_env` | Conda environment for RoboDojo simulation/client. |

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `policy_name` | Runtime or checkpoint option consumed by this adapter. |
| `env_cfg` | Runtime or checkpoint option consumed by this adapter. |
| `protocol` | Runtime or checkpoint option consumed by this adapter. |
| `config_name` | Runtime or checkpoint option consumed by this adapter. |
| `checkpoint_path` | Runtime or checkpoint option consumed by this adapter. |
| `base_model_path` | Runtime or checkpoint option consumed by this adapter. |
| `rollout_mode` | Runtime or checkpoint option consumed by this adapter. |
| `result_dir` | Runtime or checkpoint option consumed by this adapter. |
| `va_server_host` | Runtime or checkpoint option consumed by this adapter. |
| `va_server_port` | Runtime or checkpoint option consumed by this adapter. |
| `obs_transform_pipeline` | Runtime or checkpoint option consumed by this adapter. |

Frequently used environment variables detected in the adapter scripts:

| Variable | Notes |
|---|---|
| `AFTER` | Optional override used by the local scripts or upstream runtime. |
| `BASE_MODEL_PATH` | Optional override used by the local scripts or upstream runtime. |
| `CHECKPOINT_PATH` | Optional override used by the local scripts or upstream runtime. |
| `CONDA_ENV` | Optional override used by the local scripts or upstream runtime. |
| `CONFIG_NAME` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_CONFIG_NAME` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_VA_SERVER_HOST` | Optional override used by the local scripts or upstream runtime. |
| `DEFAULT_VA_SERVER_PORT` | Optional override used by the local scripts or upstream runtime. |
| `EXCEPT` | Optional override used by the local scripts or upstream runtime. |
| `JOINT_CONTROL_INDICES` | Optional override used by the local scripts or upstream runtime. |
| `LAUNCH_VA_SERVER` | Optional override used by the local scripts or upstream runtime. |
| `LEROBOT_DATASET_REPO_ID` | LeRobot dataset folder name under `LEROBOT_DATA_ROOT`; ignored when `LINGBOT_VA_DATASET_PATH` is set. |
| `LINGBOT_VA_DATASET_PATH` | **Required for training:** path to a latent-prepared LeRobot dataset (must contain `latents/`, `empty_emb.pt`, and `meta/episodes.jsonl` with `action_config`). |
| `LINGBOT_VA_BASE_MODEL_PATH` | Base model path for the wan_va backend; used when `deploy.yml` `base_model_path` is null. |

## Notes

- **Latent pipeline is mandatory:** do not point training at a plain LeRobot dataset without `latents/` and `empty_emb.pt`. Run [Demo Data Processing](#demo-data-processing) first.
- **`action_config` must align with latent filenames** (`episode_{idx}_{start}_{end}.pth`). `process_data.sh` keeps them consistent; if you edit segments manually, regenerate the matching latent files.
- Keep `ckpt_name` stable between data processing, training, and evaluation. For data-size ablations, encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- `task_name` is only the evaluation task; multi-task checkpoints can be evaluated on different tasks without renaming the checkpoint directory.
- Prefer running `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` separately when debugging dependency, CUDA, or model-loading issues.
- `eval.sh` auto-launches the wan_va backend (which holds the real weights). It requires a base model path: set `LINGBOT_VA_BASE_MODEL_PATH` or `base_model_path` in `deploy.yml`. To reuse an already-running backend, export `LINGBOT_VA_VA_HOST` / `LINGBOT_VA_VA_PORT`.
- **Training vs inference `attn_mode`:** set `"flex"` in `transformer/config.json` for training; switch to `"torch"` or `"flashattn"` before eval (upstream requirement).
- Full upstream data-prep details: `lingbot_va/README.md` § Post-Training LingBot-VA.
