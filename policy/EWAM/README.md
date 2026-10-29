# EWAM

**Contributor:** EWAM Team | **Paper:** EWAM: Emergent Depth-Wise Specialization in a Unified Embodied Model — From Semantic Understanding through Visual Foresight to Action | **arXiv:** [2609.39973](https://arxiv.org/abs/2609.39973) | **Original code:** https://github.com/Wanghao00pro/EWAM

`EWAM` adapts the EWAM unified embodied model (Qwen3-VL-2B vision-language expert + Wan2.2-TI2V-5B video expert + action expert, jointly denoised by flow matching) to XPolicyLab for RoboTwin 2.0 (`aloha_agilex`) and RoboDojo (`arx_x5`), with absolute joint control and 16-step chunks. `EWAM/` holds the vendored upstream EWAM code (model, training, data loaders).

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Create a Python >= 3.10 environment, install the PyTorch CUDA wheel, then the adapter dependencies:

```bash
conda create -n ewam python=3.10
conda activate ewam
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

cd XPolicyLab/policy/EWAM
bash install.sh
# Required: the Wan and action-expert attention calls flash-attn directly (no SDPA fallback).
pip install flash-attn --no-build-isolation
```

`install.sh` editable-installs XPolicyLab and `requirements.txt`. The vendored `EWAM/` tree is imported from source by `model.py`. `transformers` is pinned to `5.0.0rc0` like upstream EWAM; later releases change the return type of Qwen3-VL `get_image_features`, which the vendored code unpacks as a tuple. On headless servers without `libGL`, replace `opencv-python` with `opencv-python-headless`.

## Data Processing

```bash
cd XPolicyLab/policy/EWAM
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: first 50 clean demos of every RoboTwin task under ../data/RoboTwin/
bash process_data.sh RoboTwin clean50 aloha_agilex joint 50 all

# Example: every RoboDojo task under ../data/RoboDojo/
bash process_data.sh RoboDojo all arx_x5 joint all
```

EWAM does not train on LeRobot exports. `process_data.sh` reads XPolicyLab-standard HDF5, and for RoboTwin also the RoboTwin 2.0 release layout (`<task>/demo_clean`; pass `<task>/demo_randomized` to add randomized demos); `expert_data_num` counts per task dir. RoboTwin output matches upstream [`robotwin_converter.py`](https://github.com/Wanghao00pro/EWAM/tree/main/data/robotwin2/robotwin_data_convert); for RoboDojo the HDF5 is read directly and only UMT5 embeddings are precomputed. UMT5 encoding needs a GPU and the Wan2.2 files from `download_checkpoints.sh`. Optional variables: `EWAM_RAW_DATA_ROOT` (default `../data/<bench_name>`), `EWAM_DATA_OUT_DIR`, `EWAM_WAN_PATH`, `EWAM_PROCESS_WORKERS`.

## Training

```bash
cd XPolicyLab/policy/EWAM
bash download_checkpoints.sh pretrain
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example
bash train.sh RoboTwin clean50 aloha_agilex joint 0 0,1,2,3,4,5,6,7
```

`train.sh` finetunes the released `pretrain` checkpoint with `EWAM/configs/ewam_robotwin.yaml` or `ewam_robodojo.yaml`. For multi-node runs, launch it on every node with `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT`. The run directory's `pytorch_model` link points at the final step, so `eval.sh` takes the same `ckpt_name`. Optional variables: `EWAM_DATA_DIR`, `EWAM_PRETRAIN_PATH`, `EWAM_WAN_PATH`, `EWAM_VLM_PATH`, `EWAM_MAX_STEPS` (defaults to the released checkpoint's step count), `EWAM_REPORT_TO` (`tensorboard`, `wandb` or `all`), `EWAM_DRY_RUN=1`.

## Evaluation

```bash
cd XPolicyLab/policy/EWAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: offline wiring check (no simulator, no weights loaded)
EVAL_ENV_TYPE=debug EWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboTwin beat_block_hammer dummy aloha_agilex joint 0 0 0 <policy_conda_env> <eval_env_conda_env>

# Example: released clean-to-randomized checkpoint on RoboTwin
bash eval.sh RoboTwin beat_block_hammer robotwin-c2r aloha_agilex joint 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>

# Example: released RoboDojo checkpoint
bash eval.sh RoboDojo push_T robodojo arx_x5 joint 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`ckpt_name` may be a directory under `checkpoints/`, the 5-tuple run directory, or a path to the directory / `mp_rank_00_model_states.pt` file. `<policy_conda_env>` may be a conda env name or an env prefix path. `EVAL_ENV_TYPE=debug` runs the offline wiring check; leave it unset or set `EVAL_ENV_TYPE=sim` for simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Model Assets

Released checkpoints are on [Hugging Face HaoWang00/EWAM](https://huggingface.co/HaoWang00/EWAM): `robotwin-c2r` (RoboTwin clean-to-randomized), `robotwin-indomain` (RoboTwin in-domain), `robodojo` (RoboDojo) and `pretrain` (the initialization `train.sh` finetunes from), each a single `mp_rank_00_model_states.pt` (~17 GB) holding the full model. Inference additionally needs the Wan2.2 VAE, the UMT5-XXL text encoder, and the Qwen3-VL processor files. `download_checkpoints.sh` fetches only those files (honors `HF_ENDPOINT`):

```bash
cd XPolicyLab/policy/EWAM
bash download_checkpoints.sh robotwin-c2r   # or robotwin-indomain / robodojo / pretrain
```

Resulting layout:

```text
checkpoints/
├── robotwin-c2r/mp_rank_00_model_states.pt
├── Wan2.2-TI2V-5B/        # config.json, Wan2.2_VAE.pth, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/
└── Qwen3-VL-2B-Instruct/  # config + tokenizer/processor files
```

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_path`, `wan_path`, `vlm_path`, `ewam_root`, `ewam_config`, `device`, `num_inference_steps`, `replan_steps`, `max_batch_size`, `scene_prefix`, `image_layout`, `action_stride`, `clip_gripper`, `default_instruction`, `allow_dummy_policy`.

| Variable | Notes |
|---|---|
| `EWAM_CKPT_PATH` | Explicit checkpoint directory or `mp_rank_00_model_states.pt` (overrides `checkpoint_path`). |
| `EWAM_WAN_PATH` | Wan2.2-TI2V-5B directory; default `checkpoints/Wan2.2-TI2V-5B`. |
| `EWAM_VLM_PATH` | Qwen3-VL-2B-Instruct directory; default `checkpoints/Qwen3-VL-2B-Instruct`. |
| `EWAM_ROOT` | Optional EWAM source override; defaults to the vendored `EWAM/`. |
| `EWAM_ALLOW_DUMMY_POLICY` | Debug-only: skip weight loading and return hold-position chunks. |

## Notes

- `deploy.py` differs from `demo_policy/deploy.py` in one way: EWAM conditions only on the observation at chunk start, so `update_obs` is called once per chunk instead of after every env step. Mid-chunk observations are rendered every 3 env steps for the eval video only and are not sent to the server. The executed actions are identical; this avoids sending ~2.7 MB of unused images per env step.
- Each `get_action` returns the full chunk (`replan_steps: null`). The VAE decode of predicted frames is skipped since only actions are used.
- `eval_batch` defaults to `false` (one env per sim process), the setting that produced the reported RoboDojo results. With `eval_batch: true`, the RoboDojo client runs up to `sim.scene.num_envs` envs per process, and `get_action_batch` runs envs whose VLM prompts have the same token length (in practice, the same instruction) in one batched forward, capped by `max_batch_size`; batched and per-env actions agree to bf16 rounding. On an H20, a batch of 10 takes about 7.3 s (vs 1.25 s for one env), and inference peaks at about 34 GiB allocated (30 GiB for one env), below the 42 GiB loading peak.
- The policy process peaks at about 42 GiB allocated (`torch.cuda.max_memory_allocated`) / 43 GB as reported by `nvidia-smi` on an H20 (EWAM 8.5B parameters + UMT5-XXL encoder, both bf16); use a GPU with at least 48 GB. Loading takes about 3 minutes and one `get_action` (10 flow-matching steps) about 1.25 s on an H20.
- Upstream `configs/ewam_robotwin.yaml` writes `norm_eps: 1e-5`, which PyYAML loads as a string; `model.py` casts it to `float`.
