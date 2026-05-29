# Mem_0 Installation

Conda only (no Docker). Three environments for a full Mn pipeline:

- `mem0` — execution module, data conversion, inference
- `llama_factory` — planning module LoRA training (Mn tasks)
- `vllm` — planning module inference server (Mn eval)

## 1. Execution / inference env

```bash
cd policy/Mem_0
bash install.sh mem0
```

## 2. Backbone checkpoints

```bash
cd Mem_0/checkpoints
python _download.py     # Qwen3-VL-2B-Instruct (execution) + Qwen3-VL-8B-Instruct (planning)
```

## 3. Planning training env (Mn tasks)

```bash
cd policy/Mem_0
bash install_planning.sh
```

This creates the `llama_factory` conda env, clones [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) into `Mem_0/LlamaFactory` when missing, and runs `pip install -e` plus metrics/wandb dependencies.

Optional: `wandb login` before training if `ENABLE_WANDB=true` (default).

## 4. Training workflow

Unified entrypoint [`train.sh`](train.sh):

```bash
bash train.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> \
             <action_type> <seed> <gpu_ids> [train_module]
```

| `train_module` (8th arg) | Behavior |
|--------------------------|----------|
| `both` (default) | Execution then Planning (Mn full pipeline) |
| `execution` | Execution Module only (M1 or Mn low-level policy) |
| `planning` | Planning Module only (Mn; requires `process_data.sh ... Mn`) |

**M1 (single-stage)** — pass `execution` explicitly (Planning preflight will fail on M1 data):

```bash
bash process_data.sh RoboDojo test_data arx_x5 3 joint M1
bash train.sh RoboDojo test_data arx_x5 3 joint 42 0 execution
```

**Mn (multi-stage)** — omit 8th arg or pass `both`:

```bash
bash process_data.sh RoboDojo cover_blocks arx_x5 50 joint Mn
bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7
```

Planning-only or execution-only on Mn data:

```bash
bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7 execution
bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7 planning
```

Legacy wrapper `bash train_planning.sh ...` (7 args) forwards to `train.sh ... planning`.

Checkpoints under `Mem_0/checkpoints/`:

- Execution: `<dataset_id>-seed<seed>/` (torchrun outputs + `train_config.yaml`)
- Planning LoRA: `<dataset_id>-seed<seed>_planning_sft_lora/`
- Planning merged: `<dataset_id>-seed<seed>_planning_merged/`
- Planning configs: `<dataset_id>-seed<seed>/planning_train.yaml`, `planning_merge.yaml`

Planning partial runs: `STEPS=prepare`, `STEPS="train merge"`, or `DRY_RUN=true`. Execution smoke: `DRY_RUN=true` skips torchrun after config generation.

Planning data prep uses episodes `[EPISODE_START_ID, EPISODE_END_ID)` (LeRobot global `episode_index`). If `EPISODE_END_ID` is unset, `train.sh` sets it to `total_episodes` from `meta/info.json`.

**Mn-only planning on cotrain** — no code filter; set episode IDs by layout (`process_data_batch.sh` writes all M1 tasks first, then Mn, 100 episodes each):

```bash
# 32 M1 × 100 + 3 Mn × 100 = 3500  →  Mn episodes are [3200, 3500)
REPO_ID=Mem_0/lerobot_datasets/RoboDojo-cotrain-arx_x5-100-joint \
EPISODE_START_ID=3200 EPISODE_END_ID=3500 FORCE_PREPARE=true \
bash train.sh RoboDojo cover_blocks arx_x5 100 joint 42 0 planning
```

**Single Mn task** (`process_data.sh ... Mn`): use `[0, expert_data_num)` or omit both env vars when the dataset only contains that task.

Override with `EPISODE_END_ID` / `EPISODE_START_ID` for subsets; use `FORCE_PREPARE=true` after changing the range.

## 5. Planning inference (Mn eval)

After training, serve the **merged** planning weights with vLLM (not the base 8B checkpoint):

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
pip install vllm

export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve Mem_0/checkpoints/<dataset_id>-seed<seed>_planning_merged \
  --tensor-parallel-size 4 \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 \
  --port 8123
```

Point eval overrides at the vLLM URL, e.g. `VLLM_URL=http://localhost:8123/v1 bash eval.sh ...`.

Or let `eval.sh` auto-start vLLM for Mn tasks (see section 6).

## 6. Evaluation (debug / sim / Mn dual-port)

Eval uses the XPolicyLab three-script layout plus optional planning server:

- `eval.sh` — orchestrator (11 contract args + optional 12th `planning_gpu_ids`)
- `setup_eval_policy_server.sh` — execution module (`mem0` conda)
- `setup_eval_env_client.sh` — env client (`eval_env_conda_env`; routes via `deploy.yml`)
- `setup_eval_planning_server.sh` — vLLM for Mn (`vllm` conda; auto-started when 12th arg set)

```bash
bash eval.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> \
             <expert_data_num> <action_type> <seed> \
             <policy_gpu_id> <env_gpu_id> \
             <policy_conda_env> <eval_env_conda_env> [planning_gpu_ids]
```

`task_name` is the simulator task; `ckpt_name` resolves checkpoint paths (may differ, e.g. `cotrain`).

Checkpoint layout (matches `train.sh`):

- Execution: `Mem_0/checkpoints/<dataset_name>-<ckpt_name>-<env_cfg_type>-<expert_data_num>-<action_type>-seed<seed>/final_step*.pt`
- Norm stats: `Mem_0/assets/<ckpt_name>/norm_stats.json`
- Planning merged: `Mem_0/checkpoints/<dataset_name>-<ckpt_name>-...-seed<seed>_planning_merged/`

**Debug gate (M1)** — set `eval_env: debug` in `deploy.yml`:

```bash
cd policy/Mem_0
bash eval.sh RoboDojo swap_blocks swap_blocks arx_x5 50 joint 0 0 0 mem0 XPolicyLab
```

**Simulator** — set `eval_env: sim` in `deploy.yml` (no change to `eval.sh`). Requires `${REPO_ROOT}/scripts/eval_policy.sh` in the XPolicyLab base env.

**Mn dual-port** — pass `planning_gpu_ids` to auto-start vLLM, or set `VLLM_URL` manually:

```bash
GLOBAL_TASK="On the table, cover blocks with lids..." \
bash eval.sh RoboDojo cover_blocks cover_blocks arx_x5 50 joint 0 \
    0 0 mem0 XPolicyLab 4,5,6,7
```

GPU layout example: planning on `4,5,6,7`, execution on `0`, sim/debug client on `0`.

### Eval environment variables

| Variable | Purpose |
|----------|---------|
| `MEM0_EXECUTION_CKPT` | Override execution checkpoint path |
| `MEM0_STATE_STATS_PATH` | Override norm stats JSON |
| `MEM0_PLANNING_MERGED_PATH` | Override merged planning weights for vLLM |
| `MEM0_VLLM_TP_SIZE` | vLLM tensor parallel size (default: GPU count) |
| `VLLM_URL` | Skip auto vLLM; use existing server (`.../v1`) |
| `GLOBAL_TASK` | M1/Mn episode-level task instruction |
| `CONDA_ENV_VLLM` | Conda env for vLLM (default `vllm`) |
| `MEM0_ACTION_HORIZON` | Override `action_horizon` (default 30) |
| `MEM0_THRESHOLD` | Mn subtask-end count threshold (default 2) |

Mn eval requires `ffmpeg` in the `mem0` conda env for planner video buffers.
