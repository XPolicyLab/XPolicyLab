# Mem_0 (XPolicyLab)

XPolicyLab adapter for [MemoryMatters / Mem_0](Mem_0/README.md) dual-module VLA (execution + optional Mn planning).

Environment and backbone weights: [INSTALLATION.md](INSTALLATION.md).

## Quick start

```bash
cd policy/Mem_0
# install envs + download weights — see INSTALLATION.md
bash install.sh mem0
cd Mem_0/checkpoints && python _download.py
```

### M1 (single-stage)

```bash
bash process_data.sh RoboDojo test_data arx_x5 3 joint M1
python Mem_0/xpolicylab_adapter/gen_norm_stats.py \
    --repo_id data/RoboDojo-test_data-arx_x5-joint-lerobot --ckpt_name test_data
bash train.sh RoboDojo test_data arx_x5 3 joint 42 0 execution
# deploy.yml: eval_env: debug
bash eval.sh RoboDojo test_data test_data arx_x5 joint 0 0 0 mem0 XPolicyLab
```

### Mn (multi-stage + planning)

```bash
bash process_data.sh RoboDojo cover_blocks arx_x5 50 joint Mn
bash install_planning.sh   # INSTALLATION.md §2
bash train.sh RoboDojo cover_blocks arx_x5 50 joint 42 0,1,2,3,4,5,6,7 both
GLOBAL_TASK="..." bash eval.sh RoboDojo cover_blocks cover_blocks arx_x5 joint 0 \
    0 0 mem0 XPolicyLab 4,5,6,7
```

### Cotrain batch

```bash
bash process_data_batch.sh RoboDojo_first100 arx_x5 100 joint
bash train.sh RoboDojo cotrain arx_x5 100 joint 42 0 execution
bash eval.sh RoboDojo cover_blocks cotrain arx_x5 joint 0 0 0 mem0 XPolicyLab
```

Set `MEM0_EXPERT_DATA_NUM=100` when legacy or ambiguous artifact paths need disambiguation.

## Training

```bash
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <expert_data_num> \
             <action_type> <seed> <gpu_ids> [train_module]
```

| `train_module` (8th arg) | Behavior |
| --- | --- |
| `both` (default) | Execution then planning (Mn full pipeline) |
| `execution` | Execution module only |
| `planning` | Planning module only (Mn data) |

**Cotrain planning slice** (32 M1 × 100 + 3 Mn × 100 → Mn episodes `[3200, 3500)`):

```bash
REPO_ID=data/RoboDojo-cotrain-arx_x5-100-joint-lerobot \
EPISODE_START_ID=3200 EPISODE_END_ID=3500 FORCE_PREPARE=true \
bash train.sh RoboDojo cotrain arx_x5 100 joint 42 0 planning
```

Checkpoints under `policy/Mem_0/checkpoints/`:

- Execution: `<dataset>-<ckpt>-<env>-<action>-<seed>/`
- Planning merged: `<run_id>_planning_merged/`

## Norm stats (inference)

```bash
python Mem_0/xpolicylab_adapter/gen_norm_stats.py \
    --repo_id data/RoboDojo-test_data-arx_x5-joint-lerobot \
    --ckpt_name test_data
```

Writes `policy/Mem_0/assets/<ckpt_name>/norm_stats.json`.

## Evaluation

Set `eval_env: debug` or `sim` in `deploy.yml`.

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> \
             <action_type> <seed> <policy_gpu_id> <env_gpu_id> \
             <policy_conda_env> <eval_env_conda_env> [planning_gpu_ids]
```

Examples:

```bash
bash eval.sh RoboDojo swap_blocks swap_blocks arx_x5 joint 0 0 0 mem0 XPolicyLab

GLOBAL_TASK="On the table, cover blocks with lids..." \
bash eval.sh RoboDojo cover_blocks cover_blocks arx_x5 joint 0 \
    0 0 mem0 XPolicyLab 4,5,6,7
```

| Variable | Purpose |
| --- | --- |
| `MEM0_EXPERT_DATA_NUM` | Disambiguate legacy/new artifact paths |
| `MEM0_EXECUTION_CKPT` | Override execution checkpoint file |
| `MEM0_STATE_STATS_PATH` | Override norm stats JSON |
| `MEM0_PLANNING_MERGED_PATH` | Override merged planning weights |
| `VLLM_URL` | Use existing vLLM server (`.../v1`) |
| `GLOBAL_TASK` | Episode-level task instruction |
| `MEM0_LEGACY_PATHS` | Write/read legacy `Mem_0/lerobot_datasets` layout |

## Artifact naming (README §4.2)

| Artifact | Standard name | Standard location |
| --- | --- | --- |
| Processed dataset | `<dataset>-<ckpt>-<env>-<action>-lerobot` | `policy/Mem_0/data/` |
| Training checkpoint | `<dataset>-<ckpt>-<env>-<action>-<seed>` | `policy/Mem_0/checkpoints/` |
| Norm stats | `assets/<ckpt_name>/norm_stats.json` | `policy/Mem_0/assets/` |
| Planning merged | `<run_id>_planning_merged` | `policy/Mem_0/checkpoints/` |
| Qwen backbones | `Qwen3-VL-2B-Instruct`, `Qwen3-VL-8B-Instruct` | `Mem_0/checkpoints/` |

**Legacy fallback** (auto-resolved by `Mem_0/xpolicylab_adapter/_artifact_paths.sh`):

- `Mem_0/lerobot_datasets/<dataset>-<ckpt>-<env>-<N>-<action>`
- `Mem_0/checkpoints/<dataset>-<ckpt>-<env>-<N>-<action>-seed<seed>/`

Write legacy paths only with `MEM0_LEGACY_PATHS=1`.

## Parameters

| Parameter | `process_data` / `train` | `eval` |
| --- | --- | --- |
| `ckpt_name` | Names experiment + artifacts; HDF5 source dir for single-task | Resolves checkpoint |
| `task_name` | — | Simulator task (env client) |
| `expert_data_num` | Episode count per task | Not passed; use `MEM0_EXPERT_DATA_NUM` if needed |

## Eval layout

Standard XPolicyLab three-script split plus Mn extension:

- `eval.sh` — 10 args (+ optional 11th `planning_gpu_ids` for Mn vLLM)
- `setup_eval_policy_server.sh` — execution module (`mem0` conda)
- `setup_eval_env_client.sh` — debug/sim/real client (`deploy.yml` `eval_env`)
- `setup_eval_planning_server.sh` — vLLM for Mn (auto-started by `eval.sh`)

`deploy.py` uses `begin_episode` / `step` / `reset` (chunk-based rollout), not the minimal `update_obs`/`get_action` loop.

## Environments

| Env | Purpose |
| --- | --- |
| `mem0` | Data conversion, execution train/infer |
| `llama_factory` | Mn planning LoRA train |
| `vllm` | Mn planning inference |

## Pitfalls

- Do not add `BGR2RGB` after `decode_image_bit` on XPolicyLab HDF5 (preview mp4 via `VideoCapture` is the exception).
- `ckpt_name` resolves checkpoints; `task_name` is only the simulator task at eval.
- Train and eval must share the same `run_id` (`<dataset>-<ckpt>-<env>-<action>-<seed>`).
- Mn eval needs vLLM (`planning_gpu_ids` or `VLLM_URL`) and `ffmpeg` in the `mem0` env.
- Set `MEM0_EXPERT_DATA_NUM` when resolving legacy artifacts that embed episode count in the path.

See upstream [Mem_0/README.md](Mem_0/README.md) for model architecture details.

## Legacy (do not use for XPolicyLab eval)

- `Mem_0/eval.sh`, `Mem_0/deploy_policy.py` — RMBench standalone API
- `Mem_0/scripts/hdf5_to_lerobot/` — superseded by `xpolicylab_adapter/`
