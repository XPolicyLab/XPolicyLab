# AHA_WAM

**Contributor:** RoboDojo Team | **Paper:** AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing | **arXiv:** https://arxiv.org/abs/2606.09811 | **Original code:** https://github.com/serene-sivy/AHA-WAM

`AHA_WAM` adapts the AHA-WAM world-action model to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `AHAWAM/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/AHA_WAM
bash install.sh
conda activate <policy_env>  # e.g. aha-wam
```

## Data Processing

No top-level `process_data.sh`. Training consumes the prepared RoboDojo LeRobot v2.1 video dataset directly (see Training below), in the standard keys of `XPolicyLab/scripts/transform_lerobot_v21_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)) — run that script to build a dataset for your own task subset or resolution. For anything beyond it, follow the upstream README under `AHAWAM/`.

## Model Assets

Prepare the Wan/DiffSynth model cache and the ActionDiT backbone once before the first training run (upstream details: [AHA-WAM README](https://github.com/serene-sivy/AHA-WAM#model-assets--checkpoints)):

```bash
cd XPolicyLab/policy/AHA_WAM
conda activate <policy_env>
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth/model/cache

cd AHAWAM
mkdir -p checkpoints
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

If preprocessing fails on unresolved Hydra placeholders in `configs/model/ahawam.yaml`, rerun with `configs/model/ahawam_preprocess.yaml` and the same `--output` path. Set `AHA_WAM_ACTION_DIT_PATH` when the backbone file is stored outside the default location.

## Training

```bash
cd XPolicyLab/policy/AHA_WAM
export AHA_WAM_TRAIN_DATASET_DIR=/path/to/RoboDojo_lerobot_v21_video
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/diffsynth/model/cache
export AHA_WAM_RESUME=/path/to/official-compatible/states_5000

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path. The training dataset must contain `meta/`, `dataset_stats.json`, and a T5 text-embedding cache at `text_embeds_cache/` unless overridden via `AHA_WAM_TRAIN_DATASET_STATS_PATH` / `AHA_WAM_TEXT_EMBED_CACHE_DIR`. Multi-process count is inferred from a comma-separated `gpu_id`.

The default task is `robodojo_ahawam_finetune_32_kv_xpolicy`: batch size 8,
gradient accumulation 2, learning rate `1e-5`, 32 epochs / 30,000 max steps,
and GC every 50 optimizer steps. Set `AHA_WAM_RESUME` to the reference
`states_5000`-compatible state for a true official-style finetune; an unset
resume starts from the configured base weights and is not equivalent.

## Evaluation

```bash
cd XPolicyLab/policy/AHA_WAM
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. AHA-WAM batch evaluation is intentionally disabled because one model instance owns one mutable history state; use one policy-server process per environment. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `env_cfg_root`, `action_dim`, `elava_root`, `task_config`, `checkpoint_path`, `dataset_stats_path`, `diffsynth_model_base_path`, `sim_cfg_name`, `sim_task`, `device`, `mixed_precision`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `AHA_WAM_TRAIN_DATASET_DIR` | Required for training; points to the prepared RoboDojo LeRobot v2.1 video dataset. |
| `DIFFSYNTH_MODEL_BASE_PATH` | Required for training and normally required for model loading; points to the Wan/DiffSynth model cache. |
| `AHA_WAM_ACTION_DIT_PATH` | Optional ActionDiT backbone override; defaults to `AHAWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`. |
| `AHA_WAM_TRAIN_DATASET_STATS_PATH` | Optional training stats override; defaults to `$AHA_WAM_TRAIN_DATASET_DIR/dataset_stats.json`. |
| `AHA_WAM_TEXT_EMBED_CACHE_DIR` | Optional text embedding cache override; defaults to `$AHA_WAM_TRAIN_DATASET_DIR/text_embeds_cache`. |
| `AHA_WAM_OUTPUT_ROOT` | Optional training checkpoint root; defaults to `policy/AHA_WAM/checkpoints`. |
| `AHA_WAM_CHECKPOINT_PATH` | Optional explicit eval checkpoint file; overrides `ckpt_name` lookup. |
| `AHA_WAM_DATASET_STATS_PATH` | Optional explicit eval dataset stats file. |
| `AHA_WAM_CKPT_SETTING` | Optional eval run directory override under `checkpoints/`. |
| `AHA_WAM_ENV_CFG_ROOT` | Optional env config root; defaults to `<workspace>/env_cfg` (the parent of this checkout). |
| `AHA_WAM_APPTAINER_IMAGE` | Optional Apptainer image for the policy server. |
| `AHA_WAM_APPTAINER_BINDS` | Optional Apptainer bind arguments when using `AHA_WAM_APPTAINER_IMAGE`. |
| `AHA_WAM_CHUNKS_PER_VIDEO_PREFILL` | Optional video prefill cadence; official RobotWin-style default is `2`. |
| `AHA_WAM_ALLOW_DUMMY_POLICY` | Debug-only option to skip real checkpoint/stats loading. |
| `AHA_WAM_DEBUG_EVAL_EPISODE_NUM` | Debug-client episode count override. |
| `XPOLICYLAB_BENCH_ROOT` | Optional client `--root_dir` override; defaults to the RoboDojo repo root. |
