# CSU-AI-0

**Contributor:** CSU-AI Team | **Paper:** Not applicable — routed XPolicyLab adapter | **arXiv:** Not applicable | **Original code:** [Xiaomi_Robotics_1](../Xiaomi_Robotics_1/README.md), [Spatial_Forcing](../Spatial_Forcing/README.md), [Hy_Embodied_05_VLA](../Hy_Embodied_05_VLA/README.md)

`CSU-AI-0` exposes one standard `Model(ModelTemplate)` policy while routing each task to one of three process-isolated experts. The frozen `QRouter` is a HistGradientBoosting classifier; Xiaomi, Spatial Forcing, and Hunyuan remain in separate runtime environments because their dependency stacks and action heads are incompatible.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install the router-side XPolicyLab dependencies, the public-download clients, and the three isolated expert environments:

```bash
cd XPolicyLab/policy/CSU-AI-0
bash install.sh
```

Download the three public expert checkpoints independently, normalize them under one local root, and run one complete path/size/SHA256 verification:

```bash
# Optional: inspect the exact sources, revisions, sizes and destinations first.
CSU_DOWNLOAD_DRY_RUN=1 bash download_checkpoints.sh

# Downloads all three sources and installs only after the full bundle passes.
bash download_checkpoints.sh
```

The pinned sources are recorded in `checkpoint_sources.json`:

| Expert | Public source | Pinned source |
|---|---|---|
| Xiaomi | [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) | Hugging Face dataset revision `c037c1b3183a030724e694d69a08cb62369ed285` |
| Spatial Forcing | [zzyzzzz699/pi05sf](https://www.modelscope.cn/models/zzyzzzz699/pi05sf) | ModelScope `master`, fail-closed against the 27-file SHA256 manifest |
| Hunyuan | [zzilch/rd20](https://www.modelscope.cn/models/zzilch/rd20) | ModelScope `master`, fail-closed against the 12-file SHA256 manifest |

ModelScope's public branches are mutable, so the downloader does not trust branch names alone: every normalized file must match the immutable LFS SHA256 copied into `checkpoint_sources.json` from the official RoboDojo dataset revision above. A changed upstream file stops installation.

The downloader stages all sources outside the final path. It runs a single complete verification and then atomically installs the bundle. The report is written next to the bundle as `CSU-AI-0-v1.verification.json`. An existing valid bundle is reused; an invalid bundle is never overwritten unless `CSU_REPLACE_INVALID_CHECKPOINTS=1`, in which case it is preserved as a timestamped backup.

## Data Processing

This is an evaluation-only submission. No data converter is shipped in this adapter; `process_data.sh` reports that QRouter consumes already prepared task metadata. The PR must declare the eval-only status, notify the maintainers, and state the planned training-code release date.

```bash
bash process_data.sh RoboDojo QRouter arx_x5 auto
```

## Training

Training is not included in this eval-only adapter. The checked-in `QRouter.joblib` is frozen, its SHA256 is pinned in `adapter_manifest.json`, and seed 0 was not used for training. `train.sh` reports the frozen status and does not alter the router.

```bash
bash train.sh RoboDojo QRouter arx_x5 auto 0 0
```

The submission PR must include an agreed training release date before review.

## Evaluation

The entry point follows the standard ten-argument XPolicyLab convention. `action_type=auto` lets the router supply the selected expert's standard action type.

```bash
cd XPolicyLab/policy/CSU-AI-0
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> \
  <seed> <policy_gpu_id> <env_gpu_id> <policy_env_or_uv_path> \
  <eval_env_conda_env>

# Offline closed-loop check
EVAL_ENV_TYPE=debug bash eval.sh \
  RoboDojo stack_bowls QRouter arx_x5 joint 0 0 0 mibot mibot

# Simulator-backed example
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo stack_bowls QRouter arx_x5 auto 0 0 0 mibot RoboDojo
```

The router first records the deterministic route, then the outer policy server starts only the selected expert and forwards reset, observation, action, case, and trial messages over localhost WebSocket.

## Model Assets

The code-only PR contains no expert weights. The public downloader installs one logical bundle without tensor-merging incompatible architectures:

```text
policy/CSU-AI-0/checkpoints/CSU-AI-0-v1/
├── xiaomi/RoboDojo-sim-arx_x5-ee-0/
│   ├── config.py
│   ├── config.yaml
│   └── model_states.pt
├── spatial_forcing/RoboDojo-sim-arx_x5-joint-0/
└── hunyuan/rd20/
```

For Xiaomi, the official source filename `last.ckpt/checkpoint/mp_rank_00_model_states.pt` is normalized to `model_states.pt` without modifying its bytes. Spatial Forcing and Hunyuan retain their upstream internal layouts. Run the verifier independently at any time with:

```bash
python verify_checkpoints.py
```

## Configuration

- Router: `QRouter`, `HistGradientBoostingClassifier`.
- Router SHA256: `1600870c4999b7e183bb34cb528d8fcb8e9a1527b30c4e6034b4fe5dc9972a87`.
- Training rows: 4,122 from seeds 1 and 2; seed 0 was excluded.
- Expected seed-0 routes: Xiaomi 2,000; Hunyuan 50; Spatial Forcing 50.
- Default protocol: WebSocket (`protocol: ws`).

Optional runtime overrides:

```bash
export CSU_CHECKPOINT_ROOT=/absolute/path/to/CSU-AI-0-v1
export CSU_XIAOMI_CHECKPOINT=/absolute/path/to/xiaomi
export CSU_SPATIAL_CHECKPOINT=/absolute/path/to/spatial-forcing
export CSU_HUNYUAN_CHECKPOINT=/absolute/path/to/hunyuan
export CSU_XIAOMI_HF_HOME=/absolute/path/to/huggingface-cache
export CSU_XIAOMI_POLICY_ENV=/absolute/path/to/mibot
export CSU_SPATIAL_POLICY_ENV=/absolute/path/to/spatial-env
export CSU_HUNYUAN_POLICY_ENV=/absolute/path/to/hunyuan-env
```

## Notes

- `model.py` implements `update_obs`, `update_obs_batch`, `get_action`, `get_action_batch`, and `reset` on `ModelTemplate`.
- `expert_proxy.py` owns the selected child process and its dedicated WebSocket I/O thread; expert dependencies are never imported into the router process.
- The proxy forwards the expert's standard action dictionaries unchanged, so dimensions remain those registered by the selected upstream XPolicyLab adapter.
- Contract tests are available under `tests/`; the required end-to-end smoke test is the `EVAL_ENV_TYPE=debug` command above.
