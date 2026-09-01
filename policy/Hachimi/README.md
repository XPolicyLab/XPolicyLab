# Hachimi

**Contributor:** HKUST-RockAI | **Paper:** technical report in preparation | **arXiv:** TBD | **Original code:** [openpi](https://github.com/Physical-Intelligence/openpi) · [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) (arXiv:2510.12276)

`Hachimi` is a pi0.5-based VLA for RoboDojo, trained with the Spatial Forcing recipe (VGGT feature alignment) on top of the public `pi05_base` checkpoint. The XPolicyLab integration layer of this adapter is derived from [`policy/Spatial_Forcing`](../Spatial_Forcing) — `model.py`, `deploy.py`, `eval.sh`, `setup_eval_policy_server.sh` and `setup_eval_env_client.sh` are byte-identical to that adapter, so the runtime log prefix reads `[Spatial_Forcing]`. The vendored implementation lives in `open_sf_wam/`, a fork of the `openpi-SF/` subtree of [Spatial Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) (MIT, arXiv:2510.12276), itself built on openpi (Apache-2.0); upstream licenses are retained in `open_sf_wam/LICENSE*` and `open_sf_wam/src/vggt/LICENSE.txt` — see [License and attribution](#license-and-attribution). The policy runtime is a `uv`-managed environment rather than a conda environment.

Two things are specific to this submission:

1. **Self-generated, instruction-varied demonstrations** for the open-vocabulary dimension, merged with the full released training set. The released corpus pairs 35 fixed instruction strings with 35 scenes one-to-one and has no demonstrations for the Open-dimension tasks; ours vary the instruction per episode with instance-level object descriptions. Layouts come from our own generator; no released evaluation layout is read during data generation or training. (Released layouts were read offline, after the fact, only to measure how much of the evaluation vocabulary the corpus covers.)
2. **An action-conditioned future latent prediction head**: a FiLM-conditioned MLP predicting the VGGT latents of the head camera 2 s ahead, conditioned on the action chunk, trained with a cosine loss (`future_loss_coeff=0.05`). **Training-time only** — `sample_actions` never calls it, so the deployed inference graph is identical to the Spatial Forcing adapter.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

**Supported:** `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`.

Files changed in `open_sf_wam/` relative to the upstream `open_sf` release:

| File | Change |
|---|---|
| `src/openpi/external_config/pi05sfwam_jax_robodojo_v21_merged.py` | new — training config of the submitted checkpoint (merged dataset + future head) |
| `src/openpi/external_config/pi05sf_jax_robodojo_v21_merged.py` | new — the same recipe without the future head, kept as the ablation config referenced below |
| `src/openpi/models/pi0_align.py`, `src/openpi/models/projectors.py` | the future-prediction head and its loss (training-time only) |
| `src/openpi/training/align_utils.py` | future-frame VGGT targets; pool only the alignment layer that is used instead of all 24 (memory) |
| `src/openpi/training/data_loader.py` | future-frame fields through `delta_timestamps` with an episode-end padding mask; skip decoding the unused fourth camera stream |
| `src/openpi/training/config.py` | config fields for the future head |
| `src/openpi/training/checkpoints.py` | `use_ocdbt=False` for orbax saves (network-filesystem safe); no effect on loading |
| `scripts/train_align.py` | wires the future loss; JAX compilation cache directory from the environment |

## Installation

```bash
cd XPolicyLab/policy/Hachimi
bash install.sh
bash download_ckpt.sh
```

`install.sh` requires `uv` on `PATH` and builds `open_sf_wam/.venv`; it ends by asserting that the training config registers from the vendored tree, so a broken package fails at install time rather than at evaluation time. `download_ckpt.sh` fetches the evaluated checkpoint (~12 GB) into the layout below; override the source with `CKPT_REPO=<owner>/<repo>`.

## Data Processing

No `process_data.sh` is shipped with this adapter. Training uses the released RoboDojo LeRobot v2.1 export merged with our self-generated demonstration corpus (1,978 validated episodes); that corpus and the conversion/merge pipeline that builds the merged dataset are **not yet public** and will be released together, per the research-disclosure plan accompanying our challenge submission. Until then the training stage below is reproducible only from a dataset you supply yourself in LeRobot v2.1 form under `HF_LEROBOT_HOME`, with normalization statistics computed by:

```bash
cd open_sf_wam
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05sfwam_jax_robodojo_v21_merged --num-workers 16
```

## Training

```bash
HF_LEROBOT_HOME=<lerobot datasets root> \
PI05_BASE_PATH=<pi05_base checkpoint dir> \
VGGT_WEIGHT_PATH=<VGGT-1B weights dir> \
PI05SF_ASSETS_DIR=<dir holding RoboDojo_lerobot_v21_merged/norm_stats.json> \
  bash train.sh
```

8×80 GB GPUs, 60k steps, ~76 h. Starts from the public `pi05_base` checkpoint
(`gs://openpi-assets/checkpoints/pi05_base`); the alignment projector and the future head are freshly initialized, and VGGT-1B is used frozen as the feature target. Optional: `EXP_NAME`, `STEPS`, `WORKERS`, `CKPT_BASE`.

Two launch flags deviate from the stock Spatial Forcing recipe and are required: `--no-sf-cache-enable` (VGGT targets computed online — the offline cache is read-only with `miss_policy=error`, so new episodes would fail at step 0) and `--align-vggt-devices 4 --fsdp-devices 4` (with the cache disabled the 8 GPUs split 4 VGGT / 4 policy FSDP; 4/4 is the only split that keeps the stock `batch_size=256` divisible).

**Checkpoint layout.** `train.sh` writes to `open_sf_wam/checkpoints_out/<EXP_NAME>/<step>/{params,assets}`, while evaluation resolves `checkpoints/<ckpt_name>/`. To evaluate a checkpoint you trained, copy or symlink it into place:

```bash
ln -s "$PWD/open_sf_wam/checkpoints_out/<EXP_NAME>/59999" checkpoints/RoboDojo-sim-arx_x5-joint-0
```

## Evaluation

```bash
cd XPolicyLab/policy/Hachimi

# wiring check without a simulator
EVAL_ENV_TYPE=debug bash eval.sh RoboDojo stack_bowls RoboDojo-sim-arx_x5-joint-0 \
  arx_x5 joint 0 0 0 uv <eval_env_conda_env>

# RoboDojo simulation
bash eval.sh RoboDojo <task_name> RoboDojo-sim-arx_x5-joint-0 arx_x5 joint <seed> \
  <policy_gpu_id> <env_gpu_id> uv <eval_env_conda_env>
```

`<policy_uv_env>` (argument 9) is `uv`, which resolves the `deploy.yml` `policy_uv_env_path`; there is no conda environment on the policy side. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys beyond the standard set, all of which should be checked before evaluation:

| Key | Meaning |
|---|---|
| `checkpoint_num` | training step selected inside the checkpoint root (`59999`) |
| `result_dir` | where the adapter writes its own run artifacts |
| `obs_transform_pipeline` | observation transform preset (`xspark-v1.0`) |
| `policy_uv_env_path` | uv environment root, relative to this directory (`open_sf_wam`) |
| `train_config_name` | openpi training config to reconstruct the model (`pi05sfwam_jax_robodojo_v21_merged`) |
| `repo_id` | subdirectory of the checkpoint's `assets/` holding `norm_stats.json` (`RoboDojo_lerobot_v21_merged`) |

Environment variables: `OPENPI_DATA_HOME` overrides the OpenPI cache used by the policy server; `UV_CACHE_DIR` overrides the uv cache used by `install.sh`.

Checkpoint layout expected by evaluation:

```
checkpoints/RoboDojo-sim-arx_x5-joint-0/
├── params/                                            openpi/orbax parameters
└── assets/RoboDojo_lerobot_v21_merged/norm_stats.json
```

## License and attribution

| Component | License |
|---|---|
| openpi (base stack) | Apache-2.0 — `open_sf_wam/LICENSE` |
| Spatial Forcing (VGGT feature alignment, `align_projector`) | MIT — `open_sf_wam/LICENSE_SPATIAL_FORCING`, Copyright (c) 2025 Fuhao Li, Wenxuan Song, Han Zhao, Jingbo Wang, Pengxiang Ding, Donglin Wang, Long Zeng, Haoang Li |
| Gemma / PaliGemma weights (via `pi05_base`) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — `open_sf_wam/LICENSE_GEMMA.txt` |
| `open_sf_wam/src/vggt/` — Meta's VGGT | [VGGT License](https://github.com/facebookresearch/vggt) — `open_sf_wam/src/vggt/LICENSE.txt`, **not** Apache-2.0; some files under `src/vggt/layers/` are DINOv2-derived and carry their own Apache-2.0 headers |

**No VGGT weights are redistributed here.** VGGT-1B is downloaded separately by the user (`VGGT_WEIGHT_PATH`) and used frozen as a training-time feature-alignment target; it contributes no parameters to the published checkpoint.

Spatial Forcing is the basis of the `align_projector` weights: Li Fuhao, Song Wenxuan, Zhao Han, Wang Jingbo, Ding Pengxiang, Wang Donglin, Zeng Long, Li Haoang. *Spatial Forcing: Implicit Spatial Representation Alignment for Vision-Language-Action Model.* arXiv:2510.12276, 2025.

The published checkpoint is a derivative of `pi05_base` and therefore of Gemma; the Gemma Terms of Use apply to it and to any further derivative.

## Limitations / notes

- Single embodiment: the adapter and the checkpoint support `arx_x5` with `joint` actions only.
- The self-generated demonstration corpus and its conversion pipeline are not yet public (see Data Processing), so the training stage is not end-to-end reproducible from this repository today.
- Of the 85 `general_pickup` target assets, 12 are not graspable by the ARX X5 gripper (jaw opening or geometry), which caps the achievable open-vocabulary coverage independently of the policy.
- `install.sh` and `setup_eval_policy_server.sh` derive a workspace root five levels above the policy directory to place caches; on an unusually shallow checkout, set `UV_CACHE_DIR` and `OPENPI_DATA_HOME` explicitly.
