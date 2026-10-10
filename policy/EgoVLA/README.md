# EgoVLA

**Contributor:** XPolicyLab | **Paper:** [EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos](https://arxiv.org/abs/2507.12440) | **arXiv:** https://arxiv.org/abs/2507.12440 | **Original code:** https://github.com/RchalYang/EgoVLA_Release

`EgoVLA` adapts the released EgoVLA architecture to the SparkArena `tianji_marvin_wuji` raw-joint task. The adapter keeps the upstream model tree in `EgoVLA_Release/`; SparkArena preprocessing, the lazy HDF5 loader, and the training entry follow the corresponding upstream `human_plan/` and `VILA/llava/` directories.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). This adapter is documented for SparkArena.

## Installation

The installer keeps dependencies inside the policy directory and does not require machine-specific paths. Set `EGOVLA_CONDA_ENV` to the training/evaluation environment when the active shell is not already configured:

```bash
source <conda_root>/etc/profile.d/conda.sh
conda activate egovla
cd XPolicyLab/policy/EgoVLA
export EGOVLA_CONDA_ENV=egovla
bash install.sh
```

MANO dependencies and licensed MANO model files are optional. The SparkArena raw-joint recipe uses `--use_mano False`; set `EGOVLA_INSTALL_MANO=1` only when the MANO assets and `EGOVLA_MANO_ROOT` are available.

## Data Processing

`process_data.sh` converts the seven SparkArena tasks into the lazy HDF5 manifest consumed by the training dataset. Set `SPARKARENA_RAW_ROOT` to the raw root containing `<task>/tianji_marvin_wuji/data/episode_*.hdf5`:

```bash
cd XPolicyLab/policy/EgoVLA
export SPARKARENA_RAW_ROOT=<spark0_bench_7tasks_root>
bash process_data.sh SparkArena egovla tianji_marvin_wuji joint
```

For a small alignment check, limit conversion to one episode per task and use a 20% validation split:

```bash
SPARKARENA_VAL_PERCENT=20 \
bash process_data.sh SparkArena egovla tianji_marvin_wuji joint 1
```

The converter reads these source keys:

- `/state/left_arm_joint_states` and `/state/right_arm_joint_states` (7 joints per arm)
- `/state/left_ee_joint_states` and `/state/right_ee_joint_states` (20 joints per hand)
- the matching four `/action/*_joint_states` datasets
- `/vision/cam_head/colors` for the model image

The leading `/` in these entries denotes an HDF5-internal key, not a machine-specific filesystem path.

It writes `data/SparkArena-egovla-tianji_marvin_wuji-joint/` with `train.jsonl`, `val.jsonl`, `metadata.json`, `sparkarena_provenance.json`, and the 54-D state/action statistics. The source HDF5 files remain read-only. The image contract is RGB, full-frame 480×640 resized to 384×384; no crop, pad, or channel swap is applied.

## Model Assets

Download the official human-video pretrained weights into the vendored release directory:

```bash
cd XPolicyLab/policy/EgoVLA/EgoVLA_Release
huggingface-cli download rchal97/ego_vla_human_video_pretrained \
  --repo-type model \
  --local-dir checkpoints
```

The training wrapper uses `EgoVLA_Release/checkpoints` by default. Set `EGOVLA_PRETRAINED_PATH` when the weights are stored elsewhere.

## Training

Training is restricted to eight A800 GPUs and enforces the SparkArena raw-joint contract. The default effective global batch is `8 GPUs × 4 samples × 2 accumulation = 64`; the run has 80,000 steps and saves/evaluates every 10,000 steps.

```bash
cd XPolicyLab/policy/EgoVLA
export EGOVLA_CONDA_ENV=<training_env>
export SPARKARENA_RAW_ROOT=<spark0_bench_7tasks_root>
export WANDB_API_KEY=<wandb_api_key>
bash train.sh SparkArena egovla tianji_marvin_wuji joint 0 all
```

Run `process_data.sh` first. Checkpoints are written to `checkpoints/SparkArena-egovla-tianji_marvin_wuji-joint-0/` unless `EGOVLA_OUTPUT_DIR` is set. The evaluation side expects a finalized checkpoint directory such as `checkpoint-50000` under that run directory.

## Evaluation

The policy server validates the SparkArena contract. The evaluator checkout must provide `task/SparkArena` and `Assets/Object/SparkArena`. The standard wrapper starts both sides on one machine:

```bash
cd XPolicyLab/policy/EgoVLA
export EGOVLA_EVAL_ROOT=<absolute_evaluator_checkout>
export EGOVLA_CONDA_ENV=<policy_env>
export EGOVLA_EVAL_CONDA_ENV=<evaluator_env>

bash eval.sh SparkArena put_food_in_microwave \
  checkpoints/SparkArena-egovla-tianji_marvin_wuji-joint-0/checkpoint-50000 \
  tianji_marvin_wuji joint 0 0 1 "${EGOVLA_CONDA_ENV}" "${EGOVLA_EVAL_CONDA_ENV}"
```

For split evaluation, set `EGOVLA_COMPONENT=policy` on the policy machine and `EGOVLA_COMPONENT=environment` on the evaluator machine. Set `EGOVLA_POLICY_SERVER_HOST`, `EGOVLA_POLICY_SERVER_PORT`, and the evaluator control variables required by the shared deployment flow; the two setup scripts verify the checkpoint provenance and deploy contract before starting.

`EVAL_ENV_TYPE=debug` is available for protocol checks without a simulator. Leave it unset for the simulator evaluation.

## Configuration

| Variable | Purpose |
|---|---|
| `SPARKARENA_RAW_ROOT` | Raw `spark0_bench_7tasks` root used by preprocessing and training. |
| `EGOVLA_DATA_DIR` | Processed manifest/statistics directory override. |
| `EGOVLA_PRETRAINED_PATH` | Local official pretrained checkpoint directory override. |
| `EGOVLA_UPSTREAM_ROOT` | Alternate `EgoVLA_Release` tree; defaults to the vendored tree. |
| `EGOVLA_OUTPUT_DIR` | Training output directory override. |
| `EGOVLA_CONDA_ENV` / `EGOVLA_PYTHON_BIN` / `EGOVLA_TORCHRUN_BIN` | Training or policy-server runtime selection. |
| `WANDB_API_KEY` | Required by production training. |
| `EGOVLA_RESUME` | Set to `1` only for an intentional verified resume. |
| `EGOVLA_MAX_STEPS`, `EGOVLA_SAVE_STEPS`, `EGOVLA_EVAL_STEPS` | Training schedule overrides; production defaults are 80000/10000/10000. |
| `EGOVLA_EVAL_ROOT` | Absolute evaluator checkout used by the evaluation client. |
| `EGOVLA_EVAL_CONDA_ENV` | Evaluator environment used by `eval.sh`. |
| `EGOVLA_COMPONENT` | `policy` or `environment` for split evaluation. |

## Notes

- The adapter supports only `SparkArena`, `egovla`, `tianji_marvin_wuji`, and `joint`.
- The action target is the original `/action/*` joint data at the same timestep as the observation; next-state and MANO action fields are not used.
- The policy uses the head camera only and returns 30-step chunks of 54 raw joint values.
- The original `EgoVLA_Release` model architecture and assets remain under the release tree; SparkArena preprocessing and dataset integration live in its corresponding `human_plan/` and `VILA/llava/` locations.
