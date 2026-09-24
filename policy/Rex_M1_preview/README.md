# Rex_M1_preview

**Contributor:** Futian Lab & VisIncept | **Paper:** Not released | **arXiv:** Not released | **Original code:** See vendored `rex_m1/`.

`Rex_M1_preview` adapts the Rex-M1-preview policy to XPolicyLab/RoboDojo for evaluation. Integration scripts live at this directory level; the vendored implementation lives in `rex_m1/`. It supports RoboDojo simulation with Dual ARX5 (`arx_x5`) and `action_type=ee`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` for the manual setup, checkpoint layout, integrity hashes, and smoke checks. The recommended installer creates a conda environment with the policy dependencies:

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash install.sh
conda activate rex_m1_preview
```

Set `REX_M1_PREVIEW_CONDA_ENV` before running `install.sh` to use a different environment name.

## Data Processing

No `process_data.sh` is provided. This is an eval-only adapter, and RoboDojo observations are consumed online through XPolicyLab.

## Training

The training pipeline for Rex-M1-preview is not included in this eval-only release (ETA: TBD).

To be precise about what the vendored tree does and does not contain: upstream's training entry
points are still present under `rex_m1/`, and `rex_m1/mibot/models/VLA/xr1.py` carries the
past-reconstruction auxiliary-loss path (`ptp_length`, `ptp_loss_coefficient`). That path cannot be
exercised here — the vendored collators never emit `past_action`, so any `ptp_length > 0` run raises
— and the released checkpoint sets `ptp_length: 0`, which leaves it inert. What is genuinely absent
is the history sampling and collation this policy's checkpoint was trained with: the vendored
`json_dataset.py` and `custom_collate.py` are byte-identical to upstream, so running the vendored
training entry point as-is trains a single-frame model, not this one.

## Model Assets

Place the released checkpoint under `checkpoints/<ckpt_name>/`, or set `model_dir` in `deploy.yml` to an existing checkpoint directory. The directory must contain `config.py` and `last.ckpt/`; `config.py` carries the history settings used at inference.

```text
policy/Rex_M1_preview/checkpoints/Rex_M1_preview/
├── config.py
└── last.ckpt/
```

See `INSTALLATION.md` for the complete checkpoint preparation and verification commands.

## Evaluation

```bash
cd XPolicyLab/policy/Rex_M1_preview
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate the released checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls Rex_M1_preview arx_x5 ee 0 0 0 \
  <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check; leave it unset or set it to `sim` for RoboDojo simulation. For split-machine deployment, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

Policy-specific `deploy.yml` keys worth checking before evaluation:

| Key | Notes |
|---|---|
| `action_type` | Must be `ee`. |
| `model_dir` | Optional checkpoint directory; takes precedence over `ckpt_name`. |
| `action_length` | Number of leading actions executed before replanning; `0` executes the full predicted chunk. |
| `vlm_processor_path` | Hugging Face repository ID or local path for the Qwen3-VL processor. |
| `image_factor` | Camera frames are resized to a multiple of this before tokenization. |
| `image_max_pixels` | Upper bound on pixels per current-view frame after resizing. |
| `default_prompt` | Instruction used when the environment supplies none. |

History parameters are read from the checkpoint's `config.py` so training and inference settings remain aligned.

## Acknowledgements

Rex-M1-preview builds on the [codebase](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1) and
[pretrained weights](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B) released by the
Xiaomi Robotics team. We sincerely thank the team for making these resources publicly available and
for their valuable contributions to the community.

## Notes

- **One policy server per evaluation.** This adapter keeps a per-environment history buffer keyed on
  `obs["env_idx"]` alone, so two concurrent evaluations pointed at the same server would interleave
  their histories without any error being raised.
