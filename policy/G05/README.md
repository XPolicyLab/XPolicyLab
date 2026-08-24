# G05 RoboDojo Policy Adapter

**Contributor:** OpenGalaxea | **Paper:** Not applicable | **arXiv:** Not applicable | **Original code:** See vendored `G05/`.

This adapter serves G05 checkpoints through the XPolicyLab websocket policy-server interface. Integration scripts live at this directory level; the vendored public G05 implementation lives in `G05/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, and `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Install the XPolicyLab-side adapter dependencies:

```bash
cd XPolicyLab/policy/G05
export G05_PYTHON=/path/to/python
bash install.sh
```

The vendored G05 checkout is used by default. Set `G05_ROOT` only when using another compatible checkout:

```bash
export G05_PYTHON=/path/to/python
# optional:
export G05_ROOT=/path/to/another/G05_checkout
```

`G05_PYTHON` must point to an environment that can import the vendored G05 checkout and its dependencies. Checkpoint archives do not include a Python runtime.

## Model Assets

Model files are hosted in:

```text
https://huggingface.co/OpenGalaxea/g05-robodojo
```

### RoboDojo simulation checkpoint

Use this archive for RoboDojo simulator evaluation:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_fm_only_checkpoint.tar \
  g05_robodojo_fm_only_checkpoint.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_sim

cd ./checkpoints/g05_robodojo_sim
sha256sum -c g05_robodojo_fm_only_checkpoint.tar.sha256
tar -xf g05_robodojo_fm_only_checkpoint.tar
```

The extracted checkpoint file is:

```text
hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint
```

Then set:

```bash
export G05_CKPT_PATH=/path/to/hf_g05_robodojo_fm_only_checkpoint/checkpoints/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm
```

The public archive name intentionally does not encode the training step.

### RoboDojo-real checkpoints

Real-policy bundles are robot-specific and must not be interchanged. Separate
ARX-X5, Piper, and Piper-X bundles are available for official testing. Download
only the bundle that matches the evaluation robot:

```bash
modelscope download --model ZhyRobert/g05-robodojo-real \
  --include 'real/<robot>/**' \
  --local_dir ./checkpoints/g05_robodojo_real
```

Replace `<robot>` with `arx_x5`, `piper`, or `piper_x`. Each bundle is
self-contained for evaluation and has this layout:

```text
real/<robot>/checkpoints/checkpoint.pt
real/<robot>/config.yaml
real/<robot>/dataset_stats.json
real/<robot>/action_tokenizer_hf/
real/<robot>/input_processor/
real/<robot>/inference_runtime/
```

Select the matching deployment configuration:

```text
arx_x5 -> deploy_real_arx_x5.yml -> robodojo_arx_x5
piper  -> deploy_real_piper.yml  -> robodojo_piper
piper_x -> deploy_real_piper_x.yml -> robodojo_piper_x
```

Install the selected runtime and start the adapter with matching paths. For
example, for Piper:

```bash
export G05_CKPT_PATH=/path/to/checkpoints/g05_robodojo_real/real/piper/checkpoints/checkpoint.pt
export G05_DEPLOY_CONFIG=deploy_real_piper.yml
export G05_ROOT=/path/to/checkpoints/g05_robodojo_real/real/piper/inference_runtime
export G05_PYTHON=/path/to/python3.10
export ROBODOJO_G05_ACTION_SOURCE=fm

cd XPolicyLab/policy/G05
bash install.sh
```

All three real configurations use bilateral 14-dimensional absolute
joint-position actions, 25 Hz control, a 32-step predicted horizon, and 16
returned actions per request. They are AR+FM checkpoints served through the
continuous FM action head. Offline evaluation found lower continuous-action
error from FM than AR, so FM is the released default. The default `deploy.yml`
remains the RoboDojo-sim configuration and is unchanged.

This release does not include RoboDojo-real training code or data-processing scripts. The real checkpoint is provided for official RoboDojo-real evaluation with this adapter.

### Training base assets

Training base assets are separate from evaluation checkpoints:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_train_base_assets.tar \
  g05_robodojo_train_base_assets.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_train_assets

cd ./checkpoints/g05_robodojo_train_assets
sha256sum -c g05_robodojo_train_base_assets.tar.sha256
tar -xf g05_robodojo_train_base_assets.tar
```

The extracted directory contains:

```text
g05_robodojo_train_base_assets/pretrained/checkpoints/model_state_dict.pt
g05_robodojo_train_base_assets/pretrained/hf_processor/
g05_robodojo_train_base_assets/pretrained/action_tokenizer_hf/
```

Point the selected G05 training config to these assets using the config keys expected by that G05 checkout. RoboDojo datasets are not included in this archive.

## Data Processing

This adapter does not provide a standalone data converter. Use the official RoboDojo data download and conversion pipeline for simulator data. The policy reads the RoboDojo LeRobot v3.0 dataset directly.

## Training

`train.sh` is a compatibility launcher around the vendored G05 checkout. It supports RoboDojo simulator training with `action_type=joint` and mode selection through `G05_TRAIN_MODE`.

Simulation training example:

```bash
export G05_PYTHON=/path/to/python
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_lerobot_v30_video
export G05_TRAIN_MODE=fm_only
export G05_MAX_STEPS=<num_training_steps>

cd XPolicyLab/policy/G05
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7 \
  <g05_training_overrides>
```

Supported `G05_TRAIN_MODE` values are `fm_only`, `ar_only`, and `ar_fm`. Training length is intentionally user-defined: set `G05_MAX_STEPS`, `G05_MAX_EPOCHS`, or pass `model.max_steps=...` / `model.max_epochs=...` as a Hydra override. Trailing arguments are passed through to the G05 training script.

Training length is not fixed by this adapter. For RoboDojo simulator training, users can start from a moderate number of optimizer steps and extend training if validation or simulator evaluation metrics continue to improve. The released checkpoint is intended as a reproducible evaluation baseline; further tuning of training length, data mixture, and model configuration may improve performance.

RoboDojo-real support in this release is evaluation-only. Use the released real checkpoint for official RoboDojo-real evaluation; real-world training code and data-processing scripts are not included here.

## Evaluation

Offline adapter check:

```bash
cd XPolicyLab/policy/G05
export EVAL_ENV_TYPE=debug
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=/path/to/extracted/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm

bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" base
```

Simulator-backed evaluation uses the same adapter entrypoint with `EVAL_ENV_TYPE` unset or set to `sim`:

```bash
cd XPolicyLab/policy/G05
unset EVAL_ENV_TYPE
export G05_PYTHON=/path/to/python
export G05_CKPT_PATH=/path/to/extracted/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm

bash eval.sh RoboDojo stack_bowls checkpoint arx_x5 joint 0 0 0 \
  "$G05_PYTHON" sim
```

For split-machine deployment via `setup_eval_policy_server.sh` and `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Notes

- Use `ROBODOJO_G05_ACTION_SOURCE=fm` for FM-style continuous action inference.
- Use `ROBODOJO_G05_ACTION_SOURCE=ar` only with checkpoints and G05 runtimes trained and validated for AR action decoding.
- For simulator evaluation, run from a RoboDojo checkout where evaluator-side `env_cfg/`, `scripts/`, `src/`, `task/`, and `Assets/` are available next to `XPolicyLab`.
