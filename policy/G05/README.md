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

### RoboDojo-real checkpoint

Use this archive for RoboDojo-real evaluation only:

```bash
huggingface-cli download OpenGalaxea/g05-robodojo \
  g05_robodojo_real_checkpoint.tar \
  g05_robodojo_real_checkpoint.tar.sha256 \
  --local-dir ./checkpoints/g05_robodojo_real

cd ./checkpoints/g05_robodojo_real
sha256sum -c g05_robodojo_real_checkpoint.tar.sha256
tar -xf g05_robodojo_real_checkpoint.tar
```

The extracted checkpoint file is:

```text
g05_robodojo_real_checkpoint/checkpoint/checkpoints/checkpoint.pt
```

Then set:

```bash
export G05_CKPT_PATH=/path/to/g05_robodojo_real_checkpoint/checkpoint/checkpoints/checkpoint.pt
export ROBODOJO_G05_ACTION_SOURCE=fm
```

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

This adapter does not provide a standalone data converter. Use the official RoboDojo data download and conversion pipeline for simulator data. The policy reads the RoboDojo **LeRobot v3.0** dataset directly, from the root given by `ROBODOJO_LEROBOT_V30_ROOT`.

The keys are the standard ones of `XPolicyLab/scripts/transform_lerobot_v30_format.py` ([Official LeRobot conversion](../../README.md#official-lerobot-conversion)) — run that script to build a dataset for your own task subset. `G05/configs/data/robodojo.yaml` names them explicitly as `lerobot_key` entries: `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist`, plus the flat `observation.state` and `action` vectors sliced by `start_index`. Nothing beyond the official output is required — no `meta/modality.json`, and normalization stats are computed during training.

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

## WUJI 58-D hybrid (tianji_wuji)

The adapter also serves the WUJI dual-arm dexterous-hand platform (`env_cfg_type=tianji_wuji`, two 7-DOF arms with 20-DOF Wuji hands) in the WUJI SimBench `wrist58` hybrid layout, selected by `action_type=ee`. The 58-D state/action vector is, per side, 20 absolute hand joints followed by a 9-D `xyz + rot6d` palm pose; state wrists are absolute in the fixed TE2 head-camera frame and action wrists are camera-frame deltas against the live wrist. The layout and every convention live in `XPolicyLab/utils/wuji58.py`; the adapter never spells them out.

```text
[ 0:20]  left_hand    [20:29]  left_wrist    [29:49]  right_hand    [49:58]  right_wrist
```

G05 sees these four blocks as its state/action keys (`left_hand`, `left_wrist`, `right_hand`, `right_wrist`), configured by `G05/configs/task/robodojo_tianji_wuji_ee.yaml` (`action_dim: 58`, `proprio_dim: 58`, q01/q99 normalization, `GroupedPaddingMerger` over `configs/data/parts_meta/wuji58.yaml`), `configs/data/robodojo_wuji58.yaml` (embodiment `robodojo_wuji58`, slicing the flat 58-D LeRobot columns) and `configs/tokenizer/actioncodec_wuji58.yaml` (`max_component_dim: 20`). The arx_x5 joint path is untouched.

Data — export the tianji_wuji trajectories with the official converter in the hybrid representation; `train.sh` refuses an export whose `meta/info.json` is not stamped `repr_mode: wrist_rel_te2` with 58-D columns:

```bash
python XPolicyLab/scripts/transform_lerobot_v30_format.py --representation wuji58 \
  <converter arguments>   # see Official LeRobot conversion in the XPolicyLab README
export ROBODOJO_LEROBOT_V30_ROOT=/path/to/RoboDojo_tianji_wuji_lerobot_v30_wuji58
```

Training — `train.sh` picks `G05_TASK_CONFIG=robodojo_tianji_wuji_ee` automatically for `tianji_wuji ee`:

```bash
export G05_PYTHON=/path/to/python
export G05_TRAIN_MODE=fm_only
export G05_MAX_STEPS=<num_training_steps>
bash train.sh RoboDojo wuji tianji_wuji ee 0 0,1,2,3,4,5,6,7
```

Use `fm_only` unless you also train an ActionCodec tokenizer with `actioncodec_wuji58.yaml`: the released `action_tokenizer.pt` was trained with `max_component_dim: 9` and cannot tokenize the 20-wide hand blocks, so `ar_only` / `ar_fm` need a matching tokenizer checkpoint.

Evaluation — the standard entrypoint with `tianji_wuji ee`; `eval_embodiment` resolves to `robodojo_wuji58` when left `null` in `deploy.yml`:

```bash
export G05_CKPT_PATH=/path/to/wuji58/checkpoint
export ROBODOJO_G05_ACTION_SOURCE=fm
bash eval.sh RoboDojo <task_name> wuji tianji_wuji ee 0 0 0 "$G05_PYTHON" sim
```

`deploy.yml` keys for this mode: `wrist_frame` (`te2`, the WUJI SimBench head camera the checkpoints are trained in, or `world` for data recorded in world coordinates) and `eval_embodiment`. The robot must be registered as `tianji_wuji` (`arm_dim: [7, 7]`, `ee_dim: [20, 20]`) in both robot-info files (see the XPolicyLab README).

## Notes

- Use `ROBODOJO_G05_ACTION_SOURCE=fm` for FM-style continuous action inference.
- Use `ROBODOJO_G05_ACTION_SOURCE=ar` only with checkpoints and G05 runtimes trained and validated for AR action decoding.
- For simulator evaluation, run from a RoboDojo checkout where evaluator-side `env_cfg/`, `scripts/`, `src/`, `task/`, and `Assets/` are available next to `XPolicyLab`.
