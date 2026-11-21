# MotuBrain2.0-Robodojo

**Contributor:** Motubrain Team

Evaluation-only adapter for `RoboDojo / arx_x5 / joint`. It loads the downloaded
checkpoint on the evaluator's GPU. The package includes the required `ss-falcon`,
`ss-soma`, `ss-morph` and `ss-modelzoo` inference source under `source/`.
There is no hosted remote-model dependency.

## Installation

For a standalone checkout set `MOTUBRAIN_SOURCE_ROOT` to its absolute path;
this package includes that source under `source/`.

```bash
cd XPolicyLab/policy/MotuBrain2_0
source_root="${MOTUBRAIN_SOURCE_ROOT:-$PWD/source}"
(cd "$source_root" && conda env create -f environment-policy.yml)
conda activate motubrain-robodojo
bash install.sh
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$source_root/third_party/Depth-Anything-3"
git -C "$source_root/third_party/Depth-Anything-3" checkout --detach 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
export PYTHONPATH="$source_root/third_party/Depth-Anything-3/src${PYTHONPATH:+:$PYTHONPATH}"
```

Follow the source README to download the VAE, T5Gemma and DA3 weights. Export
`SOMA_VAE_PATH`, `SOMA_TEXT_ENCODER_PATH`, `DA3_CONFIG_PATH` and
`DA3_CHECKPOINT_PATH` as shown there. Use a separate environment for the
RoboDojo simulator. The policy environment uses Python 3.11.

## Data Processing

This is an eval-only contribution. Dataset conversion is not included, so
`process_data.sh` is intentionally absent.

## Training

Training and optimizer states are not included, so `train.sh` is intentionally
absent. Training release ETA: not announced.

## Evaluation

```bash
bash eval.sh RoboDojo <task_name> <checkpoint_bundle> \
  arx_x5 joint <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <simulator_env>

# Example, after downloading the bundle below and installing the simulator
bash eval.sh RoboDojo stack_bowls /path/to/checkpoints/motubrain \
  arx_x5 joint 0 0 1 motubrain-robodojo <robodojo-simulator-env>
```

Pass the checkpoint **bundle directory** as `ckpt_name`; the shared resolver
also supports conventional `checkpoints/<bench>-<ckpt>-<robot>-<action>-<seed>/`
directories. Keep `EVAL_ENV_TYPE=sim` or unset for simulator evaluation.
`EVAL_ENV_TYPE=debug` also loads the configured policy and requires its weights.

`deploy.py` reuses the demo episode loop, which already captures every native
observation during the action chunk; our wrapper only adds `observation_step`.
Soma then samples the episode anchor and eight past frames at 16-step intervals.
`model.py` receives RGB arrays directly from the server; it never decodes or swaps
channels. Robot dimensions, state packing and action unpacking use shared helpers.
Run one environment per model process with `eval_batch=false`.

## Switching compatible checkpoints

Download another compatible checkpoint with `hf download`, then update
`checkpoint_file`, `model_config_file` and `action_stats_file` in `deploy.yml`.
These paths are relative to the bundle selected by `ckpt_name`, or absolute.
`policy_config` optionally selects a Python policy config for inference steps,
solver, compile settings and preprocessing; relative paths use this adapter's
directory. The horizon comes from the model configuration and normalization
uses the selected statistics. `query_seed_base: null` disables fixed inference
seeds; the default `42` preserves the published query sequence.

This supports RoboDojo / `arx_x5` / joint14 with `causal_9_to_3`. See the source
README for the full procedure. Different robot protocols or history encodings
require a separate port.

## Model Assets

The published checkpoint uses normal BF16 weights with four CM denoising steps. The download script pins
public Hugging Face revision `26c53046384721b21ef2b6d39b47b339cbc4bba8`.

The repository contains only three inference assets at its root:

- `step-5000.safetensors` (39.05 GB).
- `config.yaml` (inference architecture and stride-16 sampling metadata).
- `action_stats.json` (matching action normalization statistics).

```bash
bash download_checkpoint.sh /path/to/checkpoints/motubrain
```

Pass that directory as the checkpoint bundle to `eval.sh`.

[Checkpoint](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/step-5000.safetensors), [configuration](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/config.yaml), [action statistics](https://huggingface.co/Motubrain/MotuBrain2.0-Robodojo/resolve/26c53046384721b21ef2b6d39b47b339cbc4bba8/action_stats.json).

Downloads use the HF CLI directly. Frozen encoders remain separate upstream
assets. The embodied Chinese prompt, action mask, 32-action horizon and four-step
CM solver are preserved; the selected checkpoint uses history stride 16.
