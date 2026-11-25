# MotuBrain2.0

**Contributor:** Motubrain Team

Evaluation-only adapter for `RoboDojo / arx_x5 / joint` and `RoboTwin / aloha_agilex / ee`. It loads the downloaded
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

Follow the [frozen encoder download instructions](source/README.md#download-weights) for the VAE, T5Gemma and DA3 weights. Export
`SOMA_VAE_PATH`, `SOMA_TEXT_ENCODER_PATH`, `DA3_CONFIG_PATH` and
`DA3_CHECKPOINT_PATH` as shown there. Use a separate environment for the
selected benchmark's simulator. The policy environment uses Python 3.11.

## Data Processing

This is an eval-only contribution. Dataset conversion is not included, so
`process_data.sh` is intentionally absent.

## Training

Training and optimizer states are not included, so `train.sh` is intentionally
absent. Training release ETA: not announced.

## RoboDojo evaluation

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
For RoboDojo, Soma samples the episode anchor and eight past frames at 16-step intervals.
`model.py` receives RGB arrays directly from the server; it never decodes or swaps
channels. Robot dimensions, state packing and action unpacking use shared helpers.
Run one environment per model process with `eval_batch=false`.

## RoboTwin 2.0

The [RoboTwin model card](https://huggingface.co/GensPI/Motubrain2.0-robotwin2.0) documents the EEF student. Its [DMD-1000 bundle](https://huggingface.co/GensPI/Motubrain2.0-robotwin2.0/tree/main/weights/dmd-step1000) contains three files:

- [Policy weights](https://huggingface.co/GensPI/Motubrain2.0-robotwin2.0/resolve/main/weights/dmd-step1000/ckpts/step-1000.safetensors)
- [Model configuration](https://huggingface.co/GensPI/Motubrain2.0-robotwin2.0/blob/main/weights/dmd-step1000/config.yaml)
- [Action normalization statistics](https://huggingface.co/GensPI/Motubrain2.0-robotwin2.0/blob/main/weights/dmd-step1000/rw_n2_q1.json)

Download all three files and pass the directory containing `config.yaml` and `rw_n2_q1.json` to the evaluator:

```bash
hf download GensPI/Motubrain2.0-robotwin2.0 \
    weights/dmd-step1000/ckpts/step-1000.safetensors weights/dmd-step1000/config.yaml \
    weights/dmd-step1000/rw_n2_q1.json \
    --local-dir /path/to/checkpoints/robotwin
bash eval.sh RoboTwin adjust_bottle /path/to/checkpoints/robotwin/weights/dmd-step1000 \
    aloha_agilex ee 0 0 1 motubrain-robodojo <robotwin-simulator-env>
```

The RoboTwin launcher selects [deploy_robotwin.yml](deploy_robotwin.yml). Each query encodes the latest RGB frame from each camera with the FP32 video VAE, passes the task instruction directly to the text encoder, and produces 24 actions. The policy receives absolute
end-effector poses and the head camera's OpenCV extrinsics. It converts tool
axes and quaternion order on input and output. The selected statistics and
current poses define the camera-relative action conversion. Diffusion noise is
unseeded by default.

## Switching compatible checkpoints

Pass the new bundle directory as `ckpt_name`. If its filenames differ, update
`checkpoint_file`, `model_config_file` and `action_stats_file` in
[deploy.yml](deploy.yml) for RoboDojo or [deploy_robotwin.yml](deploy_robotwin.yml)
for RoboTwin. Relative asset paths start at `ckpt_name`.

Leave `policy_config: null` for the published checkpoints. This selects the
built-in inference settings for the chosen benchmark. The downloaded
`config.yaml` goes in `model_config_file`.

To supply custom inference settings, set `policy_config` to a Python file
defining `policy` and `inference`. Use an absolute path or a path relative to
`policy/MotuBrain2_0/`. A custom file replaces the benchmark defaults. For
RoboTwin it must provide the camera-relative EEF action profile, normalization
settings and single-frame video encoding. The loader still reads the selected
checkpoint's model YAML and action statistics.

## RoboDojo Model Assets

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
