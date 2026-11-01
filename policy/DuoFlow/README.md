# DuoFlow

**Contributor:** XJTU IAIR & WisdomStride | **Paper:** [Technical report](docs/DuoFlow_Technical_Report.pdf) | **arXiv:** Not published | **Original code:** [DuoFlow adapter source](.)

DuoFlow is a language-conditioned bimanual manipulation policy. This contribution provides the XPolicyLab integration, inference and evaluation code, checkpoint download script, and technical report for official reproduction and evaluation.

Supported configuration: bench_name=RoboTwin, env_cfg_type=aloha_agilex, action_type=joint, ckpt_name=duoflow.

Training support: eval-only. Training code is not included in this contribution.

Contact: zhengyj@stu.xjtu.edu.cn.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Place this directory at `XPolicyLab/policy/DuoFlow/` in a RoboTwin workspace. Use a separate Python 3.11 CUDA environment for the policy; the RoboTwin simulator keeps its own environment.

```bash
cd XPolicyLab/policy/DuoFlow
conda create -n duoflow python=3.11 -y
conda activate duoflow
bash install.sh
python download_checkpoints.py
```

`install.sh` installs PyTorch 2.7.1 with CUDA 12.6 and the pinned policy dependencies. It does not install RoboTwin or create the parent workspace's robot registry. The registered `aloha_agilex` robot must expose two six-joint arms and one gripper value per arm. Real-model inference requires an available CUDA device and the downloaded assets.

## Data Processing

Unsupported in this eval-only contribution. `process_data.sh` is intentionally absent; no training trajectory conversion is performed. The adapter receives three decoded HWC uint8 RGB images and the 14-dimensional joint state through XPolicyLab. Image decoding belongs to the shared policy server; the adapter performs no RGB/BGR channel swap.

## Training

Training is not supported in this contribution.
Neither `train.sh` nor the private training implementation is included in this package.
Training consists of an initial shared fine-tuning stage with 5,000 parameter updates, followed by 5,000 DuoFlow-specific updates.
The additional demonstrations are drawn from a subset of the official Clean50 training pool. Therefore, this adaptation does not introduce any new demonstrations or Randomized training data.

## Evaluation

Use the shared positional argument convention:

```bash
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>
```

For example, after installing the policy environment `duoflow`, the simulator environment `robotwin_env`, and registering `aloha_agilex` in the parent workspace:

```bash
EVAL_ENV_TYPE=debug bash eval.sh RoboTwin lift_pot duoflow aloha_agilex joint 0 0 0 duoflow robotwin_env
DEBUG_OBS_ENCODED=1 EVAL_ENV_TYPE=debug bash eval.sh RoboTwin lift_pot duoflow aloha_agilex joint 0 0 0 duoflow robotwin_env
EVAL_ENV_TYPE=sim bash eval.sh RoboTwin lift_pot duoflow aloha_agilex joint 0 0 0 duoflow robotwin_env
```

The device indices above are examples; select permitted devices in your own environment. Debug still loads the real model, so it requires CUDA and weights even though it does not run a simulator. Follow the shared README for split-machine deployment and the RoboTwin workspace configuration for the evaluation condition.

## Model Assets

The [published adapter](https://huggingface.co/Sva1in/DuoFlow) is pinned to revision `889544d8c10ab3c7cfba7f8a7bba35b405a16149`. `download_checkpoints.py` obtains it together with the base weights, normalization sidecars, and tokenizers, using the repositories and revisions in `ASSET_SOURCES.json`. The base model is approximately 14.1 GB; weights are not stored in this source contribution.

The default adapter location is `checkpoints/duoflow/duoflow_adapter.safetensors`; use `ckpt_name=duoflow`. This is the shared checkpoint resolver's documented name fallback, rather than the five-field training-directory name. Explicit checkpoint paths are also resolved by XPolicyLab.

```bash
# Optional shared asset location; keep the same value for policy-server execution.
export DUOFLOW_ASSETS=/absolute/duoflow_assets
python download_checkpoints.py --assets-dir "$DUOFLOW_ASSETS"
# Download only the adapter if the base assets are already present.
python download_checkpoints.py --adapter-only
```

## Configuration

`deploy.yml` uses `protocol: ws`, `policy_name: DuoFlow`, and `eval_batch: false`. Only `action_type=joint` with Aloha-AgileX is supported by this checkpoint. The `camera_map` maps native `cam_high` to XPolicyLab `cam_head`, and maps the two wrist cameras by name. The model predicts 50 joint actions with 10 function evaluations and returns the first 30 for execution.

## Limitations / notes

The released checkpoint targets Aloha-AgileX joint control. This contribution provides inference and evaluation support; it does not include training entry points.

## License

DuoFlow code is Apache-2.0; retained upstream licenses and notices apply to vendored components and assets.
