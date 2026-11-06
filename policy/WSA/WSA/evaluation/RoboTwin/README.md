# RoboTwin Evaluation

This document provides instructions for reproducing our experimental results
with [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin).

The `eval_wsa_base.sh` and `eval_wsa_large.sh` entrypoints run WSA on the
RoboTwin randomized 50-task benchmark. By default they load the released
`zaleni/WSA-Base-RoboTwin` and `zaleni/WSA-Large-RoboTwin` checkpoints,
respectively. You can override the checkpoint, task range, episode count, and
GPU allocation with environment variables.

## Introduction

- `eval_wsa_base.sh`: maintained WSA-Base randomized 50-task evaluation
  wrapper. It defaults to `PRETRAINED_CKPT=zaleni/WSA-Base-RoboTwin`.
- `eval_wsa_large.sh`: WSA-Large delta-action randomized 50-task evaluation
  wrapper. It defaults to `PRETRAINED_CKPT=zaleni/WSA-Large-RoboTwin`.
- `inference.py`: shared RoboTwin evaluator called by the shell wrapper.
- `eval_config.py`: resolves per-task `infer_horizon` and
  `binarize_gripper` from `configs/robotwin_eval_config.yaml`.
- `image_tools.py`: image processing helpers used by the evaluator.

## Requirements

You can follow the official [RoboTwin installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) guide for the complete
simulator environment setup.

This repository expects the RoboTwin codebase to live under
`third_party/RoboTwin`. If you cloned WSA without submodules, initialize the
submodule first:

```bash
git submodule update --init --recursive third_party/RoboTwin
```

If you install RoboTwin manually instead of using submodules, clone it into the
same path:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
```

### Environment Set-up

After the main WSA environment is ready, keep using the same environment
and add the RoboTwin2.0 dependencies.

```bash
conda activate wsa

sudo apt update
sudo apt install -y libvulkan1 mesa-vulkan-drivers vulkan-tools
vulkaninfo
```

Before running RoboTwin's installer, replace RoboTwin's default requirements
with the WSA-compatible extra requirements in this directory. The installer
will run `pip install -r script/requirements.txt`, so do not install the same
file separately unless you are skipping `_install.sh`.

```bash
cp evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
```


Then install RoboTwin's remaining native/simulator dependencies and download the assets:

```bash
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../..
```


For WSA-Base, install `transformers==4.57.1` and patch the installed Qwen3-VL
code with `src/lerobot/policies/WSA_Base/transformers_replace/models`.
```bash
TRANSFORMERS_DIR=${CONDA_PREFIX}/lib/python3.10/site-packages/transformers/
cp -r src/lerobot/policies/WSA_Base/transformers_replace/models ${TRANSFORMERS_DIR}
```

## WSA-Base Quick Start

```bash
PRETRAINED_CKPT=zaleni/WSA-Base-RoboTwin \
QWEN3_VL_PRETRAINED_PATH=Qwen/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=Qwen/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=nvidia/Cosmos-Tokenizer-CI8x8 \
DISABLE_DA3_TEACHER_FOR_EVAL=true \
ACTION_MODE=delta \
GPU_IDS=0,1 \
MAX_JOBS_PER_GPU=2 \
bash evaluation/RoboTwin/eval_wsa_base.sh
```

## WSA-Large Quick Start

Start with one task and a small episode count to verify the environment and
model assets before launching the full benchmark:

```bash
PRETRAINED_CKPT=zaleni/WSA-Large-RoboTwin \
GPU_IDS=0 \
START_TASK_IDX=0 \
TASK_COUNT=1 \
TEST_NUM=10 \
bash evaluation/RoboTwin/eval_wsa_large.sh
```

The released WSA-Large RoboTwin checkpoint uses delta actions. Its evaluation
wrapper defaults to `WSA_LARGE_LOAD_TEXT_ENCODER=true`, so plain task
instructions work without a precomputed text cache. Set `TASK_COUNT=50` and
increase `GPU_IDS` only after the smoke test succeeds.

## Common Options

- `PRETRAINED_CKPT`: checkpoint directory or Hugging Face model id. Defaults to
  the task-specific released checkpoint selected by the wrapper.
- `GPU_IDS`: comma-separated GPUs used by the scheduler, for example `0,1`.
- `MAX_JOBS_PER_GPU`: maximum parallel RoboTwin tasks per GPU. Lower this if
  memory is tight.
- `TASK_CONFIG`: RoboTwin task config setting `demo_clean` or `demo_randomized`. The default is set to
  `demo_randomized`.
- `ROBOTWIN_EVAL_CONFIG`: per-task eval setting file. By default this loads
  `configs/robotwin_eval_config.yaml` and applies each task's `infer_horizon`
  and `binarize_gripper`. Task keys must exactly match `inference.py`
  `TASK_NAMES`. Set `ROBOTWIN_EVAL_CONFIG=` to disable this.
- `START_TASK_IDX` and `TASK_COUNT`: evaluate a slice of the 50
  tasks, useful for debugging or resuming partial runs.
- `TEST_NUM`: number of episodes per task. Defaults to `100`.
- `ACTION_MODE`: action representation expected by the checkpoint. The released
  RoboTwin checkpoint uses `delta`.
- `INFER_HORIZON` and `ACTION_HORIZON_SIZE`: policy rollout horizon settings.
  `INFER_HORIZON` is used as the fallback when the per-task eval config does
  not provide an override.
- `DISABLE_DA3_TEACHER_FOR_EVAL`: keep this `true` for standard action
  evaluation without loading the 3D teacher.
- `QWEN3_VL_PRETRAINED_PATH`, `QWEN3_VL_PROCESSOR_PATH`, and
  `COSMOS_TOKENIZER_PATH_OR_NAME`: WSA-Base-only overrides; use them only when
  loading local copies of those assets.
- `WSA_LARGE_LOAD_TEXT_ENCODER`: WSA-Large-only option. Keep it `true` for the
  simplest evaluation path, or set it to `false` when the exact task prompts
  have been cached with `tools/precompute_text_embeds.py`.

## Outputs

Each run writes per-task logs under `evaluation/RoboTwin/output*/tasks/task_##/` with:

- `summary.json`
- `summary.txt`
- `job_status.tsv`

## Notes

- `ACTION_MODE` should match the checkpoint/training setup, `abs` or `delta`.
- `STATS_KEY` is usually `aloha`.
