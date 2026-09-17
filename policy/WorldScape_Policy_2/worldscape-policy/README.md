# WorldScape Policy 2.0: **Empowering Steerable World Action Modeling with Reasoning-Augmented Memory**

WorldScape Policy 2.0 is a controllable World Action Model (WAM) that introduces multimodal controllability and reasoning-augmented memory, enabling interactive robotic manipulation through **Long-Horizon Autonomous Planning**, **Fine-Grained Instruction Following**, and **In-Context Learning** (Visual Reasoning or Skill Imitation). This repository provides a natively pretrained model checkpoint, pretraining and post-training recipes, evaluation tutorials, and real-robot deployment tools.

[![Project Page](https://img.shields.io/badge/Project-Page-6F35C7?logo=googlechrome&logoColor=white)](https://manifoldai-research.github.io/WorldScape-Policy/)
[![Paper](https://img.shields.io/badge/arXiv-2607.18840-b31b1b.svg)](https://arxiv.org/abs/2607.18840)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-WorldScape--Policy--2-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/manifoldai-research/WorldScape-Policy-2)


## 📢 News

- **[2026-08-31]** 🚀 We released the pretrained and posttrained model checkpoints (e.g. RoboTwin2.0-C2R), pre-training and post-training recipes, and evaluation code for simulation benchmark and real-robot deployment.
- **[2026-07-20]** 🌐 Our [project page](https://manifoldai-research.github.io/WorldScape-Policy/) is now available.

## 📖 Overview

World Action Models jointly model future visual state transitions and robot actions, providing a natural interface for robot planning and controllable execution. However, existing WAMs are often limited by short temporal context and coarse episode-level language supervision. Besides, most existing WAMs rely primarily on text-only conditioning and lack support for video prompting. Consequently, they cannot readily acquire or imitate new behaviors from visual demonstrations provided at inference time, substantially limiting their capacity for example-driven in-context learning—analogous to how large language models learn from demonstrations in context.

WorldScape Policy 2.0 addresses this limitation by introducing multimodal controllability and reasoning-augmented memory. It supports text instructions, goal images, and demonstration videos as interchangeable conditioning signals, enabling fine-grained instruction following, visual skill imitation, and long-horizon autonomous planning within a unified world-action policy.

## ✨ Key Features

- **Reasoning-augmented memory:** Short-term visual memory incorporates recent observations through causal prefill to preserve local interaction dynamics, while long-term event memory organizes historical reasoning outputs into hierarchical representations for progress-aware retrieval.
- **Latent autonomous planning:** A vision-language model autoregressively generates latent planning tokens conditioned on the task instruction, current observation, and retrieved event history, enabling adaptive long-horizon decision-making.
- **Diverse interaction modes:** A unified world-action policy supports high-level task instructions, fine-grained language commands, goal-image prompts, and one-shot human demonstrations for visual in-context learning.
- **Event-grounded pretraining:** Temporally localized events are aligned with language descriptions, visual prompts, video demonstrations, and action trajectories, providing fine-grained supervision beyond conventional episode-level annotations.
- **Joint video-action modeling:** The WAM jointly predicts future three-view visual observations and robot actions under a diffusion-based training objective, grounding action generation in anticipated scene dynamics.
- **Multi-embodiment transfer:** Cross-dataset pretraining preserves embodiment-specific action adapters within a shared model, while post-training selects and exports a single adapter specialized for the target robot.

## 🧠 Autonomous Planning and Instruction Following

Text-conditioned training and inference support two exchangeable modes:

- **Auto:** autonomous planning. The high-level instruction and head-camera observation are wrapped by the VLM planning prompt. Qwen3-VL generates latent conditions (subgoal / planning tokens) for the WAM after event memory retrieval.
- **Interactive:** instruction following. The fine-grained current task or atomic subtask instruction is encoded directly by T5 without VLM planning, which is then adopted by the WAM.

Select the appropriate mode with `WSP_MODE=auto` or `WSP_MODE=interactive`. Training and
evaluation must use checkpoints with matching modes.

## 🎬 In-Context Learning and Adaptation

Demonstration-video conditioning supports two complementary workflows:

- **One-shot in-context learning:** The pretrained model can directly use a single human demonstration video as a visual prompt at inference time and execute the demonstrated behavior without task-specific parameter updates.
- **Video-prompt adaptation:** For a new task or domain, users can collect their own demonstration videos and post-train the pretrained checkpoint with `VISUAL_PROMPT=demo`. The adapted model retains the same demonstration-conditioned inference interface while specializing to the target task, robot embodiment, and data distribution.

See [Demonstration-video conditioning](#3-demonstration-video-conditioning) for post-training and [Real-robot evaluation and deployment](#-real-robot-evaluation-and-deployment) for inference examples.

## 🛠️ Installation

The validated environment uses Python 3.11, PyTorch 2.8.0, CUDA 12.9,
TorchVision 0.23.0, and DeepSpeed 0.18.9.

```bash
# Conda
conda env create -f environment.yml
conda activate worldscape-policy

# Or an existing Python 3.11/3.12 environment
python -m pip install -r requirements.txt
python -m pip install -e .
```

Recipes resolve Python in this order: `WSP_PYTHON`, `CONDA_ENV/bin/python`,
the active `$CONDA_PREFIX`, then `python` from `PATH`.

```bash
export CONDA_ENV="$CONDA_PREFIX"
```

FlashAttention 2.8.3 is optional; PyTorch SDPA is the fallback. RoboTwin2 and the real-robot `manifold_msg` SDK must be installed from their upstream distributions when needed.

## 📊 Data Preparation

WorldScape Policy can train from raw HDF5 episodes or existing LeRobot v2 datasets. Before training, generate the native metadata required by the dataset adapters:

- Raw HDF5: add native metadata without rewriting the episode files.
- LeRobot v2: scan the existing Parquet/video data and add or refresh metadata.
- Full conversion: optionally convert HDF5 episodes into LeRobot Parquet and
MP4 files.

See the [data preparation guide](tools/data/README.md) for expected schemas,
conversion commands, output layouts, and validation steps.

## 📦 Download models

All checkpoint paths below are relative to `manifoldai-research/worldscape-policy/`.

| Model              | Use Case                   | Description                                                                                       | Download&nbsp;Size | Checkpoint Path                 |
| ------------------ | -------------------------- | ------------------------------------------------------------------------------------------------- | ------------------ | ------------------------------- |
| WSP2-Pretrain      | Fine-Tuning / Mid-Training | Cross-embodiment pretrained checkpoint used to initialize downstream post-training / mid-training | 25.8&nbsp;GB       | `wsp_2_pretrain`                |
| WSP2-RoboTwin2-C2R | Inference                  | Post-trained RoboTwin 2.0 checkpoint on clean-only dataset for benchmark evaluation               | 35.4&nbsp;GB       | `wsp_2_posttrain_robotwin2_c2r` |


Download the pre-trained checkpoint for fine-tuning:

```bash
export WSP_MODEL_ROOT="$HOME/models/worldscape-policy"
mkdir -p "$WSP_MODEL_ROOT"

hf download manifoldai-research/WorldScape-Policy-2 \
  --include "wsp_2_pretrain/**" \
  --local-dir "$WSP_MODEL_ROOT"

export PRETRAINED_MODEL_PATH="$WSP_MODEL_ROOT/wsp_2_pretrain"
```

Download the RoboTwin 2.0 checkpoint for inference:

```bash
hf download manifoldai-research/WorldScape-Policy-2 \
  --include "wsp_2_posttrain_robotwin2_c2r/**" \
  --local-dir "$WSP_MODEL_ROOT"

export ROBOTWIN2_EVAL_MODEL_PATH="$WSP_MODEL_ROOT/wsp_2_posttrain_robotwin2_c2r"
```

The launchers place missing public base components below `WSP_MODEL_ROOT` and download them automatically. Set `WSP_AUTO_DOWNLOAD=false` for offline jobs, or override paths explicitly:

```bash
export WAN_CKPT_DIR="$WSP_MODEL_ROOT/Wan2.2-TI2V-5B"
export CLIP_CKPT_DIR="$WSP_MODEL_ROOT/Wan2.1-I2V-14B-480P"
export TOKENIZER_DIR="$WSP_MODEL_ROOT/umt5-xxl"
export Qwen_CKPT_DIR="$WSP_MODEL_ROOT/Qwen3-VL-4B-Instruct"
```

Training loads checkpoints in this order:

1. latest complete `checkpoint-N` in `OUTPUT_DIR`;
2. `PRETRAINED_MODEL_PATH`;
3. the base components above.

Every saved `checkpoint-N/` is both resumable and evaluable. Evaluation requires the complete native directory, including `model.safetensors.index.json` and all `model-00001-of-0000N.safetensors` shards (or an unsharded `model.safetensors`), `checkpoint_manifest.json`, `transform_bundle.json`, and `.complete`. New checkpoints use 5 GB Hugging Face shards by default; override this with `CHECKPOINT_MAX_SHARD_SIZE` (for example, `10GB`).

## 🚀 Real-Robot Fine-Tuning on Your Own Data

AgileX post-training consumes HDF5 episodes with synchronized head, left-wrist, and right-wrist images, EEF state/actions, and task/subtask metadata. Set the common inputs first:

```bash
export CONDA_ENV="$CONDA_PREFIX"
export PRETRAINED_MODEL_PATH="$WSP_MODEL_ROOT/wsp_2_pretrain"
export DATA_ROOT=/data/agilex-task
export NUM_GPUS=8
```

### 1. Text-Instruction Conditioning

Configuration:

- `VISUAL_PROMPT=none`
- `NATIVE_DATASET_NAME=worldscape_hdf5_text`
- `WSP_MODE=auto` for autonomous planning
- `WSP_MODE=interactive` for direct instruction following

See [Post-training: Text-instruction prompts](docs/posttraining.md#text-instruction-prompts)
for prompt templates, source-field parsing, and semantic-target handling.

```bash
# Autonomous planning
WSP_MODE=auto RUN_NAME=fold-shirt-auto \
  ./recipes/posttrain/posttrain_agilex_fold_shirt_text.sh

# Instruction following
WSP_MODE=interactive RUN_NAME=fold-shirt-interactive \
  ./recipes/posttrain/posttrain_agilex_fold_shirt_text.sh
```

### 2. Goal-Image Conditioning

Configuration:

- `VISUAL_PROMPT=goal`
- `NATIVE_DATASET_NAME=worldscape_hdf5_goal`
- default `WSP_MODE=interactive`

```bash
DATA_ROOT=/data/build-block-goal RUN_NAME=build-block-goal \
  ./recipes/posttrain/posttrain_agilex_build_block_goal.sh
```

### 3. Demonstration-Video Conditioning

Configuration for In-Context Adaptation:

- `VISUAL_PROMPT=demo`
- `NATIVE_DATASET_NAME=worldscape_hdf5_demo`
- default `WSP_MODE=interactive`

```bash
# Skill imitation
DATA_ROOT=/data/build-block-demo RUN_NAME=build-block-demo \
  ./recipes/posttrain/posttrain_agilex_build_block_demo.sh

# Visual Reasoning
DATA_ROOT=/data/shell-game-demo RUN_NAME=shell-game-demo \
  ./recipes/posttrain/posttrain_agilex_shell_game_demo.sh
```

Use `CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1` for single-GPU debugging. For multiple nodes, set `MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, `MLP_WORKER_0_HOST`, `MLP_WORKER_0_PORT`, and the per-node `NUM_GPUS`.

## 🤖 Real-Robot Evaluation and Deployment

Always test with read-only HDF5 replay before enabling robot actuation:

```bash
export CONDA_ENV="$CONDA_PREFIX"
export FOLD_SHIRT_TEXT_EVAL_MODEL_PATH=/path/to/checkpoint
export WORLDSCAPE_HDF5_EPISODE=/data/replay/episode.hdf5
export WSP_MODE=auto
export WSP_INSTRUCTION="Fold the T-shirt into a rectangle."

AGILEX_TRANSPORT=hdf5 \
  ./recipes/eval/eval_agilex_fold_shirt_text.sh
```

Live deployment:

```bash
export AGILEX_TRANSPORT=manifold
export WSP_SERVER_HOST=0.0.0.0
export WSP_SERVER_PORT=11451
export WSP_NODE_NAME=WSP

./recipes/eval/eval_agilex_fold_shirt_text.sh
```

The launcher enables `--live-hardware` only for `AGILEX_TRANSPORT=manifold`. Install the robot-side `manifold_msg` SDK and verify joint limits, emergency stop, workspace clearance, and network isolation before deployment.

Other interaction types use their task checkpoint and launcher. The following
commands are live-hardware examples and require the same safety checks:

```bash
# Goal-Conditioned Policy: Act to goal
AGILEX_TRANSPORT=manifold \
BUILD_BLOCK_GOAL_EVAL_MODEL_PATH=/path/to/checkpoint \
WSP_GOAL_IMAGE=/data/prompts/goal.png \
./recipes/eval/eval_agilex_build_block_goal.sh

# In-Context Learning: Skill imitation tasks
AGILEX_TRANSPORT=manifold \
BUILD_BLOCK_DEMO_EVAL_MODEL_PATH=/path/to/checkpoint \
./recipes/eval/eval_agilex_build_block_demo.sh

# In-Context Learning: Visual reasoning tasks
AGILEX_TRANSPORT=manifold \
SHELL_GAME_DEMO_EVAL_MODEL_PATH=/path/to/checkpoint \
./recipes/eval/eval_agilex_shell_game_demo.sh
```



## 🪿 RoboTwin 2.0 Evaluation

RoboTwin 2.0 evaluation follows the official evaluation interface and workflow. For convenience, a copy of the official RoboTwin repository is included under `third_party/RoboTwin`, with several local modifications applied to accelerate evaluation. Before running evaluation, follow the official instructions in `third_party/RoboTwin/README.md` to install the environment and download the required assets. The manager creates `third_party/RoboTwin/policy/wsp2_policy` automatically; an equivalent manual setup is:

```bash
ln -s "$(pwd)/experiments/robotwin/wsp2_policy" \
  third_party/RoboTwin/policy/wsp2_policy
```

Clean-only training:

```bash
DATA_ROOT=/data/robotwin2-clean \
ZSCORE_STATS_PATH=/data/robotwin2-clean/dataset_stats.json \
PRETRAINED_MODEL_PATH="$WSP_MODEL_ROOT/wsp_2_pretrain" \
NUM_GPUS=8 ./recipes/posttrain/posttrain_robotwin2.sh
```

For the full dataset:

```bash
DATA_ROOT=/data/robotwin2-full \
ZSCORE_STATS_PATH=/data/robotwin2-full/dataset_stats.json \
PRETRAINED_MODEL_PATH="$WSP_MODEL_ROOT/wsp_2_pretrain" \
NUM_GPUS=8 ./recipes/posttrain/posttrain_robotwin2_full.sh
```

Evaluation uses 14-D absolute joint actions, 24-step chunks, nine
rolling observation frames, and one persistent model per GPU worker:

```bash
ROBOTWIN2_EVAL_MODEL_PATH=/path/to/checkpoint \
ROBOTWIN_ROOT="$(pwd)/third_party/RoboTwin" \
ROBOTWIN_GPU_IDS='[0,1,2,3]' \
./recipes/eval/eval_robotwin2.sh
```

See `experiments/robotwin/README.md` for single-task and manager examples. RoboTwin assets and separately downloaded models remain subject to their own distribution and use terms.

## 📚 Documentation

- [Architecture](docs/architecture.md)
- [Data-preparation](tools/data/README.md)
- [Post-training](docs/posttraining.md)
- [Evaluation](docs/evaluation.md)
- [Policy-server](docs/server.md)

## 🙏 Acknowledgements

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [DreamZero](https://github.com/dreamzero0/dreamzero)
- [FastWAM](https://github.com/yuantianyuan01/FastWAM)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)

Please follow each upstream repository's license and model-use terms.

## 📝 Citation

```bibtex
@article{su2026worldscape_policy_2,
  title={WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory},
  author={Su, Haisheng and Liu, Zongdai and Jin, Xin and Dou, Haoxuan and Hu, Chengming and Li, Baorun and Liu, Zhanwang and Xu, Ruiyan and Fang, Jianjie and Zhang, Xin and Yang, Zhenjie and Yang, Xue and Gao, Chen and Yan, Junchi and Li, Yong and Wu, Wei},
  journal={arXiv preprint arXiv:2607.18840},
  year={2026}
}

@article{su2026worldscape_policy,
  title={WorldScape Policy: Generalizable Robotic Learning via a Foundation World Model},
  author={Su, Haisheng and Shang, Yu and Zhang, Xin and Dou, Haoxuan and Jin, Xin and Zhang, Hongling and Gao, Chen and Li, Yong and Wu, Wei},
  year={2026}
}
```

