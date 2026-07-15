# Xiaomi_Robotics_1 Installation

> **Note:** Only inference code is available at this time. Training code will be released soon.

`install.sh` is the recommended path. This document provides additional detail on manual installation and checkpoint preparation for inference evaluation.

## 1. One-command Install

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
bash install.sh
conda activate mibot
```

The installer creates the `mibot` conda environment, installs PyTorch 2.8, Flash Attention, and other core dependencies.

## 2. Manual Install Equivalent

```bash
conda create -n mibot python=3.12 -y
conda activate mibot

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1 scipy numpy Pillow ninja
pip install flash-attn==2.8.3 --no-build-isolation
```

## 3. Prepare Inference Weights

Download the checkpoint from the [RoboDojo official dataset](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo). Only the `ckpt/RoboDojo/Xiaomi_Robotics_1/` folder is needed:

```bash
cd XPolicyLab/policy/Xiaomi_Robotics_1
mkdir -p checkpoints

# Using hf cli
hf download RoboDojo-Benchmark/RoboDojo \
  --repo-type dataset \
  --include "ckpt/RoboDojo/Xiaomi_Robotics_1/*" \
  --local-dir checkpoints/Xiaomi_Robotics_1 \
  --local-dir-use-symlinks False
```

The expected layout:

```text
policy/Xiaomi_Robotics_1/checkpoints/Xiaomi_Robotics_1/
```

At evaluation time, the checkpoint is resolved via `deploy.yml` field `ckpt_name` or the `model_dir` field pointing to an absolute path.

## 4. Smoke Checks

```bash
conda activate mibot
python -c "import torch; print('cuda:', torch.cuda.is_available())"
python -c "import transformers; print('transformers:', transformers.__version__)"
python -c "import XPolicyLab; print('XPolicyLab ok')"
```
