
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>Motus: A Unified Latent Action World Model</h1>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://motus-robotics.github.io/motus"><img alt="Homepage"
    src="https://img.shields.io/badge/Motus-Homepage-4287f5?logo=readme&logoColor=white"/></a>
  <a href="https://huggingface.co/motus-robotics"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-motus--robotics-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://arxiv.org/abs/2512.13030"><img alt="arXiv"
    src="https://img.shields.io/badge/arXiv-2512.13030-b31b1b?logo=arxiv&logoColor=white"/></a>
  <br>
  <a href="https://motus-robotics.github.io/assets/motus/png/feishu.jpg"><img alt="Feishu"
    src="https://img.shields.io/badge/Feishu-Motus-blue?logo=lark&logoColor=white"/></a>
  <a href="https://motus-robotics.github.io/assets/motus/png/wechat.jpg"><img alt="WeChat"
    src="https://img.shields.io/badge/WeChat-Motus-green?logo=wechat&logoColor=white"/></a>
  <a href="LICENSE"><img alt="License"
    src="https://img.shields.io/badge/License-Apache--2.0-f5de53?logo=apache&color=f5de53"/></a>
</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Updates](#updates)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Checkpoints](#model-checkpoints)
- [Data Format](#data-format)
- [Running Inference](#running-inference)
- [Training](#training)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

**Motus** is a **unified latent action world model** that leverages existing pretrained models and rich, sharable motion information. Motus introduces a **Mixture-of-Transformers (MoT)** architecture to integrate three experts (understanding, action, and video generation) and adopts a **UniDiffuser-style scheduler** to enable flexible switching between different modeling modes (World Models, Vision-Language-Action Models, Inverse Dynamics Models, Video Generation Models, and Video-Action Joint Prediction Models). Motus further leverages **optical flow** to learn **latent actions** and adopts a **three-phase training pipeline** and **six-layer data pyramid**, thereby extracting pixel-level "delta action" and enabling large-scale action pretraining.

| Component | Base Model | Parameters |
|-----------|------------|------------|
| **VGM (Video Generation Model)** | [Wan2.2-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | ~5.00B |
| **VLM (Vision-Language Model)** | [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | ~2.13B |
| **Action Expert** | - | ~641.5M |
| **Understanding Expert** | - | ~253.5M |
| **Total** | - | **~8B** |

**Key Results (RoboTwin 2.0 Simulation.** With 50 clean and 500 randomized data entries per task, we merge the data from all 50 tasks for multi-task training.):
- **87.02%** average success rate (+15% over X-VLA, +45% over π₀.₅)

## Updates

- [2025-12-16] **Initial release of Motus with pretrained checkpoints and training code.**
- [2025-12-20] **Simple RoboTwin inference**
- [2025-12-24] **LeRobotDataset format support**  
- [2025-12-24] **Optimized training scripts**
- [2025-12-26] **MultiLeRobotDataset format support**
- [2025-12-27] **RoboTwin raw dataset conversion**
- [2025-12-27] **Three-view image concatenation scripts**

We welcome community members to help maintain and extend Motus. Welcome to join the Motus community and contribute together!

## Requirements

| Mode | VRAM | Recommended GPU |
|------|------|-----------------|
| Inference (with pre-encoded T5) | > 24 GB | RTX 5090 |
| Inference (without pre-encoded T5) | ~ 41 GB | A100 (40GB) / A100 (80GB) / H100 / B200 |
| Training | > 80 GB | A100 (80GB) / H100 / B200 |

## Installation

```bash
# Clone the repository
git clone https://github.com/thu-ml/Motus.git
cd Motus

# Create conda environment
conda create -n motus python=3.10 -y
conda activate motus

# install torch (cuda12.8)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# install flash 
pip install flash-attn --no-build-isolation

# Install motus dependencies
pip install -r requirements.txt

# (Optinal) Install lerobot dependencies
pip install --no-deps lerobot==0.3.2
pip install -r requirements/lerobot.txt
```

## Model Checkpoints

We provide multiple checkpoints for different use cases:

| Model | Use Case | Description | Checkpoint Path |
|-------|----------|-------------|-----------------|
| **Motus_Wan2_2_5B_pretrain** | Pretrain / VGM Backbone | Stage 1 VGM pretrained checkpoint | [`motus-robotics/Motus_Wan2_2_5B_pretrain`](https://huggingface.co/motus-robotics/Motus_Wan2_2_5B_pretrain) |
| **Motus** | Fine-Tuning | Stage 2 latent action pretrained checkpoint | [`motus-robotics/Motus`](https://huggingface.co/motus-robotics/Motus) |
| **Motus_robotwin2** | Inference / Fine-Tuning | Stage 3 RoboTwin2 fine-tuned checkpoint | [`motus-robotics/Motus_robotwin2`](https://huggingface.co/motus-robotics/Motus_robotwin2) |

**Download checkpoints:**
```bash
# Create pretrained models directory
mkdir -p pretrained_models

# Download Motus checkpoints
huggingface-cli download motus-robotics/Motus_Wan2_2_5B_pretrain --local-dir ./pretrained_models/Motus_Wan2_2_5B_pretrain
huggingface-cli download motus-robotics/Motus --local-dir ./pretrained_models/Motus
huggingface-cli download motus-robotics/Motus_robotwin2 --local-dir ./pretrained_models/Motus_robotwin2

# Download foundation models
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./pretrained_models/Qwen3-VL-2B-Instruct
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./pretrained_models/Wan2.2-TI2V-5B
```

**Update config paths** in your embodiment-specific config file (e.g., `configs/robotwin.yaml`, `configs/ac_one.yaml`, or other embodiment configs):
```yaml
model:
  wan:
    checkpoint_path: "./pretrained_models/Motus_Wan2_2_5B_pretrain"
    config_path: "./pretrained_models/Motus_Wan2_2_5B_pretrain"
    vae_path: "./pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  vlm:
    checkpoint_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
    config_path: "./pretrained_models/Qwen3-VL-2B-Instruct"
```

## Data Format

Motus supports three types of datasets with specific directory structures for optimal training and inference.

**📖 See detailed guide:** [**Data Format Guide**](DATA_FORMAT.md)

**Quick Overview:**
- **RoboTwin 2.0**: Simulation data with clean/randomized splits
- **Real-World**: AC-One, Aloha-Agilex-2 robot data

**Data Conversion Tools:**
- [RoboTwin Dataset Converter](data/robotwin2/robotwin_data_convert/README.md)
- [Multi-Camera Concatenation](data/utils/multi_camera_concat.py)

## Running Inference

**📖 See detailed guide:** [**Inference Guide**](INFERENCE.md)

- **RoboTwin 2.0**: [Evaluation Setup](inference/robotwin/Motus/README.md)
- **Real-World**: Minimal inference without robot environment

## Training

Motus follows a **three-stage training pipeline**:

| Stage | Data | Training |
|-------|------|----------|
| **Pretrained Foundation Models** | Level 1: Web Data | VGM and VLM |
| **Stage 1 (VGM Training)** | Level 2: Egocentric Human Videos<br>Level 3: Synthetic Data<br>Level 5: Multi-Robot Task Trajectory | Only VGM |
| **Stage 2 (Motus Pretraining)** | Level 2: Egocentric Human Videos<br>Level 3: Synthetic Data<br>Level 4: Task-agnostic Data<br>Level 5: Multi-Robot Task Trajectory | Motus (all 3 experts, with **latent actions**) |
| **Stage 3 (Motus SFT)** | Level 6: Target-Robot Task Trajectory | Motus (all 3 experts, with actions) |

The six-layer data pyramid is shown in the figure here:

<img width="615" height="455" alt="image" src="https://github.com/user-attachments/assets/b1389887-2f6b-4e82-87f9-08f0525301b5" />

**📖 See detailed guide:** [**Training Guide**](TRAINING.md)

**Data Preparation:**
- [RoboTwin Dataset Converter](data/robotwin2/robotwin_data_convert/README.md)
- [Multi-Camera Concatenation](data/utils/multi_camera_concat.py)

## Troubleshooting

**📖 Detailed guides:**
- [Inference Issues](INFERENCE.md#troubleshooting)
- [Training Issues](TRAINING.md#troubleshooting)
- [Data Format Issues](DATA_FORMAT.md)

## Citation
If you find our work helpful, please cite us:

```bibtex
@misc{bi2025motusunifiedlatentaction,
      title={Motus: A Unified Latent Action World Model}, 
      author={Hongzhe Bi and Hengkai Tan and Shenghao Xie and Zeyuan Wang and Shuhe Huang and Haitian Liu and Ruowen Zhao and Yao Feng and Chendong Xiang and Yinze Rong and Hongyan Zhao and Hanyu Liu and Zhizhong Su and Lei Ma and Hang Su and Jun Zhu},
      year={2025},
      eprint={2512.13030},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.13030}, 
}
```

Thank you!

