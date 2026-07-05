<div align="center">

  # X-WAM

  **Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising**

  [![Paper](https://img.shields.io/badge/📄-Paper-red)](https://arxiv.org/abs/2604.26694)
  [![Project Page](https://img.shields.io/badge/🌐-Project_Page-blue)](https://sharinka0715.github.io/X-WAM/)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/sharinka0715/X-WAM-checkpoints)
  [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

</div>

---

## 📰 News

- **2026.06.09**: We release the post-training code, checkpoints, and datasets. Welcome to try X-WAM!
- **2026.05.07**: We update our paper on [arXiv](https://arxiv.org/abs/2604.26694), with real-robot experiments and revised figures.
- **2026.04.30**: We release our paper on [arXiv](https://arxiv.org/abs/2604.26694).

---

## 💡 About X-WAM

**X-WAM** is a unified 4D World Action Model that simultaneously targets four objectives within a single architecture: high-fidelity video generation, 3D spatial reconstruction, high policy success rate, and efficient action execution. Built on the powerful visual priors of pretrained video foundation models, X-WAM takes multi-view RGB observations and current robot states as inputs to jointly generate future 4D observations alongside the robot's future states and actions.

### Key Features:

* **🌐 Unified 4D Modeling**: Jointly optimizes video generation, 3D spatial reconstruction, and policy execution in a single framework, going beyond 2D pixel-space modeling.
* **🧊 Lightweight Depth Adaptation**: Replicates the final blocks of the pretrained DiT as an interleaved depth branch, achieving high-quality spatial modeling without doubling sequence lengths or disrupting pretrained visual priors.
* **⚡ Asynchronous Noise Sampling (ANS)**: Rapidly decodes actions with fewer denoising steps for real-time execution, while dedicating the full sequence of steps to generate high-fidelity video.
* **🚀 Strong Performance**: Pretrained on over 5,800 hours of robotic data, achieving state-of-the-art results on RoboCasa and RoboTwin 2.0 benchmarks.

</div>

---

## 🏆 Benchmark

We evaluate **X-WAM** on two standard simulation benchmarks: **RoboCasa** and **RoboTwin 2.0**.

### Policy Evaluation

| Benchmark | Setting | Performance | Checkpoints | Dataset |
| :--- | :--- | :--- | :--- | :--- |
| **RoboCasa** | 24 kitchen manipulation tasks | **79.2%** (Avg Success Rate) | [X-WAM-checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints) | [X-WAM-RoboCasa](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboCasa) |
| **RoboTwin 2.0** | Clean | **89.8%** (Avg Success Rate) | [X-WAM-checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints) | [X-WAM-RoboTwin](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin) |
| **RoboTwin 2.0** | Randomized | **90.7%** (Avg Success Rate) | [X-WAM-checkpoints](https://huggingface.co/sharinka0715/X-WAM-checkpoints) | [X-WAM-RoboTwin](https://huggingface.co/datasets/sharinka0715/X-WAM-RoboTwin) |

---


## 🚀 Installation

Please make sure these key packages have correct versions:

- `python>=3.10` (tested on Python 3.10; evaluation environments require Python 3.10)
- `torch>=2.4.0` (tested on 2.8.0+cu129)
- `numpy<1.26` (tested on 1.23.5)
- `diffusers>=0.31.0` (tested on 0.38.0)
- `transformers>=4.49.0,<=4.51.3` (tested on 4.51.3)
- `flash-attn` (tested on 2.8.3)

```bash
# 1. Install PyTorch >= 2.4.0 (below is an example)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# 2. Install required packages
pip install -r requirements.txt

# 3. Install FlashAttention (network required)
pip install flash-attn --no-build-isolation
```

### Download Checkpoints and Datasets

By default, the project expects the following directory layout:

```
X-WAM/
├── checkpoints/
│   ├── wan22_5b/          # Wan2.2-TI2V-5B base weights
│   ├── pretrained/        # X-WAM pretrained checkpoint
│   ├── robocasa_sft/      # X-WAM RoboCasa fine-tuned checkpoint
│   └── robotwin_sft/      # X-WAM RoboTwin fine-tuned checkpoint
└── datasets/
    ├── RoboCasa/          # RoboCasa post-training dataset
    └── RoboTwin/          # RoboTwin post-training dataset
```

Download with Hugging Face CLI:

```bash
# Checkpoints (pretrained + fine-tuned)
huggingface-cli download sharinka0715/X-WAM-checkpoints --local-dir checkpoints

# Datasets
huggingface-cli download sharinka0715/X-WAM-RoboCasa --repo-type dataset --local-dir datasets/RoboCasa
huggingface-cli download sharinka0715/X-WAM-RoboTwin --repo-type dataset --local-dir datasets/RoboTwin
```

For Wan2.2-TI2V-5B base weights, please refer to the [official Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) and place them under `checkpoints/wan22_5b/`.

---


## 🛠️ Post-Training

### Configuration

The training script reads two config files:

- **Model config**: `configs/model/wan22_5b_sft.yaml` — model architecture, training hyperparameters, and inference settings
- **Dataset config**: `configs/data/{bench_name}.yaml` — data path, normalization statistics, augmentation, and video settings

The dataset config is selected by the `dataset` field in the model config (default: `robocasa`). All config fields can be overridden from the command line using OmegaConf dot notation.

### Data Structure

Organize your training data in the following structure:

```
dataset_root/
├── metadata.json                          # {episode_id: num_frames} mapping
├── data/
│   ├── chunk-0000/
│   │   ├── episode_0000000.json
│   │   ├── episode_0000001.json
│   │   └── ...
│   ├── chunk-0001/
│   └── ...
├── video/
│   ├── robot0_agentview_left/
│   │   ├── chunk-0000/
│   │   │   ├── episode_0000000.mp4
│   │   │   └── ...
│   │   └── ...
│   ├── robot0_agentview_right/
│   └── robot0_eye_in_hand/
└── depth/
    ├── robot0_agentview_left/
    ├── robot0_agentview_right/
    └── robot0_eye_in_hand/
```

Each episode JSON contains:

```json
{
    "num_frames": 413,
    "instructions": ["close the cabinet doors"],
    "observations": {
        "robot0_agentview_left": {
            "type": "static",
            "rgb_path": "video/robot0_agentview_left/chunk-0000/episode_0000000.mp4",
            "depth_path": "depth/robot0_agentview_left/chunk-0000/episode_0000000.mp4",
            "start": 0, "end": 413, "fps": 20.0
        },
        "robot0_eye_in_hand": {
            "type": "dynamic",
            "rgb_path": "video/robot0_eye_in_hand/chunk-0000/episode_0000000.mp4",
            "depth_path": "depth/robot0_eye_in_hand/chunk-0000/episode_0000000.mp4",
            "start": 0, "end": 413, "fps": 20.0
        }
    },
    "proprios": {
        "left_ee_pos": [[x, y, z], ...],
        "left_ee_rotm": [[r00, r01, ..., r22], ...],
        "left_gripper_pos": [[g], ...]
    },
    "actions": {
        "left_ee_pos": [[x, y, z], ...],
        "left_ee_rotm": [[r00, r01, ..., r22], ...],
        "left_gripper_pos": [[g], ...],
        "raw_actions": [[...], ...]    // optional
    }
}
```

> **Note on `raw_actions`:** This field is only present in the RoboCasa dataset and represents the raw commands sent directly to the controller. When `raw_actions` is present in a dataset, ignore the other fields under `actions` and use `raw_actions` directly as the action.

### Training

```bash
torchrun \
    --nnodes=$MLP_WORKER_NUM \
    --node_rank=$MLP_ROLE_INDEX \
    --nproc_per_node=$MLP_WORKER_GPU \
    --master_addr=$MLP_WORKER_0_HOST \
    --master_port=$MLP_WORKER_0_PORT \
    scripts/train_sft.py dataset={bench_name}
```

Available dataset configs: `robocasa`, `robotwin`. You can override any config field from the command line.

### Examples

**RoboCasa** (single-node, 8 GPUs):

```bash
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_sft.py \
    dataset=robocasa \
    exp_name=robocasa_sft
```

**RoboTwin** (single-node, 8 GPUs, with adjusted warmup and training steps):

```bash
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train_sft.py \
    dataset=robotwin \
    exp_name=robotwin_sft \
    num_warmup_steps=400 \
    num_training_steps=40000
```

---


## 📊 Evaluation

Please refer to [`evaluation/README.md`](evaluation/README.md) for detailed evaluation instructions on RoboCasa and RoboTwin 2.0 benchmarks.

---


## 📚 Citation

If you find this project useful, please consider citing:

```bibtex
@article{guo2026xwam,
  title={Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising},
  author={Guo, Jun and Li, Qiwei and Li, Peiyan and Chen, Zilong and Sun, Nan and Su, Yifei and Wang, Heyun and Zhang, Yuan and Li, Xinghang and Liu, Huaping},
  journal={arXiv preprint arXiv:2604.26694},
  year={2026}
}
```

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
