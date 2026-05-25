# NVIDIA DreamZero: World Action Models Are Zero-Shot Policies
A research project from [NVIDIA GEAR Lab](https://research.nvidia.com/labs/gear/).

[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2602.15922-b31b1b.svg)](https://arxiv.org/abs/2602.15922)

[[Project Page](https://dreamzero0.github.io/)] [[Paper](https://arxiv.org/abs/2602.15922)]

DreamZero is a World Action Model that jointly predicts actions and videos, achieving strong zero-shot performance on unseen tasks. This release package contains everything needed to load a pretrained DreamZero model and run distributed inference via a WebSocket server.

## News

- **02/27:** DreamZero is **#1 on both [MolmoSpaces]([https://huggingface.co/spaces/ai2-adapt/MolmoSpaces](https://molmospaces.allen.ai/leaderboard)) and [RoboArena]([https://robo-arena.github.io/](https://robo-arena.github.io/leaderboard))**! DreamZero-DROID is trained *from scratch* using only the DROID dataset — no pretraining on large-scale robot data, unlike competing VLAs. This demonstrates the strength of video-model backbones for generalist robot policies (VAMs/WAMs).
- **02/27:** Released **DreamZero-AgiBot checkpoint** and **post-training code** for efficient few-shot adaptation. Post-train on just ~30 minutes of play data for your specific robot, and see the robot do basic language following and pick-and-place (see YAM experiments in our paper for more detail).
- **02/20:** Released the **full training codebase, preprocessed dataset, and guide for new embodiments** to replicate the DreamZero-DROID checkpoint and train on your own robot. See [Adding a New Embodiment to DreamZero](docs/DATASET_TO_GEAR_AND_TRAIN.md) for a step-by-step walkthrough.

## Features

**Available Now**
- Pretrained DreamZero-DROID model checkpoint [[Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)]
- Pretrained DreamZero-AgiBot checkpoint (for post-training on new embodiments) [[Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot)]
- Distributed WebSocket inference server (GB200, H100)
- DiT caching for optimized inference (~0.6s on GB200, ~3s on H100)
- DROID simulation evaluation support
- [RoboArena](https://robo-arena.github.io/) integration (DROID real)
- Video generation and saving (MP4)
- LoRA and full fine-tuning training scripts
- Training on new embodiments (AgiBot, YAM) — see [guide](docs/DATASET_TO_GEAR_AND_TRAIN.md)

**Coming Soon**
- [PolaRiS](https://polaris-evals.github.io/) simulation environment support
- [Genie 3.0](https://arxiv.org/abs/2601.02078) sim environment support for DreamZero-AgiBot

## Testing Out DreamZero in Simulation with API
We provide an inference script that directly evaluates a hosted DreamZero-DROID policy on [`sim_evals`](https://github.com/arhanjain/sim-evals). To test out the policy, first request access to the API via this form [link](https://forms.gle/zCj5zjDvHsoeuMXU7). Then, follow these instructions to install [`sim_evals`](https://github.com/arhanjain/sim-evals) and launch evaluation.

```bash
# Clone repository
git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Activate uv environment
uv sync
source .venv/bin/activate

# [Optional] update pytorch versions
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# Download assets (may need to export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> first)
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets

# Run eval script
cd ..
python eval_utils/run_sim_eval.py --host <API_HOST> --port <API_PORT> 
```

The outputs are saved in `runs` directory.


## Quick Start

### Prerequisites

- **Python**: 3.11
- **Hardware**: Multi-GPU setup (tested on GB200, H100)
  - Minimum: 2 GPUs for distributed inference
- **CUDA**: Compatible GPU with CUDA 12.9+

### Installation

1. **Create conda environment:**
```bash
conda create -n dreamzero python=3.11
conda activate dreamzero
```

2. **Install dependencies (PyTorch 2.8+ with CUDA 12.9+):**
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

3. **Install flash attention:**
```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

4. **[GB200 ONLY, SKIP FOR H100] Install Transformer Engine:**
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

5. **[GB200 ONLY FOR TENSORRT, SKIP FOR H100] Install Tensorrt:**
```bash
pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 transformer_engine_torch==2.10.0
```

## Downloading Pretrained Checkpoints

### DreamZero-DROID (for inference)

We release a 14B pretrained DROID checkpoint on [Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-DROID). To download the checkpoint, run

```bash
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir <path/to/checkpoint>
```

### DreamZero-AgiBot (for fine-tuning on new embodiments)

To fine-tune DreamZero on a new embodiment (e.g. YAM, AgiBot), download the pretrained [DreamZero-AgiBot](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot) checkpoint (~45GB) to `./checkpoints/DreamZero-AgiBot`:

```bash
git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot ./checkpoints/DreamZero-AgiBot
```

Or with the Hugging Face CLI:

```bash
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot
```

The YAM and AgiBot training scripts use `pretrained_model_path=./checkpoints/DreamZero-AgiBot` by default. See the [new embodiment guide](docs/DATASET_TO_GEAR_AND_TRAIN.md) for usage.

## Running the Inference Server

### Command Overview

The inference server uses PyTorch distributed training utilities to parallelize the model across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path <path/to/checkpoint>
```

(Optional only for GB200) Tensorrt enables faster generation
```bash
export LOAD_TRT_ENGINE=<path/to/checkpoint>/tensorrt/wan/WanModel_nvfp4.trt 
export DYNAMIC_CACHE_SCHEDULE=true 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 /mnt/aws-lfs-02/shared/seonghyeony/dreamzero/socket_test_optimized_AR.py --port 8000 --enable-dit-cache --model-path <path/to/checkpoint>
```
To verify the server is working, run a test client. The first few inferences will take a few minutes to warm up. After warming up, inference takes ~0.6s on GB200 and ~3s on H100.

```
python test_client_AR.py --port 5000
```

### Command-line Arguments

- `--port`: Port number for the WebSocket server (default: 8000)
- `--model-path`: Path to the pretrained model checkpoint directory
- `--enable-dit-cache`: Enable caching in DiT layers for faster inference (recommended)
- `--max-chunk-size`: Override max_chunk_size for inference (optional)
- `--timeout-seconds`: Server timeout in seconds (default: 50000)
- `--index`: Index for output directory naming (default: 0)


### Output

The server saves:
- **Videos**: Generated video predictions as MP4 files in `{model_path}/real_world_eval_gen_{date}_{index}/{checkpoint_name}/`
- **Input observations**: Saved per message in `{output_dir}/inputs/{msg_index}_{timestamp}/`


## Training

> **Training on a new embodiment?** See [Adding a New Embodiment to DreamZero](docs/DATASET_TO_GEAR_AND_TRAIN.md) for a complete guide on converting your dataset, configuring modalities, and launching training. <em>Make sure to align the 3 camera view order to ensure positive transfer.</em>

### Downloading Pretrained Base Model Weights

DreamZero is built on top of [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) and uses the [umt5-xxl](https://huggingface.co/google/umt5-xxl) tokenizer. Download both before training:

```bash
pip install "huggingface_hub[cli]"

# You may need to set your HuggingFace token:
# export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

# Download Wan2.1 model weights (~28GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download umt5-xxl tokenizer
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

> **Note:** The training script will auto-download these if they are not found at the configured paths, but pre-downloading is recommended to avoid delays at launch.

### DROID Dataset

We release the preprocessed DROID dataset used to train DreamZero on HuggingFace: [GEAR-Dreams/DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data).

This dataset is derived from the [DROID 1.0.1](https://droid-dataset.github.io/) dataset with the following modifications:
- Converted from RLDS/TFDS format to [LeRobot](https://github.com/huggingface/lerobot) v2.0 format
- Idle frames removed using [Physical Intelligence's idle frame detector](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#data-filtering) (`droid_sample_ranges_v1_0_1.json`)
- Episodes without language annotations are filtered out
- Successful episodes only (episodes with non-zero reward)
- 3 camera views: `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`

**To download the preprocessed dataset (~131GB):**

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
```

If you want to reproduce the dataset conversion from raw DROID 1.0.1 yourself (or modify the filtering), see [docs/DROID_CONVERSION.md](docs/DROID_CONVERSION.md).

### Running Training

```bash
# Configure paths (override defaults as needed)
export DROID_DATA_ROOT="./data/droid_lerobot"
export OUTPUT_DIR="./checkpoints/dreamzero_droid"
export NUM_GPUS=4

# Point to your downloaded model weights (if not using default paths)
export WAN_CKPT_DIR="./checkpoints/Wan2.1-I2V-14B-480P"
export TOKENIZER_DIR="./checkpoints/umt5-xxl"

# Launch training
bash scripts/train/droid_training.sh
```

**Using Wan2.2-TI2V-5B backbone (5B params, lower VRAM):** To train with the smaller Wan2.2-TI2V-5B model instead of Wan2.1-I2V-14B, see [docs/WAN22_BACKBONE.md](docs/WAN22_BACKBONE.md) and run `bash scripts/train/droid_training_wan22.sh`.

### Training Configuration

The training script uses Hydra for configuration and DeepSpeed ZeRO Stage 2 for distributed training. Key defaults:

| Parameter | Default | Description |
|---|---|---|
| `NUM_GPUS` | 4 | Number of GPUs |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `learning_rate` | 1e-5 | Learning rate |
| `max_steps` | 10 | Max training steps (increase for full training) |
| `warmup_ratio` | 0.05 | Warmup ratio |
| `weight_decay` | 1e-5 | Weight decay |
| `image_resolution_width` | 320 | Image width |
| `image_resolution_height` | 176 | Image height |
| `num_frames` | 33 | Number of video frames |
| `action_horizon` | 24 | Action prediction horizon |
| `save_lora_only` | true | Only save LoRA weights |
| `bf16` | true | Use bfloat16 precision |

> **Note:** `max_steps=10` is set for a quick sanity check. For full training, increase this to your desired number of steps and configure `save_steps` / `save_strategy` accordingly.


## Citation

If you use DreamZero in your research, please cite:

```bibtex
@misc{ye2026worldactionmodelszeroshot,
      title={World Action Models are Zero-shot Policies}, 
      author={Seonghyeon Ye and Yunhao Ge and Kaiyuan Zheng and Shenyuan Gao and Sihyun Yu and George Kurian and Suneel Indupuru and You Liang Tan and Chuning Zhu and Jiannan Xiang and Ayaan Malik and Kyungmin Lee and William Liang and Nadun Ranawaka and Jiasheng Gu and Yinzhen Xu and Guanzhi Wang and Fengyuan Hu and Avnish Narayan and Johan Bjorck and Jing Wang and Gwanghyun Kim and Dantong Niu and Ruijie Zheng and Yuqi Xie and Jimmy Wu and Qi Wang and Ryan Julian and Danfei Xu and Yilun Du and Yevgen Chebotar and Scott Reed and Jan Kautz and Yuke Zhu and Linxi "Jim" Fan and Joel Jang},
      year={2026},
      eprint={2602.15922},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.15922}, 
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for detailed error messages
- Verify your checkpoint is compatible with this release

[![Star History Chart](https://api.star-history.com/svg?repos=dreamzero0/dreamzero&type=Date)](https://star-history.com/#dreamzero0/dreamzero&Date)
