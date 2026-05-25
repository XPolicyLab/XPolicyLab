# Motus Training Guide

This guide covers training Motus models from data preparation to distributed training.

## Data Preparation

### RoboTwin 2.0 Dataset Conversion

**📖 See detailed guide:** [**RoboTwin Data Conversion Guide**](data/robotwin2/robotwin_data_convert/README.md)

### Multi-Camera View Concatenation

**📖 See utility script:** [**Multi-Camera Concatenation Utility**](data/utils/multi_camera_concat.py)

## Training

### 1. Fine-Tuning from Pretrained Checkpoint (Stage 3)

Currently, we support the following training methods:

- **Single-node training** with torchrun + DeepSpeed
- **Multi-node training** on SLURM clusters
- **Resume training** from checkpoints

Since Motus is based on proven architectures, you are free to apply other techniques (e.g., FSDP) by following standard distributed training practices. We provide example training scripts for different scenarios, which you can directly use to kick off your own training.

To provide a better understanding, we elaborate the line-by-line explanation of the basic training script (`scripts/train.sh`) with our example configuration:

```bash
#!/bin/bash
# Define your env settings here 
# e.g., nccl, network, proxy, etc.

TASK="robotwin"  # Define your task name here
CONFIG_FILE="configs/robotwin.yaml"  # Define your dataset config path here

export OUTPUT_DIR="outputs/motus-${TASK}" # Define your output directory here

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# Single-node training with torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train/train.py \
    --deepspeed configs/zero1.json \  # DeepSpeed config file, you can modify it to your own using other sharding strategies
    --config $CONFIG_FILE \
    --run_name $TASK \
    --report_to tensorboard
```

**Step 1:** Set the pretrain checkpoint path in your config (e.g., `configs/robotwin.yaml`):
```yaml
finetune:
  checkpoint_path: ./pretrained_models/Motus  # Stage 2 pretrained checkpoint
```

**Step 2:** Run training using one of the following methods:

**Option A: Basic single-node training (recommended for getting started)**
```bash
bash scripts/train.sh
```

**Option B: SLURM cluster training**
```bash
# Single node with SLURM
sbatch scripts/slurm/slurm_single_node.sh

# Multi-node with SLURM
sbatch scripts/slurm/slurm_multi_node.sh
```

### 2. Resume Training

To resume from an interrupted checkpoint:

**Step 1:** Set the resume checkpoint path in your config:
```yaml
resume:
  checkpoint_path: ./checkpoints/motus_finetune/checkpoint_step_10000
```

**Step 2:** Run training (same commands as above):
```bash
# Basic resume training
bash scripts/train.sh

# Or on SLURM cluster
sbatch scripts/slurm/slurm_multi_node.sh
```

> **Note:** When resuming or fine-tuning, WAN and VLM pretrained weights are **not reloaded** (only VAE is needed). This prevents overwriting fine-tuned weights.

### 3. Training from Scratch

To train Motus from scratch (load WAN + VLM pretrained weights):

**Step 1:** Ensure `resume.checkpoint_path` and `finetune.checkpoint_path` are both `null` in your config:
```yaml
resume:
  checkpoint_path: null
finetune:
  checkpoint_path: null
```

**Step 2:** Run training:
```bash
# Basic training from scratch
bash scripts/train.sh

# Or on SLURM cluster for large-scale training
sbatch scripts/slurm/slurm_multi_node.sh
```

This will load:
- Wan2.2-5B pretrained weights from `model.wan.checkpoint_path`
- Qwen3-VL pretrained weights from `model.vlm.checkpoint_path`

**Training Scripts Overview:**
- `scripts/train.sh` - Basic single-node training script (recommended for getting started)
- `scripts/slurm/` - SLURM cluster scripts for single-node and multi-node distributed training

## Troubleshooting

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Slow training | No flash-attn | Install flash-attention: `pip install flash-attn --no-build-isolation` |
| WAN/VLM weights not loading | Resume/finetune mode | Set both `resume.checkpoint_path` and `finetune.checkpoint_path` to `null` |
| NCCL timeout | Network issues | Check NCCL environment variables in scripts |

---

**📖 Related Guides:**
- [Inference Guide](INFERENCE.md)
- [Data Format](DATA_FORMAT.md)
- [RoboTwin Data Conversion](data/robotwin2/robotwin_data_convert/README.md)
