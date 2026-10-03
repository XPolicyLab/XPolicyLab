# Training

This directory contains the Clean50 data loader, configuration, and distributed training entry point for ME-Dex-1.0.

The recipe uses the unified tactile encoder with 16 RoboTwin surface slots, 48-dimensional latent vectors, and 18 temporal slices: 2 observed slices followed by 16 future slices. Set `topology` to `full_joint` or `h_bridge` in the YAML configuration.

Start a run with:

```bash
torchrun --nproc_per_node=16 -m training.train \
  --config training/configs/clean50_uni.yaml
```
