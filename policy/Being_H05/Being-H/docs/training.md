# Training Guide

Being-H uses FSDP distributed training via `torchrun`. See `scripts/train/train_libero_example.sh` for single-embodiment training and `scripts/train/train_cross_emb_example.sh` for cross-embodiment training.

## Quick Start

```bash
export PYTHONPATH=.

# Single-embodiment training
bash scripts/train/train_libero_example.sh

# Cross-embodiment training
bash scripts/train/train_cross_emb_example.sh
```

## Required Arguments

| Argument | Description |
|----------|-------------|
| `--mllm_path` | Path to InternVL backbone (e.g., `InternVL3_5-2B`) |
| `--expert_path` | Path to Qwen expert model (e.g., `Qwen3-0.6B`) |
| `--resume_from` | Path to Being-H checkpoint to resume from |
| `--dataset_config_file` | YAML file specifying datasets |
| `--output_dir` | Output directory for checkpoints |

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_steps` | - | Total training steps |
| `--learning_rate` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-5` | Weight decay |
| `--warmup_ratio` | `0.05` | Warmup ratio |
| `--lr_scheduler` | `cosine` | LR schedule (`cosine` or `constant`) |
| `--save_steps` | `10000` | Checkpoint save interval |
| `--save_steps_start` | `25000` | Start saving after this step |
| `--logging_steps` | `10` | Logging interval |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation |

## Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_flow_matching` | `True` | Enable flow matching |
| `--use_expert` | `True` | Use Qwen expert module |
| `--layer_module` | `Qwen3MoTDecoderLayer` | Decoder layer class |
| `--conv_style` | `being_h0` | Prompt template style |
| `--action_chunk_length` | `16` | Actions per chunk |
| `--resume_model_only` | `False` | Only load model weights |

## Action Expert Initialization

Being-H uses a Mixture of **Action Expert** architecture for action prediction. The initialization behavior depends on your training scenario:

| Scenario | Action Expert Initialization |
|----------|------------------------------|
| **Pretraining from scratch** | Random initialization by default (not from Qwen3-0.6B pretrained weights) |
| **Post-training with `--resume_from`** | Loaded from checkpoint via `load_state_dict()` (inherits pretrained Action Expert weights) |

**Key Points:**

- When starting pretraining from scratch, the Action Expert parameters are randomly initialized, even though the architecture follows Qwen3-0.6B design
- When post-training using `--resume_from /path/to/Being-H05-2B`, all model weights (including the trained Action Expert) are loaded from the checkpoint
- The codebase provides infrastructure to initialize the Action Expert from pretrained Qwen weights (`init_expert()` with `from_scratch=False`), but this is not used in our practice

## Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--force_image_size` | `224` | Input image size |
| `--down_sample_ratio` | `0.5` | ViT downsample ratio |
| `--max_view_num` | `-1` | Max camera views (-1 = all) |
| `--use_fixed_view` | `False` | Use single fixed view |
| `--num_workers` | `12` | DataLoader workers |
| `--prefetch_factor` | `8` | DataLoader prefetch factor |

## Sequence Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_num_tokens` | `8704` | Max tokens per batch |
| `--expected_num_tokens` | `8192` | Target tokens per batch |
| `--prefer_buffer_before` | `4096` | Buffer preference threshold |
| `--max_buffer_size` | `4` | Maximum buffer size |
| `--attn_mode` | `causal` | Attention mode |

## Freeze Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--freeze_mllm` | `False` | Freeze entire VLM backbone |
| `--freeze_llm` | `False` | Freeze language model only |
| `--freeze_vit` | `False` | Freeze ViT vision encoder |
| `--freeze_vit_mlp` | `False` | Freeze ViT MLP layers only |

## MPG Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_mpg` | `True` | Enable MPG enhancement |
| `--mpg_lambda` | `0.1` | MPG residual strength |
| `--mpg_num_projections` | `32` | Sliced Wasserstein projections |
| `--mpg_refinement_iters` | `1` | Inference refinement iterations |
| `--mpg_gate_temperature` | `1.0` | Gate temperature |
| `--mpg_use_stop_gradient` | `True` | Stop gradient on gate |

## RTC Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_training_time_rtc` | `False` | Enable Training-Time RTC |
| `--simulated_delay` | `0` | Max simulated delay steps |
| `--rtc_delay_exp_weight` | `True` | Exponential delay weighting |
| `--use_inference_prefix_overwrite` | `True` | Enable prefix locking |

## Multi-GPU Training

```bash
# 4 GPUs on single node
torchrun --nnodes=1 --nproc_per_node=4 --master_port=29106 \
  BeingH/train/train.py ...

# 8 GPUs across 2 nodes
torchrun --nnodes=2 --nproc_per_node=4 --master_addr=<master_ip> \
  BeingH/train/train.py ...
```

## Monitoring

Training logs are saved to `${OUTPUT_DIR}/training.log`:

```bash
tail -f /path/to/checkpoint/training.log
```

---

## Cross-Embodiment Training

To train on multiple robot embodiments simultaneously, use a cross-embodiment YAML config.

### Enabling Merged Metadata

**Critical:** For cross-embodiment inference, you must enable `--save_merged_metadata True` during training. This saves hierarchical metadata with task-specific and embodiment-merged statistics.

```bash
--save_merged_metadata True
```

The metadata file will be saved at `<checkpoint>/experiment_cfg/<dataset_name>_metadata.json`.

### YAML Configuration Format

See `configs/posttrain/cross-embodiment/libero_robocasa.yaml` for a complete example.

**Required YAML Fields:**

| Field | Description |
|-------|-------------|
| `dataset_names` | List of dataset names (must be registered in `dataset_info.py`) |
| `data_config_names` | List of DataConfig names (one per dataset) |
| `embodiment_tags` | List of embodiment identifiers (one per dataset) |

**Optional YAML Fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `sampling_strategy` | `"step"` | `"step"` or `"trajectory"` sampling |
| `video_backend` | `"torchvision_av"` | Video decoding backend |
| `stats_level` | `"auto"` | Statistics level for normalization |
| `frame_step_size` | `[1, ...]` | Frame sampling stride per dataset |
| `num_used_episodes_per_task` | `[-1, ...]` | Episode limit per task (-1 = all) |
| `weight` | `1` | Dataset sampling weight |

### Inference After Cross-Embodiment Training

After training, use the saved metadata for inference:

```python
policy = BeingHPolicy(
    model_path="<checkpoint>",
    dataset_name="uni_posttrain",              # Must match YAML group name
    metadata_variant="<task-or-embodiment>",   # Select variant
    stats_selection_mode="task",               # or "embodiment"
    # ...
)
```

See [Inference Guide](inference.md#cross-embodiment-metadata) for details on metadata selection.
