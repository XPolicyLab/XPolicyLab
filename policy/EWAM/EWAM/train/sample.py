#!/usr/bin/env python3
"""
Evaluation utilities for EWAM.
Implements inference sampling and metrics computation for validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Suppress matplotlib font manager debug messages
matplotlib.set_loglevel("WARNING")
from PIL import Image
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)


def save_eval_video(pred_video_tensor: torch.Tensor, vae_video_tensor: torch.Tensor,
                    gt_video_tensor: torch.Tensor, eval_dir: str, global_step: int,
                    rank: int = 0, fps: int = 8):
    """
    Save evaluation video as MP4 with predicted, VAE reconstruction, and ground truth frames.

    Args:
        pred_video_tensor: (B, T, C, H, W) predicted video frames [0, 1]
        vae_video_tensor: (B, T, C, H, W) VAE reconstruction video frames [0, 1]
        gt_video_tensor: (B, T, C, H, W) ground truth video frames [0, 1]
        eval_dir: Directory to save the video
        global_step: Current training step (for filename)
        rank: Process rank (for filename)
        fps: Frames per second for the video
    """
    import imageio

    os.makedirs(eval_dir, exist_ok=True)

    # Stitch videos horizontally: [pred | vae | gt]
    stitched = torch.cat([pred_video_tensor, vae_video_tensor, gt_video_tensor], dim=4)  # [B, T, C, H, W*3]

    frames = []
    for t in range(stitched.shape[1]):
        frame = (stitched[0, t].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        frames.append(frame)

    video_path = os.path.join(
        eval_dir,
        f"step_{global_step:06d}_rank_{rank:03d}.mp4",
    )
    writer = imageio.get_writer(video_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    logger.info(f"Saved eval video: {video_path}")


def create_video_grid(predicted_frames: torch.Tensor, ground_truth_frames: torch.Tensor,
                     num_samples: int = 4) -> Image.Image:
    """
    Create a grid visualization comparing predicted and ground truth video frames.

    Args:
        predicted_frames: (B, T, C, H, W) predicted video frames
        ground_truth_frames: (B, T, C, H, W) ground truth video frames
        num_samples: number of samples to visualize

    Returns:
        PIL Image of the comparison grid
    """
    batch_size = min(predicted_frames.shape[0], num_samples)
    num_frames = predicted_frames.shape[1]

    # Convert to numpy (B, T, H, W, C)
    pred_np = predicted_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()
    gt_np = ground_truth_frames[:batch_size].detach().cpu().permute(0, 1, 3, 4, 2).numpy()

    # Clip values to [0, 1] (safety)
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)

    # Create grid: rows are samples, columns are [GT_frame1, GT_frame2, ..., GT_frameN, Pred_frame1, Pred_frame2, ..., Pred_frameN]
    fig, axes = plt.subplots(batch_size, num_frames * 2, figsize=(4 * num_frames * 2, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    elif num_frames * 2 == 1:
        axes = axes.reshape(-1, 1)

    for i in range(batch_size):
        for t in range(num_frames):
            # Ground truth frame
            axes[i, t].imshow(gt_np[i, t])
            axes[i, t].set_title(f'GT Frame {t+1}')
            axes[i, t].axis('off')

            # Predicted frame
            axes[i, t + num_frames].imshow(pred_np[i, t])
            axes[i, t + num_frames].set_title(f'Pred Frame {t+1}')
            axes[i, t + num_frames].axis('off')

    plt.tight_layout()

    # Convert to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    img_array = img_array[:, :, :3]  # Remove alpha channel

    plt.close(fig)

    return Image.fromarray(img_array)


@torch.no_grad()
def inference_sample(model, batch: Dict, config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference to predict future video frames and actions via the model's inference_step.

    Args:
        model: EWAM model (Ewam), may be DDP wrapped
        batch: Input batch containing observations, states, language embeddings, text instructions
        config: Configuration object containing inference parameters

    Returns:
        Tuple of (predicted_frames, predicted_actions)
        - predicted_frames: (B, num_pred_frames, C, H, W) in pixel space [0, 255]
        - predicted_actions: (B, action_chunk_size, action_dim)
    """
    # Handle DDP-wrapped model
    if hasattr(model, 'module'):
        model.module.eval()
    else:
        model.eval()

    # Extract inference parameters from config
    num_inference_steps = config.model.inference.num_inference_timesteps

    # Move batch data to device
    device = next(model.parameters()).device
    first_frame = batch['first_frame'].to(device)  # [B, C, H, W] - conditioning frame
    video_frames = batch['video_frames'].to(device)  # [B, num_video_frames, C, H, W] - target frames

    state = batch['initial_state'].to(device) if 'initial_state' in batch and batch['initial_state'] is not None else None

    language_embeddings = batch['language_embedding']
    if language_embeddings is not None:
        language_embeddings = language_embeddings.to(device)

    vlm_inputs = batch['vlm_inputs']
    if vlm_inputs is not None and len(vlm_inputs) > 0:
        # Move all tensors in the VLM inputs dict to device
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in vlm_inputs.items()}
    else:
        vlm_inputs = None

    with torch.no_grad():
        # Handle DDP-wrapped model: use model.module for inference_step
        inference_model = model.module if hasattr(model, 'module') else model
        predicted_frames, predicted_actions = inference_model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )

    if hasattr(model, 'module'):
        model.module.train()
    else:
        model.train()
    return predicted_frames, predicted_actions


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor,
                           action_dim_is_pad: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute action prediction metrics (MSE and L2 error) in normalized space.

    Args:
        predicted_actions: (B, T, action_dim) predicted actions (normalized [-1,1])
        ground_truth_actions: (B, T, action_dim) ground truth actions (normalized [-1,1])
        action_dim_is_pad: (B, action_dim) bool, True = padded dim (UnifyTo16). If provided,
            pad dims are excluded from MSE/L2 (matches training action-loss masking), so only
            the valid dims count. Without it, all dims are used (backward compatible).

    Returns:
        Dictionary containing MSE and L2 error metrics in normalized space
    """
    # Compute MSE loss
    mse_loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='none').float()  # [B, T, D]

    if action_dim_is_pad is not None:
        # Mask padded dims (matches training action-loss masking):
        # dim_valid [B, D] -> [B, 1, D] broadcast over T, then mean over valid D per (B, T).
        dim_valid = (~action_dim_is_pad.bool()).to(device=mse_loss.device, dtype=mse_loss.dtype)  # [B, D]
        dim_valid_sum = dim_valid.sum(dim=1).clamp(min=1.0)                                       # [B]
        # Per-(B,T) mean over valid dims only
        mse_per_token = (mse_loss * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum.unsqueeze(1)  # [B, T]
        mse_loss_per_sample = mse_per_token.reshape(predicted_actions.shape[0], -1).mean(1)            # [B]
        # L2 (RMSE) over valid dims: sqrt(per-dim MSE) averaged over valid dims
        l2_per_token = (mse_loss.sqrt() * dim_valid.unsqueeze(1)).sum(dim=2) / dim_valid_sum.unsqueeze(1)  # [B, T]
        l2_loss_per_sample = (l2_per_token / (1 + 1e-3)).reshape(predicted_actions.shape[0], -1).mean(1)
    else:
        mse_loss_per_sample = mse_loss.reshape(predicted_actions.shape[0], -1).mean(1)
        # Compute L2 error (RMSE)
        l2_loss = mse_loss.sqrt() / (1 + 1e-3)
        l2_loss_per_sample = l2_loss.reshape(predicted_actions.shape[0], -1).mean(1)

    return {
        'mse_loss': mse_loss_per_sample.mean().item(),
        'l2_error': l2_loss_per_sample.mean().item(),
        'mse_std': mse_loss_per_sample.std().item(),
        'l2_std': l2_loss_per_sample.std().item()
    }


def denormalize_actions(actions: torch.Tensor, dataset_stats: dict) -> torch.Tensor:
    """
    Denormalize actions from [-1,1] back to raw space using dataset stats.
    Inverse of the normalization in libero_lerobot_dataset.py:
        normalized = raw * scale + offset
        raw = (normalized - offset) / scale
    where scale = 2/(max-min), offset = -1 - scale*min

    Args:
        actions: (B, T, action_dim) normalized actions in [-1,1]
        dataset_stats: dict with 'action' key containing 'min' and 'max' lists

    Returns:
        (B, T, action_dim) denormalized actions in raw space
    """
    if dataset_stats is None or 'action' not in dataset_stats:
        return actions

    action_stats = dataset_stats['action']
    if 'min' not in action_stats or 'max' not in action_stats:
        return actions

    dim = actions.shape[-1]
    mins = torch.tensor(action_stats['min'][-dim:], device=actions.device, dtype=actions.dtype)
    maxs = torch.tensor(action_stats['max'][-dim:], device=actions.device, dtype=actions.dtype)

    ranges = maxs - mins
    # For dimensions where range is near-zero, avoid division by zero
    ignore_dim = ranges < 1e-4
    ranges = torch.where(ignore_dim, torch.tensor(1.0, dtype=actions.dtype, device=actions.device), ranges)

    scale = 2.0 / ranges  # (output_max - output_min) / ranges = 2 / ranges
    offset = -1.0 - scale * mins  # output_min - scale * mins = -1 - scale * mins

    # Denormalize: raw = (normalized - offset) / scale
    denormalized = (actions - offset) / scale

    # For near-zero range dims, set to mins (matching the normalization centering)
    denormalized = torch.where(ignore_dim, mins, denormalized)

    return denormalized


def compute_action_metrics_denorm(predicted_actions: torch.Tensor,
                                  ground_truth_actions: torch.Tensor,
                                  dataset_stats: dict,
                                  action_dim_is_pad: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute action prediction metrics in raw (denormalized) space.
    Validation loss computation:
        action_diff = pred_action_denorm - gt_action_denorm
        action_l1 = action_diff.abs().mean()
        action_l2 = action_diff.pow(2).mean()

    Args:
        predicted_actions: (B, T, action_dim) predicted actions (normalized [-1,1])
        ground_truth_actions: (B, T, action_dim) ground truth actions (normalized [-1,1])
        dataset_stats: dict with 'action' key containing 'min' and 'max' lists (raw 7-D)
        action_dim_is_pad: (B, action_dim) bool, True = padded dim. If provided, only valid
            (non-pad) dims are denormalized and scored. dataset_stats is over the raw 7-D
            action, so we extract valid dims first, then denormalize, then compute L1/L2.

    Returns:
        Dictionary containing L1 and L2 metrics in raw (denormalized) space
    """
    if action_dim_is_pad is not None:
        # dataset_stats['action'] min/max are over the raw 7-D action. The valid (non-pad)
        # dims of the 16-D UnifyTo16 layout are exactly those 7 dims (in order), so extract
        # them per-sample before denormalizing. dim_valid: [B, D] -> gather valid cols.
        dim_valid = ~action_dim_is_pad.bool()  # [B, D]
        # Number of valid dims should match the stats length; use the first sample's count.
        # Build a [B, T, n_valid] tensor by selecting valid columns (same valid set per sample
        # for UnifyTo16, but handle per-sample generally via masked select on dim=-1).
        pred_list, gt_list = [], []
        for b in range(predicted_actions.shape[0]):
            valid_idx = dim_valid[b]  # [D]
            pred_list.append(predicted_actions[b, :, valid_idx])  # [T, n_valid]
            gt_list.append(ground_truth_actions[b, :, valid_idx])
        pred_valid = torch.stack(pred_list, dim=0)  # [B, T, n_valid]
        gt_valid = torch.stack(gt_list, dim=0)
        pred_denorm = denormalize_actions(pred_valid.float(), dataset_stats)
        gt_denorm = denormalize_actions(gt_valid.float(), dataset_stats)
    else:
        pred_denorm = denormalize_actions(predicted_actions.float(), dataset_stats)
        gt_denorm = denormalize_actions(ground_truth_actions.float(), dataset_stats)

    action_diff = pred_denorm - gt_denorm
    action_l1 = action_diff.abs().mean().item()
    action_l2 = action_diff.pow(2).mean().item()

    return {
        'l1_denorm': action_l1,
        'l2_denorm': action_l2,
    }


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2,
                   eval_dir: Optional[str] = None, global_step: int = 0,
                   dataset_stats: Optional[dict] = None) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.

    Args:
        dataset_stats: Optional dict with 'action' key containing 'min'/'max' for denormalization.
                       If provided, computes denormalized L1/L2 in raw action space
                       (masked to valid dims when action_dim_is_pad is present in the batch).

    Action metrics are masked over padded dims (UnifyTo16) when batch['action_dim_is_pad'] is
    present, matching the training action-loss masking — so val MSE/L2 reflect only the 7 valid
    LIBERO dims, not the 9 pad dims.
    """
    logger.info(f"Running EWAM evaluation for {num_eval_batches} batches...")

    # Handle DDP-wrapped model
    if hasattr(model, 'module'):
        model.module.eval()
    else:
        model.eval()

    from collections import defaultdict
    metrics = defaultdict(list)
    visual_samples = []

    for step, batch in enumerate(dataloader):
        if step >= num_eval_batches:
            break
        if batch is None:
            continue

        # Inference
        predicted_frames, predicted_actions = inference_sample(model, batch, config)
        gt_frames = batch['video_frames'].to(predicted_frames.device)  # [B, T, C, H, W]
        predicted_frames = predicted_frames.permute(0, 2, 1, 3, 4)     # [B, T, C, H, W]

        # Video metrics (local)
        video_mse = F.mse_loss(predicted_frames, gt_frames, reduction='mean').item()
        metrics['video_mse'].append(video_mse)

        # Action metrics (local)
        if 'action_sequence' in batch and predicted_actions is not None:
            gt_actions = batch['action_sequence'].to(predicted_actions.device)
            # Truncate predicted_actions to match gt_actions length
            # (model outputs action_chunk_size=16, but LeRobot dataset only has 7 actions)
            min_len = min(predicted_actions.shape[1], gt_actions.shape[1])
            pred_trunc = predicted_actions[:, :min_len]
            gt_trunc = gt_actions[:, :min_len]

            # Mask padded dims (UnifyTo16) when present — matches training action-loss masking.
            # action_dim_is_pad: [B, action_dim] bool, True = padded dim.
            action_dim_is_pad = batch.get('action_dim_is_pad', None)
            if action_dim_is_pad is not None:
                action_dim_is_pad = action_dim_is_pad.to(predicted_actions.device)
                # Truncate mask to predicted dim if needed (mask is over full action_dim).
                if action_dim_is_pad.shape[-1] == pred_trunc.shape[-1]:
                    pad_for_metrics = action_dim_is_pad
                else:
                    # pred_trunc dim may differ from full action_dim (e.g. chunk slicing); fall back
                    # to no mask rather than mismatched shapes.
                    n_valid = int((~action_dim_is_pad[0].bool()).sum().item())
                    logging.warning(
                        f"[Val] action_dim_is_pad dim {action_dim_is_pad.shape[-1]} != pred dim "
                        f"{pred_trunc.shape[-1]}; falling back to no mask (expected {n_valid} valid dims)."
                    )
                    pad_for_metrics = None
            else:
                pad_for_metrics = None

            action_metrics = compute_action_metrics(pred_trunc, gt_trunc, pad_for_metrics)
            for key, value in action_metrics.items():
                metrics[f'action_{key}'].append(value)

            # Denormalized metrics (L1/L2 in raw action space, masked to valid dims)
            if dataset_stats is not None:
                denorm_metrics = compute_action_metrics_denorm(pred_trunc, gt_trunc, dataset_stats, pad_for_metrics)
                for key, value in denorm_metrics.items():
                    metrics[f'action_{key}'].append(value)

            # Detailed action statistics: compare predicted vs ground truth magnitudes.
            # When masking, only report valid (non-pad) dims so the per-dim table isn't filled
            # with the 9 pad dims.
            if pad_for_metrics is not None:
                dim_valid = ~pad_for_metrics.bool()  # [B, D]
                # Use sample-0 valid mask (same layout for all samples in UnifyTo16)
                valid_idx = dim_valid[0]
                pred_for_stats = pred_trunc[:, :, valid_idx]
                gt_for_stats = gt_trunc[:, :, valid_idx]
            else:
                pred_for_stats = pred_trunc
                gt_for_stats = gt_trunc
            pred_mean = pred_for_stats.mean(1).detach().cpu().numpy()  # [B, n_valid]
            pred_std = pred_for_stats.std(1).detach().cpu().numpy()
            gt_mean = gt_for_stats.mean(1).detach().cpu().numpy()
            gt_std = gt_for_stats.std(1).detach().cpu().numpy()

            # Per-dim stats (valid dims only)
            pred_abs_mean = pred_for_stats.abs().mean(dim=(0,1)).detach().cpu().numpy()  # [n_valid]
            gt_abs_mean = gt_for_stats.abs().mean(dim=(0,1)).detach().cpu().numpy()
            ratio = pred_abs_mean / (gt_abs_mean + 1e-8)

            if step == 0:
                logger.info(f"  [Val Action Stats] dim  | pred_abs_mean | gt_abs_mean | ratio")
                dim_names = ['dx', 'dy', 'dz', 'dax', 'day', 'daz', 'grip']
                n_report = pred_for_stats.shape[-1]
                for d in range(n_report):
                    name = dim_names[d] if d < len(dim_names) else f'd{d}'
                    logger.info(f"  [Val Action Stats] {name:4s} | {pred_abs_mean[d]:.4f}       | {gt_abs_mean[d]:.4f}       | {ratio[d]:.2f}")
                logger.info(f"  [Val Action Stats] pred overall std: {pred_for_stats.std():.4f}, gt overall std: {gt_for_stats.std():.4f}")
                logger.info(f"  [Val Action Stats] pred range: [{pred_for_stats.min():.4f}, {pred_for_stats.max():.4f}]")
                logger.info(f"  [Val Action Stats] gt range:   [{gt_for_stats.min():.4f}, {gt_for_stats.max():.4f}]")

        # Visualization sample
        if step == 0:
            visual_samples.append({
                'predicted_frames': predicted_frames[:4],
                'ground_truth_frames': gt_frames[:4],
                'predicted_actions': predicted_actions[:4] if predicted_actions is not None else None,
                'ground_truth_actions': batch.get('action_sequence', None)[:4] if batch.get('action_sequence', None) is not None else None
            })

    # Aggregate metrics
    final_metrics = {}
    for key, values in metrics.items():
        if values:
            final_metrics[key] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values))

    if visual_samples:
        sample = visual_samples[0]
        grid_visualization = create_video_grid(
            sample['predicted_frames'],
            sample['ground_truth_frames'],
            num_samples=4
        )
        final_metrics['visualization'] = grid_visualization

    # Save eval video (pred | vae | gt horizontally) if eval_dir is provided
    # For simplicity, use predicted as vae since VAE recon is not directly available
    if eval_dir is not None and visual_samples:
        sample = visual_samples[0]
        pred_frames = sample['predicted_frames'][:1]  # [1, T, C, H, W]
        gt_frames = sample['ground_truth_frames'][:1]
        # Use predicted as VAE recon placeholder (since VAE recon not directly available)
        save_eval_video(
            pred_video_tensor=pred_frames,
            vae_video_tensor=pred_frames,  # placeholder
            gt_video_tensor=gt_frames,
            eval_dir=eval_dir,
            global_step=global_step,
            rank=accelerator.process_index if accelerator is not None else 0,
        )

    # Handle DDP-wrapped model
    if hasattr(model, 'module'):
        model.module.train()
    else:
        model.train()
    return final_metrics


def log_evaluation_metrics(metrics: Dict, writer, accelerator, global_step: int):
    """
    Log evaluation metrics to tensorboard and wandb.

    Args:
        metrics: Dictionary containing evaluation metrics
        writer: TensorBoard writer (can be None)
        accelerator: HuggingFace accelerator
        global_step: Current training step
    """
    if accelerator.is_main_process:
        # Log scalar metrics
        log_dict = {}
        for key, value in metrics.items():
            if key not in ['visualization', 'visual_samples'] and isinstance(value, (int, float)):
                log_dict[f'eval/{key}'] = value

        # Log to accelerator (wandb)
        if log_dict:
            accelerator.log(log_dict, step=global_step)

        # Log to TensorBoard
        if writer is not None:
            # Log scalar metrics to TensorBoard
            for key, value in log_dict.items():
                writer.add_scalar(key, value, global_step)

            # Log grid visualization
            if 'visualization' in metrics:
                img_array = np.array(metrics['visualization']).transpose(2, 0, 1)
                writer.add_image('eval/video_grid', img_array, global_step)

        # Print summary
        logger.info("=== EWAM Evaluation Results ===")
        for key, value in metrics.items():
            if key != 'visualization' and isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")