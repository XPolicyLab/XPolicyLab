#!/usr/bin/env python3
"""
Evaluation utilities for Motus.
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
    Run inference to predict future video frames and actions using UniDiffuser's native inference method.
    
    Args:
        model: UniDiffuser model (LatentActionWorldModel)
        batch: Input batch containing observations, states, language embeddings, text instructions
        config: Configuration object containing inference parameters
        
    Returns:
        Tuple of (predicted_frames, predicted_actions)
        - predicted_frames: (B, num_pred_frames, C, H, W) in pixel space [0, 255]
        - predicted_actions: (B, action_chunk_size, action_dim)
    """
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
    if vlm_inputs is not None:
        # Move all tensors in the VLM inputs dict to device
        vlm_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in vlm_inputs.items()}
    
    with torch.no_grad(): 
        predicted_frames, predicted_actions = model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=num_inference_steps,
            language_embeddings=language_embeddings,
            vlm_inputs=vlm_inputs,
        )
    
    model.train()
    return predicted_frames, predicted_actions


def compute_action_metrics(predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> Dict[str, float]:
    """
    Compute action prediction metrics (MSE and L2 error).
    
    Args:
        predicted_actions: (B, T, action_dim) predicted actions
        ground_truth_actions: (B, T, action_dim) ground truth actions  
        
    Returns:
        Dictionary containing MSE and L2 error metrics
    """
    # Compute MSE loss
    mse_loss = F.mse_loss(predicted_actions, ground_truth_actions, reduction='none').float()
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


@torch.no_grad()
def evaluate_model(model, dataloader, accelerator, config, num_eval_batches: int = 2) -> Dict[str, float]:
    """
    Local-only evaluation: no distributed aggregation; safe for rank0-only evaluation.
    """
    logger.info(f"Running UniDiffuser evaluation for {num_eval_batches} batches...")
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
            gt_actions = batch['action_sequence'][:, :predicted_actions.shape[1]].to(predicted_actions.device)
            action_metrics = compute_action_metrics(predicted_actions, gt_actions)
            for key, value in action_metrics.items():
                metrics[f'action_{key}'].append(value)
        
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
        logger.info("=== UniDiffuser Evaluation Results ===")
        for key, value in metrics.items():
            if key != 'visualization' and isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")