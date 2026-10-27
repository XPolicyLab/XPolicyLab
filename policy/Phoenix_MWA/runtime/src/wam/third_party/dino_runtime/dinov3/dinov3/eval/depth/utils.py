# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("dinov3")


def align_depth_least_square(
    gt_arr: np.ndarray | torch.Tensor,
    pred_arr: np.ndarray | torch.Tensor,
    valid_mask_arr: np.ndarray | torch.Tensor,
    max_resolution=None,
):
    """
    Adapted from Marigold
    https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/util/alignment.py#L8
    """
    ori_shape = pred_arr.shape  # input shape
    dtype = pred_arr.dtype
    if isinstance(pred_arr, torch.Tensor):
        assert isinstance(gt_arr, torch.Tensor) and isinstance(valid_mask_arr, torch.Tensor)
        pred_arr = pred_arr.to(torch.float32)  # unsupported other types
        device = gt_arr.device
        gt_arr = gt_arr.detach().cpu().numpy()
        pred_arr = pred_arr.detach().cpu().numpy()
        valid_mask_arr = valid_mask_arr.detach().cpu().numpy()

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float()).bool().numpy()

    assert gt.shape == pred.shape == valid_mask.shape, f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    try:
        X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
        scale, shift = X
    except np.linalg.LinAlgError:
        scale = 1
        shift = 0
        logger.info(f"Found wrong depth: \n Pred m:{pred_arr.min()} M:{pred_arr.max()} mean: {pred_arr.mean()}")
        logger.info(f"Gt m:{gt_arr.min()} M:{gt_arr.max()} mean: {gt_arr.mean()}")

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)
    if isinstance(aligned_pred, np.ndarray):
        aligned_pred = torch.as_tensor(aligned_pred, dtype=dtype, device=device)
    return aligned_pred, scale, shift


def create_chmv2_mixlog_bins(min_depth, max_depth, n_bins, device):
    """
    Creates mixed log bins for the CHMv2 model.
    Bins are interpolated between linear and log distributions.

    Note: max_depth is divided by 8.0 because the CHMv2 model was trained
    with internally scaled depth values. The scaling is reversed in
    `create_outputs_with_chmv2_mixlog_norm` by multiplying by 8.0.
    """
    scaled_max_depth = max_depth / 8.0
    linear = torch.linspace(min_depth, scaled_max_depth, n_bins, device=device)
    log = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(min_depth)),
            torch.log(torch.tensor(scaled_max_depth)),
            n_bins,
            device=device,
        )
    )
    t = torch.linspace(1.0, 0.0, n_bins, device=device)
    bins = t * log + (1.0 - t) * linear
    return bins


def create_outputs_with_chmv2_mixlog_norm(
    input: torch.Tensor,
    bins: torch.Tensor,
    max_clamp_value: float = 1e-4,
    eps_shift: float = 1e-8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Converts depth bin logits to depth values using mixlog normalization, specifically
    for the CHMv2 model.
    This function implements a "soft-argmax" style depth prediction, where the output
    is a weighted sum of depth bins, with weights derived from the input logits.
    The CHMv2 model outputs values that are 8x smaller than actual depth in meters,
    so we multiply by 8.0 at the end.

    Args:
        input: Raw logits from the decoder head.
        bins: Depth bin centers created by `create_chmv2_mixlog_bins`.
        max_clamp_value: Maximum value for the positive shift
        eps_shift: Epsilon added to shift to prevent division by zero
        eps: Epsilon for numerical stability in division and final clamping

    Returns:
        Depth map in meters (after x8.0 to the outputs)
    """
    y = torch.relu(input)

    # Ensure strictly positive values by adding a small shift
    m = y.amin(dim=1, keepdim=True)
    shift = (-m).clamp_min(0.0).clamp_max(max_clamp_value) + eps_shift
    y_pos = y + shift

    # Normalize to get weights that sum to 1 (linear normalization)
    # Similar to softmax but without the exponential to preserve relative magnitudes
    denom = y_pos.sum(dim=1, keepdim=True)
    denom = torch.nan_to_num(denom, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(eps)
    weights = y_pos / denom  # (N, n_bins, H, W), sums to 1 along dim=1

    # Compute weighted sum of depth bins (soft-argmax style depth prediction)
    bins_broadcast = bins.view(1, -1, 1, 1).clamp_min(eps)  # (1, n_bins, 1, 1) to match the weights shape
    output = (weights * bins_broadcast).sum(dim=1, keepdim=True).clamp_min(eps)  # (N, 1, H, W)

    # Scale back to meters (reverse the /8.0 applied in `create_chmv2_mixlog_bins``)
    output = output * 8.0

    return output


def setup_model_ddp(model: torch.nn.Module, device: torch.device | int):
    model = DDP(model.to(device), device_ids=[device])
    logger.info(f"Model moved to rank {device}")
    return model
