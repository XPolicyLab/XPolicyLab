from __future__ import annotations
import logging
import torch
from typing import Any
from diffusers.training_utils import compute_loss_weighting_for_sd3
logger = logging.getLogger(__name__)

def compute_action_loss(model_outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    actions = model_outputs['actions']
    target_action = model_outputs['target_action']
    pred_action = model_outputs['pred_action']
    action_sigmas = model_outputs['action_sigmas']
    action_padding_mask = model_outputs['action_padding_mask']
    sparse_length = model_outputs['sparse_length']
    sample_weights = model_outputs['sample_weights']
    action_loss_w = compute_loss_weighting_for_sd3(weighting_scheme='None', sigmas=action_sigmas)
    action_loss_w = action_loss_w.reshape(-1, 1, 1).repeat(1, 1, actions.size(-1)).to(actions.device, dtype=torch.float32)
    action_loss_map = action_loss_w * (pred_action - target_action).pow(2)
    if not torch.isfinite(action_loss_map).all():
        bad_mask = (~torch.isfinite(action_loss_map)).any(dim=(1, 2))
        bad_indices = bad_mask.nonzero(as_tuple=True)[0].tolist()
        paths = model_outputs['paths']
        logger.error('Non-finite action_loss_map at %d/%d samples (idx=%s paths=%s). actions_finite=%s pred_action_finite=%s action_loss_w_finite=%s action_sigmas=%s', len(bad_indices), action_loss_map.shape[0], bad_indices, [paths[index] for index in bad_indices] if paths is not None else None, torch.isfinite(actions[bad_indices]).all().item(), torch.isfinite(pred_action[bad_indices]).all().item(), torch.isfinite(action_loss_w[bad_indices]).all().item(), action_sigmas[bad_indices].tolist())
    action_loss_map = action_loss_map * action_padding_mask
    sparse_weights = 0.001
    action_loss_map[:, -sparse_length:] *= sparse_weights
    actual_dense_len = action_loss_map.shape[1] - sparse_length
    dense_loss_map = action_loss_map[:, :actual_dense_len]
    dense_part = action_padding_mask[:, :, :16]
    zero_part = torch.zeros_like(action_padding_mask[:, :, 16:])
    action_padding_mask_joint = torch.cat([dense_part, zero_part], dim=-1)
    dense_loss_map_joint = dense_loss_map.clone() * action_padding_mask_joint
    if sample_weights is not None:
        sample_weight_map = sample_weights.to(action_loss_map.device, dtype=action_loss_map.dtype).view(-1, 1, 1)
        action_loss_map = action_loss_map * sample_weight_map
        dense_loss_map = dense_loss_map * sample_weight_map
        dense_loss_map_joint = dense_loss_map_joint * sample_weight_map
        valid_element_count = (action_padding_mask * sample_weight_map).sum() * actions.shape[1]
        valid_dense_count = (action_padding_mask * sample_weight_map).sum() * actual_dense_len
        valid_dense_count_joint = (action_padding_mask_joint * sample_weight_map).sum() * actual_dense_len
    else:
        valid_element_count = action_padding_mask.sum() * actions.shape[1]
        valid_dense_count = action_padding_mask.sum() * actual_dense_len
        valid_dense_count_joint = action_padding_mask_joint.sum() * actual_dense_len
    return {'loss_action': action_loss_map.sum() / (valid_element_count + 1e-06), 'loss_dense_action': dense_loss_map.sum() / (valid_dense_count + 1e-06), 'loss_dense_action_joint': dense_loss_map_joint.sum() / (valid_dense_count_joint + 1e-06)}
