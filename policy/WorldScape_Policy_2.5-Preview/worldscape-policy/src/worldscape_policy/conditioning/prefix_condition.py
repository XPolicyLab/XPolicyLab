"""Direct instruction/summary injection; legacy Event Memory stays untouched."""
import torch
from worldscape_policy.types import Conditioning


def build_prefix_condition(conditioner, features, *, training):
    mask = features.direct_condition_mask
    tokens = features.perception_features
    if mask.dtype != torch.bool or mask.shape != tokens.shape[:2] or not mask.any(-1).all():
        raise ValueError('Joint-prefix features require a nonempty boolean [B,L] mask')
    lengths = mask.sum(-1)
    if not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]):
        raise ValueError('Joint-prefix valid tokens must be packed before padding')
    projected = conditioner.output_norm(conditioner._project(tokens))
    projected = projected * mask[..., None]
    supervise = training and features.semantic_supervision
    return Conditioning(cross_attention_tokens=projected, cross_attention_mask=mask,
        negative_cross_attention_tokens=projected,
        semantic_prediction=projected if supervise else None,
        semantic_mask=mask if supervise else None,
        semantic_target=conditioner.semantic_target_builder(training=training,
            explicit_target=features.semantic_target, fallback_target=None) if supervise else None)
