import torch
import torch.distributed as dist
from crwm.data.schema import ACTION, ACTION_DIM
ACTION_LOSS_GROUPS = {'joint': ('left.joints', 'right.joints'), 'gripper': ('left.gripper', 'right.gripper'), 'eef': ('left.position', 'left.rotation', 'right.position', 'right.rotation')}

def valid_sample_statistics(values, valid):
    valid = valid.to(device=values.device, dtype=torch.bool)
    local_sum = torch.where(valid, values, torch.zeros_like(values)).sum()
    count = valid.sum().to(dtype=torch.float32)
    return (local_sum, count)

def reduce_loss_statistics(parts, diagnostics=None):
    counts = torch.stack([count for _, count in parts])
    packed = counts if diagnostics is None else torch.cat((counts, diagnostics.detach().flatten()))
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed)
        world = dist.get_world_size()
    losses = tuple((total * world / count.clamp_min(1) for (total, _), count in zip(parts, packed[:len(parts)])))
    return (losses, packed[len(parts):])

def valid_sample_mean(values, valid):
    losses, _ = reduce_loss_statistics([valid_sample_statistics(values, valid)])
    return losses[0]

def _action_error_and_weight(prediction, target, valid, field_weight, horizon_weight):
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape or target.shape != prediction.shape:
        raise ValueError('Action target/prediction/valid must all have [B,T,D] shape')
    p = torch.where(valid, prediction.float(), 0)
    t = torch.where(valid, target.float(), 0)
    weight = valid.float()
    if field_weight is not None:
        weight = weight * torch.as_tensor(field_weight, device=p.device)[None, None, :]
    if horizon_weight is not None:
        weight = weight * torch.as_tensor(horizon_weight, device=p.device)[None, :, None]
    return ((p - t).square() * weight, weight)

def masked_action_loss(prediction, target, valid, diffusion_weight, field_weight=None, horizon_weight=None, *, return_stats=False):
    valid = valid.to(device=prediction.device, dtype=torch.bool)
    if valid.shape != prediction.shape or target.shape != prediction.shape:
        raise ValueError('Action target/prediction/valid must all have [B,T,D] shape')
    if (~torch.isfinite(target) & valid).any():
        raise ValueError('Nonfinite action target marked valid')
    errors, weight = _action_error_and_weight(prediction, target, valid, field_weight, horizon_weight)
    count = weight.sum((1, 2))
    error = errors.sum((1, 2)) / count.clamp_min(1e-12)
    error = error * torch.as_tensor(diffusion_weight, device=prediction.device).reshape(-1)
    reducer = valid_sample_statistics if return_stats else valid_sample_mean
    return reducer(error, count > 0)

@torch.no_grad()
def action_group_loss_statistics(prediction, target, valid, diffusion_weight, field_weight=None, horizon_weight=None):
    if prediction.shape[-1] != ACTION_DIM:
        raise ValueError('Action group metrics require the S96/A90 action schema')
    errors, weight = _action_error_and_weight(prediction, target, valid, field_weight, horizon_weight)
    diffusion_weight = torch.as_tensor(diffusion_weight, device=prediction.device).reshape(-1)
    parts = []
    for fields in ACTION_LOSS_GROUPS.values():
        indices = [i for field in fields for i in range(ACTION[field].start, ACTION[field].stop)]
        count = weight[..., indices].sum((1, 2))
        error = errors[..., indices].sum((1, 2)) / count.clamp_min(1e-12)
        parts.append(torch.stack(valid_sample_statistics(error * diffusion_weight, count > 0)))
    return torch.stack(parts)
