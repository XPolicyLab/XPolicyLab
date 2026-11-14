import torch

def validate_depth_validity(mode):
    if mode not in ('legacy_clip', 'finite_positive'):
        raise ValueError('pointmap_depth_validity must be legacy_clip or finite_positive')
    return mode

def depth_valid_mask(depth, *, mode='legacy_clip', units='mm', max_depth_m=2.0):
    validate_depth_validity(mode)
    if units not in ('mm', 'm'):
        raise ValueError('Pointmap depth units must be mm or m')
    finite = torch.isfinite(depth)
    if mode == 'finite_positive':
        return finite & (depth > 0)
    scale = 1000.0 if units == 'mm' else 1.0
    return finite & (depth > 0.01 * scale) & (depth <= max_depth_m * scale)
