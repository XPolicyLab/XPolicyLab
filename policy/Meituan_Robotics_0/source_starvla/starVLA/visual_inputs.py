"""Inference input metadata helpers; no augmentation/training pipeline."""
from contextlib import contextmanager
import torch


def first_role_mask(image_metas):
    # At episode start the current frame retains its current-camera role.
    return [meta['role'] == 'first' for metas in image_metas for meta in metas]


@contextmanager
def detach_first_vision(model, image_grid_thw, first_mask):
    """Preserve the checkpoint's first-frame vision-feature path during inference."""
    if first_mask is None:
        yield
        return
    if image_grid_thw is None or len(first_mask) != len(image_grid_thw):
        raise ValueError('First-frame mask does not match native image order/grid')
    visual = model.model.visual
    sizes = image_grid_thw.prod(-1) // visual.spatial_merge_size**2
    flags = torch.as_tensor(first_mask, dtype=torch.bool, device=image_grid_thw.device)
    mask = torch.repeat_interleave(flags, sizes.to(torch.long))
    called = False

    def hook(module, args, output):
        nonlocal called
        called = True
        features = output.pooler_output
        if not isinstance(features, torch.Tensor) or len(features) != len(mask):
            raise ValueError('Unsupported Qwen visual output/token alignment')
        output.pooler_output = torch.where(mask.to(features.device)[:, None], features.detach(), features)
        return output

    handle = visual.register_forward_hook(hook)
    try:
        yield
        if not called:
            raise ValueError('Configured first-frame detach never reached visual encoder')
    finally:
        handle.remove()
