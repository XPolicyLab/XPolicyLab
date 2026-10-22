"""Explicit inference-only adapter for the frozen DINO/STVAE encoder.

Install after semantic checkpoint validation. Only the
STVAE branch is wrapped; the caller's Wan encoder is not changed.
"""

from pathlib import Path
import types

import torch


MODE = 'repeat9_full_stvae'


def configure_current_visual(model, mode='auto'):
    """Select inference behavior without changing checkpoint training identity."""
    if mode not in ('auto', 'legacy', MODE):
        raise ValueError(f'Unknown current visual inference mode: {mode}')
    visual = model.joint_visual_encoder
    conditioning = visual.current_conditioning
    if mode == 'auto':
        mode = MODE if conditioning == 'legacy_clip_prefix' else 'legacy'
    if mode == MODE:
        if conditioning != 'legacy_clip_prefix':
            raise ValueError('repeat9 requires legacy_clip_prefix training identity')
        install_current_repeat9(visual.stvae_encoder)
    return {
        'mode': mode,
        'training_current_conditioning': conditioning,
        'adapter_path': str(Path(__file__).resolve()),
        'runtime_hash_validation': False,
        'repeat_frames': 9 if mode == MODE else None,
        'slice_after': 'full_stvae' if mode == MODE else None,
        'wan_branch_changed': False,
    }


def install_current_repeat9(encoder):
    """Use the complete nine-frame path for online single-frame observations.

    Returns the original bound method for an explicit rollback. Multi-frame
    calls keep their original implementation. Reinstallation is an error so
    nested adapters cannot silently alter the experiment.
    """
    if getattr(encoder, '_current_repeat9_original', None) is not None:
        raise RuntimeError('repeat9 current-frame adapter is already installed')
    original = encoder.encode_raw_video_latents

    def encode(self, rgb, *, num_views=1):
        if rgb.ndim != 5:
            raise ValueError(f'Expected [B,C,T,H,W], got {tuple(rgb.shape)}')
        if rgb.shape[2] != 1:
            return original(rgb, num_views=num_views)
        if rgb.shape[0] == 0:
            raise ValueError('An online observation batch must not be empty')
        current = []
        for sample in rgb.split(1, dim=0):
            repeated = sample.repeat(1, 1, 9, 1, 1)
            full = original(repeated, num_views=num_views)
            if full.ndim != 5 or full.shape[2] != 3:
                raise RuntimeError(
                    'Expected three latent time positions from nine RGB frames; '
                    f'got {tuple(full.shape)}'
                )
            current.append(full[:, :, :1].contiguous())
        return torch.cat(current, dim=0).contiguous()

    encoder._current_repeat9_original = original
    encoder.encode_raw_video_latents = types.MethodType(encode, encoder)
    return original


def restore_current_encoder(encoder):
    """Restore the exact original bound method in the current process."""
    original = getattr(encoder, '_current_repeat9_original', None)
    if original is None:
        raise RuntimeError('repeat9 current-frame adapter is not installed')
    encoder.encode_raw_video_latents = original
    del encoder._current_repeat9_original
