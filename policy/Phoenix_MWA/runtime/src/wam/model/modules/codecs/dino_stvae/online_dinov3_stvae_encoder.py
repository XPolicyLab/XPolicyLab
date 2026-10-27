"""Frozen DINO110k -> STVAE mean, without the legacy temporal pooling path.

Each independent clip resets temporal history. Cameras are independent batch
items inside STVAE, then restored to the horizontal FastWAM transport layout.
STVAE weights remain FP32 even when a parent policy calls ``to(bfloat16)``.
"""
from __future__ import annotations
from contextlib import nullcontext
from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from .online_dinov3_distilled_encoder import DEFAULT_DINOV3_SOURCE_COMMIT, OnlineDinov3DistilledEncoder
from .stvae import FORMAT_VERSION, MODEL_TYPE, STVAE
_CONFIG_KEYS = {'input_dim', 'latent_dim', 'num_heads', 'num_layers', 'intermediate_size', 'dropout', 'temporal_hidden_dim', 'gradient_checkpointing'}
_NUMPY_RNG_GLOBALS = {'numpy.core.multiarray._reconstruct', 'numpy.ndarray', 'numpy.dtype'}

def checkpoint_sha256(path: str | Path) -> str:
    Path(path).expanduser().resolve(strict=True)
    return None

def load_stvae_checkpoint(path: str | Path, expected_step: int | None=None) -> tuple[STVAE, dict[str, Any]]:
    """Strictly load a model-only or training STVAE checkpoint on CPU.

    Training checkpoints contain NumPy's MT19937 RNG array. Only its inspected
    container constructors are allowlisted; unrestricted pickle is never used.
    The optimizer and RNG state are discarded after reading the model payload.
    """
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    digest = checkpoint_sha256(checkpoint_path)
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path))
    if unsafe - _NUMPY_RNG_GLOBALS:
        raise ValueError(f'Unsupported STVAE checkpoint globals: {sorted(unsafe - _NUMPY_RNG_GLOBALS)}')
    allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('STVAE checkpoint must contain a dictionary')
    if payload.get('model_type') != MODEL_TYPE or payload.get('format_version') != FORMAT_VERSION:
        raise ValueError('Not a supported STVAE checkpoint; spatial SVAE/Wan VAE are incompatible')
    # expected_step is accepted for old configs only; progress is not architecture.
    config = payload.get('model_config')
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError('STVAE checkpoint model_config keys do not match the architecture')
    state = payload.get('state_dict')
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing STVAE state_dict')
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f'Expected FP32 STVAE tensor: {name}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite STVAE tensor: {name}')
    if 'input_std' not in state or not (state['input_std'] > 0).all():
        raise ValueError('STVAE input_std must be positive')
    model = STVAE(**config)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    provenance = payload.get('provenance')
    if not isinstance(provenance, dict):
        raise ValueError('Missing STVAE teacher provenance')
    teacher_digest = provenance.get('teacher_sha256')
    metadata = {'stvae_checkpoint_path': str(checkpoint_path), 'stvae_checkpoint_sha256': digest, 'stvae_step': payload.get('step'), 'model_type': MODEL_TYPE, 'format_version': FORMAT_VERSION, 'model_config': dict(config), 'teacher_checkpoint_recorded': provenance.get('teacher_checkpoint'), 'teacher_sha256': teacher_digest, 'checkpoint_provenance': provenance, 'calibration': payload.get('calibration'), 'posterior': 'mean', 'extra_latent_scaling': 'none', 'input_normalization': 'checkpoint_input_mean_std_fp32', 'input_stats': {name: {'channels': int(value.numel()), 'min': float(value.min()), 'max': float(value.max())} for name, value in (('mean', model.input_mean), ('std', model.input_std))}}
    return (model, metadata)

class OnlineDinov3STVAEEncoder(nn.Module):
    """Encode RGB [B,3,1+4k,H,V*W] as [B,48,1+k,16,V*16]."""
    input_size_multiple = 16
    temporal_compression_factor = 4
    embed_dim = 48

    def __init__(self, dino_kwargs: dict[str, Any], stvae_checkpoint_path: str, expected_stvae_step: int | None=None):
        super().__init__()
        kwargs = dict(dino_kwargs)
        kwargs['verify_source_commit'] = False
        required = {'checkpoint_state': 'ema_student', 'output_source': 'backbone', 'projection_mode': 'raw', 'image_size': 256, 'patch_size': 16, 'strict_load': True, 'verify_source_commit': False, 'dinov3_source_commit': DEFAULT_DINOV3_SOURCE_COMMIT, 'torch_dtype': torch.bfloat16}
        for name, expected in required.items():
            if name in kwargs and kwargs[name] != expected:
                raise ValueError(f'STVAE teacher requires {name}={expected!r}, got {kwargs[name]!r}')
            kwargs[name] = expected
        if not kwargs.get('checkpoint_path'):
            raise ValueError('An explicit DINO checkpoint_path is required for STVAE provenance')
        if kwargs.get('feature_norm_state_path') or kwargs.get('require_feature_norm', False):
            raise ValueError('STVAE requires unnormalized raw DINO features')
        self.stvae, metadata = load_stvae_checkpoint(stvae_checkpoint_path, expected_stvae_step)
        if (self.stvae.input_dim, self.stvae.latent_dim) != (1024, 48):
            raise ValueError('This encoder requires the 1024 -> 48 STVAE architecture')
        teacher_path = Path(kwargs['checkpoint_path']).expanduser().resolve(strict=True)
        teacher_digest = checkpoint_sha256(teacher_path)
        kwargs['checkpoint_path'] = str(teacher_path)
        self.dino = OnlineDinov3DistilledEncoder(**kwargs)
        self.dino.to(dtype=torch.bfloat16)
        self.metadata = {**metadata, 'teacher_checkpoint_path': str(teacher_path), 'teacher_identity_verified': False, 'hash_validation': False, 'teacher_output_source': 'backbone', 'teacher_checkpoint_state': 'ema_student', 'dino_singleton_policy': 'duplicate2_take_first', 'dino_singleton_reason': 'BF16 singleton attention numeric path differs from multi-frame batches', 'dinov3_source_commit': kwargs.get('dinov3_source_commit', DEFAULT_DINOV3_SOURCE_COMMIT), 'teacher_preprocessing': {'rgb_input': 'minus_one_to_one_cast_bfloat16_before_preprocess', 'per_camera_resize': [256, 256], 'resize_mode': 'bicubic', 'align_corners': False, 'imagenet_mean': [0.485, 0.456, 0.406], 'imagenet_std': [0.229, 0.224, 0.225], 'patch_tokens': 'x_norm_patchtokens'}, 'dino_dtype': 'bfloat16', 'stvae_parameter_dtype': 'float32', 'cuda_autocast_dtype': 'bfloat16', 'spatial_grid_per_view': [16, 16], 'temporal_compression': 4, 'temporal_policy': 'independent_window_first_frame_preserved', 'camera_policy': 'independent_batch_items_then_width_concat', 'stvae_batch_policy': 'per_window_views_as_batch'}
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode: bool=True):
        super().train(False)
        return self

    def _apply(self, fn, recurse: bool=True):
        for module, dtype in ((self.dino, torch.bfloat16), (self.stvae, torch.float32)):

            def preserve_precision(tensor, target_dtype=dtype):
                probe = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
                return tensor.to(device=probe.device, dtype=target_dtype if tensor.is_floating_point() else tensor.dtype)
            module._apply(preserve_precision, recurse=recurse)
        return self

    @torch.no_grad()
    def encode_raw_video_latents(self, torch_rgb_minus11_bcthw: torch.Tensor, *, num_views: int=1) -> torch.Tensor:
        rgb = torch_rgb_minus11_bcthw
        if rgb.ndim != 5 or rgb.shape[1] != 3 or min(rgb.shape) < 1 or (not rgb.is_floating_point()):
            raise ValueError('Expected floating RGB [B,3,T,H,V*W] in [-1,1]')
        if isinstance(num_views, bool) or int(num_views) != num_views or num_views < 1:
            raise ValueError('num_views must be a positive integer')
        views = int(num_views)
        if rgb.shape[-1] % views or (rgb.shape[2] - 1) % 4:
            raise ValueError('Expected divisible camera width and T=1+4k')
        if not torch.isfinite(rgb).all() or rgb.min() < -1.001 or rgb.max() > 1.001:
            raise ValueError('RGB must be finite and in [-1,1]')
        device = next(self.stvae.parameters()).device
        if next(self.dino.parameters()).device != device:
            raise ValueError('DINO and STVAE must be on the same device')
        self.train(False)
        self.requires_grad_(False)
        rgb = rgb.to(device=device, dtype=torch.bfloat16)
        autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with autocast:
            singleton = rgb.shape[0] * rgb.shape[2] == 1
            dino_rgb = rgb.repeat(1, 1, 2, 1, 1) if singleton else rgb
            raw = self.dino.encode_raw_video_latents(dino_rgb, num_views=views)
            if singleton:
                raw = raw[:, :, :1].contiguous()
            b, _, t, _, _ = raw.shape
            if tuple(raw.shape) != (rgb.shape[0], 1024, rgb.shape[2], 16, 16 * views):
                raise ValueError(f'Unexpected unpooled DINO shape: {tuple(raw.shape)}')
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite unpooled DINO features')
            raw_views = raw.reshape(b, 1024, t, 16, views, 16).permute(0, 4, 1, 2, 3, 5)
            raw_views = raw_views.reshape(b * views, 1024, t, 16, 16).float().contiguous()
            mean = torch.cat([self.stvae.encode_mean(raw_views[index * views:(index + 1) * views]) for index in range(b)], dim=0)
        latent_t = (t - 1) // 4 + 1
        if tuple(mean.shape) != (b * views, 48, latent_t, 16, 16) or not torch.isfinite(mean).all():
            raise ValueError('Invalid STVAE mean shape or nonfinite output')
        result = mean.reshape(b, views, 48, latent_t, 16, 16).permute(0, 2, 3, 4, 1, 5)
        return result.reshape(b, 48, latent_t, 16, views * 16).contiguous()

    @torch.no_grad()
    def encode_current_observation(self, torch_rgb_minus11_bcthw: torch.Tensor, *, num_views: int=1) -> torch.Tensor:
        if torch_rgb_minus11_bcthw.ndim != 5 or torch_rgb_minus11_bcthw.shape[2] != 1:
            raise ValueError('Current-observation encoding requires exactly T=1')
        return self.encode_raw_video_latents(torch_rgb_minus11_bcthw, num_views=num_views)
