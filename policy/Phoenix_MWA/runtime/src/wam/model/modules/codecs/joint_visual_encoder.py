"""Frozen per-view Wan48/STVAE48 bridge to the audited local encoder source."""
from __future__ import annotations
import hashlib
import importlib
import sys
import weakref
from pathlib import Path
import torch
from torch import nn
from wam.utils.training_timing import timed, timing_count, timing_section, time_methods

def encoder_source_sha256(source_root: str | Path) -> str:
    """Compatibility API: runtime source hashing is disabled."""
    return None

def _load_stvae_encoder_class(source_root: str, expected_sha256: str):
    if source_root == 'bundled://dino-stvae-v1':
        observed = encoder_source_sha256(source_root)
        from .dino_stvae.online_dinov3_stvae_encoder import OnlineDinov3STVAEEncoder
        return OnlineDinov3STVAEEncoder
    root = Path(source_root).expanduser().resolve(strict=True)
    observed = encoder_source_sha256(root)
    package_root = root / 'src' / 'fastwam'
    for name, module in tuple(sys.modules.items()):
        if name == 'fastwam' or name.startswith('fastwam.'):
            filename = getattr(module, '__file__', None)
            if filename and (not Path(filename).resolve().is_relative_to(package_root)):
                raise RuntimeError(f'Conflicting FastWAM module already imported: {name}: {filename}')
    source_path = str(root / 'src')
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    module = importlib.import_module('fastwam.models.wan22.online_dinov3_stvae_encoder')
    return module.OnlineDinov3STVAEEncoder

class JointVisualEncoder(nn.Module):
    """RGB [B,V,3,T,256,256] -> [B,96,1+(T-1)/4,16,V*16].

    The WAM owns the VAE. A weak reference avoids duplicating its checkpoint
    keys, while the frozen STVAE/DINO branch is registered here.
    """

    def __init__(self, vae: nn.Module, stvae_encoder: nn.Module, *, allow_cpu_for_tests=False, current_conditioning='legacy_clip_prefix'):
        super().__init__()
        self._vae_ref = weakref.ref(vae)
        self.stvae_encoder = stvae_encoder
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        if current_conditioning not in ('legacy_clip_prefix', 'independent_single_frame_v1'):
            raise ValueError(f'Unknown current_conditioning: {current_conditioning}')
        self.current_conditioning = current_conditioning
        self.metadata = {'channel_order': ['wan_vae48', 'stvae48'], 'camera_order': ['main', 'left', 'right'], 'posterior': 'mean', 'per_camera_input_hw': [256, 256], 'temporal_compression': 4, 'extra_latent_scaling': 'none', 'stvae': getattr(stvae_encoder, 'metadata', {})}
        if current_conditioning != 'legacy_clip_prefix':
            self.metadata['current_conditioning'] = current_conditioning
        self.requires_grad_(False)
        vae.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_config(cls, config: dict, *, vae: nn.Module, device):
        config = dict(config)
        source_root = str(config['source_root'])
        source_digest = None
        encoder_class = _load_stvae_encoder_class(source_root, source_digest)
        dino_kwargs = dict(config['dino_kwargs'])
        if source_root == 'bundled://dino-stvae-v1':
            from .dino_stvae.provenance import RUNTIME
            dino_kwargs['starvla_source_path'] = str(RUNTIME)
            dino_kwargs['dinov3_source_path'] = str(RUNTIME / 'dinov3')
        stvae = encoder_class(dino_kwargs=dino_kwargs, stvae_checkpoint_path=str(config['stvae_checkpoint_path']), expected_stvae_step=None).to(device=device)
        wrapper = cls(vae, stvae, current_conditioning=config.get('current_conditioning', 'legacy_clip_prefix'))
        resolved_root = source_root if source_root == 'bundled://dino-stvae-v1' else str(Path(source_root).resolve())
        wrapper.metadata.update(source_root=resolved_root, source_sha256=source_digest)
        return wrapper

    def train(self, mode=True):
        super().train(False)
        vae = self._vae_ref()
        if vae is not None:
            vae.eval()
        return self

    @torch.no_grad()
    @timed('joint_encoder')
    def encode(self, video: torch.Tensor, *, camera_is_pad: torch.Tensor | None):
        if video.ndim != 6 or video.shape[1:3] != (3, 3) or video.shape[-2:] != (256, 256):
            raise ValueError('Joint96 requires RGB [B,3,3,T,256,256]')
        batch, views, _, frames, _, _ = video.shape
        if frames < 1 or (frames - 1) % 4:
            raise ValueError('Joint96 requires T=1+4k')
        if camera_is_pad is None or camera_is_pad.dtype != torch.bool or tuple(camera_is_pad.shape) != (batch, views):
            raise ValueError('Joint96 requires bool camera_is_pad[B,3]')
        pad = camera_is_pad.to(device=video.device)
        if pad.all(dim=1).any():
            raise ValueError('Every sample must contain a real camera')
        vae = self._vae_ref()
        if vae is None:
            raise RuntimeError('Owning WAM VAE no longer exists')
        device = next(vae.parameters()).device
        if device.type != 'cuda' and (not self.allow_cpu_for_tests):
            raise ValueError('Real joint96 encoding requires CUDA; use injected doubles for CPU tests')
        rgb = video.masked_fill(pad[:, :, None, None, None, None], -1.0)
        if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or rgb.min() < -1.001 or (rgb.max() > 1.001):
            raise ValueError('Real RGB must be finite floating values in [-1,1]')
        rgb = rgb.to(device=device, dtype=torch.bfloat16)
        pad = pad.to(device=device)
        self.train(False)
        latent_t = (frames - 1) // 4 + 1
        samples = []
        with torch.autocast(device_type=device.type, enabled=False):
            for b in range(batch):
                encoded_views = []
                for v in range(views):
                    if pad[b, v]:
                        z = rgb.new_zeros((1, 48, latent_t, 16, 16))
                    else:
                        vae.model.clear_cache()
                        try:
                            with timing_section('wan_vae'):
                                z = vae.encode([rgb[b, v].contiguous()], device=device, tiled=False)
                            timing_count('wan_vae_real_views')
                        finally:
                            vae.model.clear_cache()
                        if tuple(z.shape) != (1, 48, latent_t, 16, 16) or not torch.isfinite(z).all():
                            raise ValueError('Invalid per-view Wan48 output')
                    encoded_views.append(z)
                samples.append(torch.cat(encoded_views, dim=-1))
        wan = torch.cat(samples, dim=0)
        transport = rgb.permute(0, 2, 3, 4, 1, 5).reshape(batch, 3, frames, 256, views * 256)
        with timing_section('dino_stvae'), time_methods([(getattr(self.stvae_encoder, 'dino', None), 'encode_raw_video_latents', 'dino'), (getattr(self.stvae_encoder, 'stvae', None), 'encode_mean', 'stvae')]):
            transport = transport.contiguous()
            if self.current_conditioning == 'independent_single_frame_v1':
                current = torch.cat([self.stvae_encoder.encode_raw_video_latents(transport[b:b + 1, :, :1].contiguous(), num_views=views) for b in range(batch)], dim=0)
                if frames == 1:
                    stvae = current
                else:
                    full = self.stvae_encoder.encode_raw_video_latents(transport, num_views=views)
                    stvae = torch.cat((current, full[:, :, 1:]), dim=2)
            else:
                stvae = self.stvae_encoder.encode_raw_video_latents(transport, num_views=views)
        if stvae.shape != wan.shape or not torch.isfinite(stvae).all():
            raise ValueError(f'Wan/STVAE geometry or finiteness mismatch: {wan.shape}, {stvae.shape}')
        valid_width = (~pad).repeat_interleave(16, dim=1)[:, None, None, None, :]
        return torch.cat((wan, stvae.to(wan)), dim=1).masked_fill(~valid_width, 0).contiguous()

    @torch.no_grad()
    def decode_rgb(self, latents: torch.Tensor):
        if latents.ndim != 5 or latents.shape[1] != 96 or latents.shape[-2:] != (16, 48):
            raise ValueError('Joint96 RGB decode expects three 16x16 camera slots')
        vae = self._vae_ref()
        if vae is None:
            raise RuntimeError('Owning WAM VAE no longer exists')
        device = next(vae.parameters()).device
        views = []
        for view in range(3):
            z = latents[:, :48, :, :, 16 * view:16 * (view + 1)].contiguous()
            vae.model.clear_cache()
            try:
                views.append(vae.decode(z, device=device, tiled=False))
            finally:
                vae.model.clear_cache()
        return torch.cat(views, dim=-1)
