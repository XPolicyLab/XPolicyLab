from __future__ import annotations
from typing import Dict, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from crwm.composite_layouts import LayoutSpec, get_layout
from crwm.utils.logging_config import get_logger
from crwm.assets import asset_path
logger = get_logger(__name__)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class DinoEncoder(nn.Module):

    def __init__(self, model_name: str='vit_base_patch16_dinov3.lvd1689m', dino_size: int=224, device: str='cpu', torch_dtype: torch.dtype=torch.float32):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError('timm is required for DinoEncoder. Install with: pip install timm')
        self.model_name = model_name
        self.dino_size = dino_size
        logger.info(f'Loading DINO model: {model_name}')
        weights_path = str(asset_path('dino'))
        logger.info('Loading local DINO weights: %s', weights_path)
        pretrained_options = {'pretrained_cfg_overlay': {'file': weights_path, 'hf_hub_id': None, 'url': ''}}
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0, **pretrained_options)
        self.model = self.model.to(device=device, dtype=torch_dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 1)
        self.embed_dim = self.model.embed_dim
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        self.patch_size = patch_size
        self.patches_per_side = dino_size // patch_size
        self.patches_per_frame = self.patches_per_side ** 2
        logger.info(f'  embed_dim={self.embed_dim}, patch_size={self.patch_size}, patches_per_frame={self.patches_per_frame}, num_prefix_tokens={self.num_prefix_tokens}')
        self.register_buffer('_mean', torch.tensor(IMAGENET_MEAN, device=device, dtype=torch_dtype).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('_std', torch.tensor(IMAGENET_STD, device=device, dtype=torch_dtype).view(1, 3, 1, 1), persistent=False)

    def _normalize_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) * 0.5
        return (x - self._mean) / self._std

    @staticmethod
    def _split_composite_frame(frames: torch.Tensor, layout: LayoutSpec) -> Dict[str, torch.Tensor]:
        H, W = layout.composite_hw
        if frames.shape[-2:] != (H, W):
            raise ValueError(f'Layout {layout.name!r} expects composite {(H, W)}, got {tuple(frames.shape[-2:])}')
        out: Dict[str, torch.Tensor] = {}
        for s in layout.cam_slots():
            out[s.key] = frames[:, :, s.top:s.top + s.h, s.left:s.left + s.w]
        return out

    @staticmethod
    def _pixel_unshuffle_patches(features: torch.Tensor, patch_grid: Tuple[int, int], factor: int) -> torch.Tensor:
        N, P, D = features.shape
        h, w = patch_grid
        if P != h * w:
            raise ValueError(f'Expected P={h * w} tokens, got P={P}')
        if h % factor or w % factor:
            raise ValueError(f'pixel_unshuffle factor {factor} must divide grid {patch_grid}')
        x = features.view(N, h, w, D).permute(0, 3, 1, 2)
        x = F.pixel_unshuffle(x, factor)
        return x.permute(0, 2, 3, 1).reshape(N, h // factor * (w // factor), D * factor * factor)

    def _extract_patches(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2] != self.dino_size or images.shape[-1] != self.dino_size:
            images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode='bilinear', align_corners=False, antialias=True)
        images = self._normalize_imagenet(images)
        images = images.to(dtype=next(self.model.parameters()).dtype)
        out = self.model.forward_features(images)
        return out[:, self.num_prefix_tokens:]

    def _encode_layout(self, frames: Optional[torch.Tensor], B: int, F_dim: int, inference_batch_size: int, layout: LayoutSpec, per_cam_flat: Optional[Dict[str, torch.Tensor]]=None, cam_patches: Optional[Sequence[Tuple[int, int]]]=None, pool_mode: str='avg', pixel_unshuffle: int=0) -> torch.Tensor:
        cam_slots = layout.cam_slots()
        if cam_patches is None:
            cam_patches = layout.dino_cam_patches
        elif len(cam_patches) != len(cam_slots):
            raise ValueError(f'cam_patches length {len(cam_patches)} != cam slots {len(cam_slots)} for layout {layout.name!r}')
        if per_cam_flat is not None:
            cam_inputs = [per_cam_flat[s.key] for s in cam_slots]
        else:
            H, W = layout.composite_hw
            frames_flat = frames.permute(0, 2, 1, 3, 4).reshape(B * F_dim, 3, H, W)
            split = self._split_composite_frame(frames_flat, layout)
            cam_inputs = [split[s.key] for s in cam_slots]
        all_features = []
        for cam_images, target_grid in zip(cam_inputs, cam_patches):
            patches_list = []
            for i in range(0, cam_images.shape[0], inference_batch_size):
                batch = cam_images[i:i + inference_batch_size]
                patches = self._extract_patches(batch)
                if target_grid != (self.patches_per_side, self.patches_per_side):
                    raise ValueError('CR-WM folds the native DINO patch grid without pooling')
                patches_list.append(patches)
            cam_features = torch.cat(patches_list, dim=0)
            if pixel_unshuffle:
                cam_features = self._pixel_unshuffle_patches(cam_features, patch_grid=target_grid, factor=pixel_unshuffle)
            all_features.append(cam_features)
        features = torch.cat(all_features, dim=1)
        n_patches_total = features.shape[1]
        feat_dim = features.shape[2]
        features = features.view(B, F_dim, n_patches_total, feat_dim)
        features = features.permute(0, 3, 1, 2).unsqueeze(-1)
        return features

    @torch.no_grad()
    def encode_video(self, video=None, *, per_cam=None, layout=None, slot_key_map=None, cam_patches=None, pool_mode='avg', pixel_unshuffle=0):
        layout = get_layout(layout)
        kmap = layout.resolve_slot_key_map(slot_key_map)
        if per_cam is None:
            if video is None or video.ndim != 5:
                raise ValueError('Expected video [B,3,T,H,W] or per-camera images')
            return self._encode_layout(video[:, :, :1], video.shape[0], 1, 27, layout, cam_patches=cam_patches, pool_mode=pool_mode, pixel_unshuffle=pixel_unshuffle)
        slots = layout.cam_slots()
        B = per_cam[kmap[slots[0].key]].shape[0]
        cams = {}
        for slot in slots:
            image = per_cam[kmap[slot.key]]
            if image.ndim == 5:
                image = image[:, :, 0]
            if image.ndim != 4 or image.shape[:2] != (B, 3):
                raise ValueError('Per-camera RGB must be [B,3,H,W]')
            cams[slot.key] = image
        return self._encode_layout(None, B, 1, 27, layout, per_cam_flat=cams, cam_patches=cam_patches, pool_mode=pool_mode, pixel_unshuffle=pixel_unshuffle)
