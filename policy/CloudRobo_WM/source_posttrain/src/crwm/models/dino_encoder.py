from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from crwm.composite_layouts import LayoutSpec, get_layout
from crwm.utils.logging_config import get_logger
from .helpers.dino import select_aux_frame_slots
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
        from crwm.assets import asset_path
        weights_path = str(asset_path('dino'))
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
    def _split_robotwin_frame(frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = get_layout('tshape_robotwin_384x320_uniform')
        d = DinoEncoder._split_composite_frame(frames, layout)
        return (d['slot_top'], d['slot_bl'], d['slot_br'])

    @staticmethod
    def _pool_patches(features: torch.Tensor, patch_grid: int=14, target_grid: Optional[tuple[int, int]]=None, mode: str='avg') -> torch.Tensor:
        if target_grid is None:
            target_grid = (7, 7)
        target_h, target_w = target_grid
        if not (1 <= target_h <= patch_grid and 1 <= target_w <= patch_grid):
            raise ValueError(f'target_grid={target_grid} must have each dim in [1, patch_grid={patch_grid}]')
        if mode not in ('avg', 'bilinear'):
            raise ValueError(f"pool mode must be 'avg' or 'bilinear', got {mode!r}")
        N, P, D = features.shape
        if P != patch_grid * patch_grid:
            raise ValueError(f'Expected P={patch_grid * patch_grid} tokens, got P={P}')
        feat_2d = features.float().view(N, patch_grid, patch_grid, D).permute(0, 3, 1, 2)
        if target_h == patch_grid and target_w == patch_grid:
            pooled = feat_2d
        elif mode == 'bilinear':
            pooled = F.interpolate(feat_2d, size=(target_h, target_w), mode='bilinear', align_corners=False, antialias=True)
        elif patch_grid % target_h == 0 and patch_grid % target_w == 0:
            kh, kw = (patch_grid // target_h, patch_grid // target_w)
            pooled = F.avg_pool2d(feat_2d, kernel_size=(kh, kw), stride=(kh, kw))
        else:
            pooled = F.adaptive_avg_pool2d(feat_2d, output_size=(target_h, target_w))
        return pooled.permute(0, 2, 3, 1).reshape(N, -1, D).to(features.dtype)

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
        return self._extract_patches_eager(images)

    def _extract_patches_eager(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2] != self.dino_size or images.shape[-1] != self.dino_size:
            images = F.interpolate(images, size=(self.dino_size, self.dino_size), mode='bilinear', align_corners=False, antialias=True)
        images = self._normalize_imagenet(images)
        images = images.to(dtype=next(self.model.parameters()).dtype)
        out = self.model.forward_features(images)
        return out[:, self.num_prefix_tokens:]

    def encode_video(self, video: Optional[torch.Tensor]=None, concat_mode: str='tshape_robotwin_384x320_uniform', inference_batch_size: int=27, temporal_stride: int=1, first_frame_only: bool=False, per_cam: Optional[Dict[str, torch.Tensor]]=None, layout: Optional[LayoutSpec]=None, slot_key_map: Optional[Mapping[str, str]]=None, cam_patches: Optional[Sequence[Tuple[int, int]]]=None, pool_mode: str='avg', pixel_unshuffle: int=0, stride_keep_far: bool=False) -> torch.Tensor:
        if per_cam is None and video is None:
            raise ValueError('encode_video requires either `video` or `per_cam`.')
        if layout is None:
            layout = get_layout(concat_mode)
        kmap = layout.resolve_slot_key_map(slot_key_map)
        cam_slots = layout.cam_slots()
        if per_cam is not None:
            proto_key = kmap[cam_slots[0].key]
            if proto_key not in per_cam:
                raise ValueError(f'per_cam missing {proto_key!r} (slot {cam_slots[0].key!r}); got {list(per_cam)!r}')
            head = per_cam[proto_key]
            if head.ndim != 5 or head.shape[1] != 3:
                raise ValueError(f'per_cam[{proto_key!r}] must be [B, 3, T, H, W]; got {tuple(head.shape)}')
            B, _, T, _, _ = head.shape
        else:
            B, C, T, H, W = video.shape
        num_latent_frames = (T - 1) // 4 + 1
        all_frame_indices = [min(4 * i, T - 1) for i in range(num_latent_frames)]
        if first_frame_only:
            frame_indices = [all_frame_indices[0]]
        else:
            slot_ids = select_aux_frame_slots(num_latent_frames, temporal_stride, keep_far=stride_keep_far)
            frame_indices = [all_frame_indices[i] for i in slot_ids]
        F_dim = len(frame_indices)
        if per_cam is not None:
            per_cam_flat: Dict[str, torch.Tensor] = {}
            for slot in cam_slots:
                cam_key = kmap[slot.key]
                if cam_key not in per_cam:
                    raise ValueError(f'per_cam missing {cam_key!r} (slot {slot.key!r}); got {list(per_cam)}')
                v = per_cam[cam_key]
                if v.ndim != 5 or v.shape[1] != 3:
                    raise ValueError(f'per_cam[{cam_key!r}] must be [B, 3, T, H, W]; got {tuple(v.shape)}')
                if frame_indices == list(range(len(frame_indices))):
                    v = v[:, :, :len(frame_indices)]
                else:
                    v = v[:, :, frame_indices]
                Hw, Ww = v.shape[-2:]
                per_cam_flat[slot.key] = v.permute(0, 2, 1, 3, 4).reshape(B * F_dim, 3, Hw, Ww)
            return self._encode_layout(frames=None, B=B, F_dim=F_dim, inference_batch_size=inference_batch_size, layout=layout, per_cam_flat=per_cam_flat, cam_patches=cam_patches, pool_mode=pool_mode, pixel_unshuffle=pixel_unshuffle)
        frames = video[:, :, frame_indices]
        return self._encode_layout(frames, B, F_dim, inference_batch_size, layout=layout, cam_patches=cam_patches, pool_mode=pool_mode, pixel_unshuffle=pixel_unshuffle)

    def encode_frames(self, frames: torch.Tensor, inference_batch_size: int=32, layout: Optional[LayoutSpec]=None, cam_patches: Optional[Sequence[Tuple[int, int]]]=None, pool_mode: str='avg') -> torch.Tensor:
        if layout is None:
            layout = get_layout('tshape_robotwin_384x320')
        cam_dict = self._split_composite_frame(frames, layout)
        cam_slots = layout.cam_slots()
        if cam_patches is None:
            cam_patches = layout.dino_cam_patches
        elif len(cam_patches) != len(cam_slots):
            raise ValueError(f'cam_patches length {len(cam_patches)} != cam slots {len(cam_slots)} for layout {layout.name!r}')
        all_features = []
        for slot, target_grid in zip(cam_slots, cam_patches):
            cam_images = cam_dict[slot.key]
            patches_list = []
            for i in range(0, cam_images.shape[0], inference_batch_size):
                batch = cam_images[i:i + inference_batch_size]
                patches = self._extract_patches(batch)
                if target_grid != (self.patches_per_side, self.patches_per_side):
                    patches = self._pool_patches(patches, patch_grid=self.patches_per_side, target_grid=target_grid, mode=pool_mode)
                patches_list.append(patches)
            cam_features = torch.cat(patches_list, dim=0)
            all_features.append(cam_features)
        return torch.cat(all_features, dim=1)

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
                    patches = self._pool_patches(patches, patch_grid=self.patches_per_side, target_grid=target_grid, mode=pool_mode)
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

    def _encode_robotwin(self, frames: Optional[torch.Tensor], B: int, F_dim: int, inference_batch_size: int, per_cam_flat: Optional[Dict[str, torch.Tensor]]=None, cam_patches: Optional[Sequence[Tuple[int, int]]]=None) -> torch.Tensor:
        layout = get_layout('tshape_robotwin_384x320_uniform')
        if per_cam_flat is not None:
            kmap = layout.default_slot_key_map
            per_cam_flat = {s.key: per_cam_flat[kmap[s.key]] for s in layout.cam_slots()}
        return self._encode_layout(frames=frames, B=B, F_dim=F_dim, inference_batch_size=inference_batch_size, layout=layout, per_cam_flat=per_cam_flat, cam_patches=cam_patches)
