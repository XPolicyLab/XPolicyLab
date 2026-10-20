from __future__ import annotations
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from crwm.composite_layouts import LayoutSpec, get_layout
from crwm.per_cam_compose import compose_from_per_cam
from crwm.utils.logging_config import get_logger
logger = get_logger(__name__)

class PointmapEncoder(nn.Module):

    def __init__(self, norm_bounds: Optional[dict]=None, max_depth_m: float=2.0, device: str='cpu', torch_dtype: torch.dtype=torch.float32):
        super().__init__()
        self.max_depth_m = float(max_depth_m)
        self._torch_dtype = torch_dtype
        nb = norm_bounds or {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [0.0, 1.5]}
        pt_min = torch.tensor([nb['x'][0], nb['y'][0], nb['z'][0]], dtype=torch.float32).view(1, 3, 1, 1)
        pt_max = torch.tensor([nb['x'][1], nb['y'][1], nb['z'][1]], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('pt_min', pt_min, persistent=False)
        self.register_buffer('pt_max', pt_max, persistent=False)

    def _unproject(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        B, _, H, W = depth.shape
        device, dtype = (depth.device, depth.dtype)
        v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx = K[:, 0, 2].view(B, 1, 1)
        cy = K[:, 1, 2].view(B, 1, 1)
        z = depth[:, 0]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        valid = (z > 0.01) & (z < self.max_depth_m)
        zero = torch.zeros_like(z)
        x = torch.where(valid, x, zero)
        y = torch.where(valid, y, zero)
        z = torch.where(valid, z, zero)
        return torch.stack([x, y, z], dim=1)

    def _normalize_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        pt_min = self.pt_min.to(device=pointmap.device, dtype=pointmap.dtype)
        pt_max = self.pt_max.to(device=pointmap.device, dtype=pointmap.dtype)
        clipped = torch.clamp(pointmap, pt_min, pt_max)
        return 2.0 * (clipped - pt_min) / (pt_max - pt_min) - 1.0

    @torch.no_grad()
    def _per_cam_xyz(self, per_cam_depth: Dict[str, torch.Tensor], camera_intrinsics: torch.Tensor, frame_indices: list, layout: Optional[LayoutSpec]=None, slot_key_map: Optional[Mapping[str, str]]=None, half_pixel: bool=False) -> list:
        if layout is None:
            layout = get_layout('tshape_robotwin_384x320')
        kmap = layout.resolve_slot_key_map(slot_key_map)
        cam_slots = layout.cam_slots()
        cam_names = [kmap[s.key] for s in cam_slots]
        missing = [n for n in cam_names if n not in per_cam_depth]
        if missing:
            raise KeyError(f'per_cam_depth missing cams {missing}; got {list(per_cam_depth)}')
        results = []
        for cam_idx, name in enumerate(cam_names):
            depth = per_cam_depth[name]
            if depth.ndim != 4:
                raise ValueError(f"per_cam_depth['{name}'] must be [B, T, H, W]; got {tuple(depth.shape)}")
            B, T, H, W = depth.shape
            depth_m_full = depth.float() / 1000.0
            if list(frame_indices) == list(range(len(frame_indices))):
                depth_m = depth_m_full[:, :len(frame_indices)]
            else:
                depth_m = depth_m_full[:, frame_indices]
            F_dim = depth_m.shape[1]
            depth_m = depth_m.reshape(B * F_dim, 1, H, W)
            K_cam = (camera_intrinsics[:, cam_idx] if camera_intrinsics.ndim == 4 else camera_intrinsics[cam_idx]).to(device=depth_m.device, dtype=torch.float32)
            if K_cam.ndim == 2:
                K_cam = K_cam[None].expand(B, 3, 3)
            elif K_cam.shape[0] == 1:
                K_cam = K_cam.expand(B, 3, 3)
            if K_cam.shape[0] != B:
                raise ValueError('Camera intrinsics batch does not match depth')
            K_batch = K_cam[:, None].expand(B, F_dim, 3, 3).reshape(B * F_dim, 3, 3)
            if half_pixel:
                h, w = layout.slot_hw()[cam_slots[cam_idx].key]
                sy, sx = (h / H, w / W)
                A = K_batch.new_tensor([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
                K_batch = A @ K_batch
                depth_m = F.interpolate(depth_m, size=(h, w), mode='nearest-exact')
            xyz = self._unproject(depth_m, K_batch)
            xyz = self._normalize_pointmap(xyz)
            results.append(xyz)
        return results

    @torch.no_grad()
    def encode_composite(self, per_cam_depth: Dict[str, torch.Tensor], camera_intrinsics: torch.Tensor, concat_mode: str='tshape_robotwin_384x320', first_frame_only: bool=False, layout: Optional[LayoutSpec]=None, slot_key_map: Optional[Mapping[str, str]]=None, half_pixel: bool=False) -> torch.Tensor:
        if camera_intrinsics is None:
            raise ValueError('`camera_intrinsics` is required for encode_composite.')
        if layout is None:
            layout = get_layout(concat_mode)
        kmap = layout.resolve_slot_key_map(slot_key_map)
        cam_slots = layout.cam_slots()
        if camera_intrinsics.shape[-2:] != (3, 3) or camera_intrinsics.ndim not in (3, 4) or camera_intrinsics.shape[-3] != len(cam_slots):
            raise ValueError(f'Expected camera_intrinsics of shape [{len(cam_slots)}, 3, 3], got {tuple(camera_intrinsics.shape)}')
        cam_names = [kmap[s.key] for s in cam_slots]
        missing = [n for n in cam_names if n not in per_cam_depth]
        if missing:
            raise KeyError(f'per_cam_depth missing cams {missing}; got {list(per_cam_depth)}')
        first = per_cam_depth[cam_names[0]]
        if first.ndim != 4:
            raise ValueError(f"per_cam_depth['{cam_names[0]}'] must be [B, T, H, W]; got {tuple(first.shape)}")
        B, T_src = first.shape[:2]
        frame_indices = [0] if first_frame_only else list(range(T_src))
        F_dim = len(frame_indices)
        xyz_per_cam = self._per_cam_xyz(per_cam_depth=per_cam_depth, camera_intrinsics=camera_intrinsics, frame_indices=frame_indices, layout=layout, slot_key_map=slot_key_map, half_pixel=half_pixel)
        per_cam_5d: Dict[str, torch.Tensor] = {}
        slot_hw_map = layout.slot_hw()
        for cam_idx, slot in enumerate(cam_slots):
            xyz = xyz_per_cam[cam_idx]
            H_dst, W_dst = slot_hw_map[slot.key]
            BF, C, H_src, W_src = xyz.shape
            if (H_src, W_src) != (H_dst, W_dst):
                xyz = F.interpolate(xyz, size=(H_dst, W_dst), mode='nearest')
            per_cam_5d[cam_names[cam_idx]] = xyz.unsqueeze(2)
        composite_flat = compose_from_per_cam(per_cam_5d, layout, slot_key_map=kmap).squeeze(2)
        H_total, W_total = layout.composite_hw
        composite = composite_flat.view(B, F_dim, 3, H_total, W_total).permute(0, 2, 1, 3, 4).contiguous()
        return composite.to(dtype=self._torch_dtype)
