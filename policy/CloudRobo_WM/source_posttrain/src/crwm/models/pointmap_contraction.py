from __future__ import annotations
import copy
import math
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from crwm.composite_layouts import get_layout
from crwm.pointmap_validity import depth_valid_mask
from .pointmap_encoder import PointmapEncoder

def validate_pointmap_contraction(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError('pointmap_contraction must be a mapping or null.')
    enabled = value.get('enabled', False)
    if isinstance(enabled, str):
        if enabled.lower() not in {'true', 'false'}:
            raise ValueError('pointmap_contraction.enabled must be true or false.')
        enabled = enabled.lower() == 'true'
    elif enabled is not None and (not isinstance(enabled, bool)):
        raise ValueError('pointmap_contraction.enabled must be true or false.')
    if not enabled:
        return None
    try:
        alpha = float(value.get('alpha', 0.8))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('pointmap_contraction.alpha must be finite and in (0, 1).') from exc
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError('pointmap_contraction.alpha must be finite and in (0, 1).')
    calibrated = value.get('calibrated_intrinsics', False)
    if not isinstance(calibrated, bool):
        raise ValueError('calibrated_intrinsics must be Boolean')
    return {'enabled': True, 'alpha': alpha, **({'calibrated_intrinsics': True} if calibrated else {})}

def create_pointmap_encoder(*, pointmap_contraction=None, **legacy_kwargs):
    resolved = validate_pointmap_contraction(pointmap_contraction)
    if resolved is None:
        return PointmapEncoder(**legacy_kwargs)
    return ContractedPointmapEncoder(alpha=resolved['alpha'], calibrated_intrinsics=resolved.get('calibrated_intrinsics', False), **legacy_kwargs)

class ContractedPointmapEncoder(PointmapEncoder):

    def __init__(self, norm_bounds=None, max_depth_m=2.0, device='cpu', torch_dtype=torch.float32, *, alpha=0.8, calibrated_intrinsics=False):
        self.calibrated_intrinsics = bool(calibrated_intrinsics)
        self.supports_batched_intrinsics = self.calibrated_intrinsics
        resolved = validate_pointmap_contraction({'enabled': True, 'alpha': alpha})
        bounds = norm_bounds if norm_bounds is not None else {'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [0.0, 1.5]}
        try:
            normalized_bounds = {}
            for axis in ('x', 'y', 'z'):
                pair = bounds[axis]
                if len(pair) != 2:
                    raise ValueError
                lower, upper = (float(pair[0]), float(pair[1]))
                if not (math.isfinite(lower) and math.isfinite(upper) and (lower < upper)):
                    raise ValueError
                normalized_bounds[axis] = [lower, upper]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('pointmap norm_bounds must contain finite ordered [min, max] pairs for x, y, z.') from exc
        super().__init__(norm_bounds=normalized_bounds, max_depth_m=max_depth_m, device=device, torch_dtype=torch_dtype)
        if not (torch.isfinite(self.pt_min).all() and torch.isfinite(self.pt_max).all() and (self.pt_max > self.pt_min).all() and torch.isfinite(self.pt_max - self.pt_min).all()):
            raise ValueError('pointmap norm_bounds must remain finite and ordered in float32.')
        self.alpha = resolved['alpha']
        self._norm_bounds = normalized_bounds
        self.depth_validity = 'finite_positive'

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self.pt_min = self.pt_min.new_tensor([self._norm_bounds[axis][0] for axis in ('x', 'y', 'z')], dtype=torch.float32).view(1, 3, 1, 1)
        self.pt_max = self.pt_max.new_tensor([self._norm_bounds[axis][1] for axis in ('x', 'y', 'z')], dtype=torch.float32).view(1, 3, 1, 1)
        return result

    def encoding_metadata(self) -> dict:
        return {'version': 2 if self.calibrated_intrinsics else 1, **({'calibrated_intrinsics': True, 'intrinsics_policy': 'full_K_per_example'} if self.calibrated_intrinsics else {}), 'mode': 'axiswise_linear_rational', 'alpha': self.alpha, 'norm_bounds': copy.deepcopy(self._norm_bounds), 'coordinate_frame': 'local_camera', 'units': 'meters', 'depth_validity': 'finite_positive', 'invalid_fill': [0.0, 0.0, -1.0]}

    def _contract(self, q: torch.Tensor) -> torch.Tensor:
        a = self.alpha
        excess = (q.abs() - 1.0).clamp_min(0.0)
        tail = q.sign() * (1.0 - (1.0 - a) ** 2 / (1.0 - a + a * excess))
        return torch.where(q.abs() <= 1.0, a * q, tail)

    def _unproject(self, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        height, width = depth.shape[-2:]
        v = torch.arange(height, device=depth.device, dtype=torch.float32).view(1, height, 1)
        u = torch.arange(width, device=depth.device, dtype=torch.float32).view(1, 1, width)
        return self._unproject_rays(depth, K, u, v)

    def _unproject_rays(self, depth, K, u, v):
        depth = depth.float()
        K = K.to(device=depth.device, dtype=torch.float32)
        valid = depth_valid_mask(depth, mode=self.depth_validity, units='m')
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        batch, _, height, width = depth.shape
        z = depth[:, 0]
        if self.calibrated_intrinsics:
            pixels = torch.stack((u.expand(batch, height, width), v.expand(batch, height, width), torch.ones_like(z)), dim=1)
            rays = torch.linalg.solve(K, pixels.reshape(batch, 3, -1)).reshape(batch, 3, height, width)
            return rays * (z / rays[:, 2]).unsqueeze(1)
        x = (u - K[:, 0, 2].view(batch, 1, 1)) / K[:, 0, 0].view(batch, 1, 1) * z
        y = (v - K[:, 1, 2].view(batch, 1, 1)) / K[:, 1, 1].view(batch, 1, 1) * z
        return torch.stack([x, y, z], dim=1)

    @torch.no_grad()
    def _per_cam_xyz(self, per_cam_depth, camera_intrinsics, frame_indices, layout=None, slot_key_map=None, half_pixel=False):
        if half_pixel:
            return super()._per_cam_xyz(per_cam_depth, camera_intrinsics, frame_indices, layout=layout, slot_key_map=slot_key_map, half_pixel=True)
        layout = layout if layout is not None else get_layout('tshape_robotwin_384x320')
        kmap = layout.resolve_slot_key_map(slot_key_map)
        results = []
        for cam_idx, slot in enumerate(layout.cam_slots()):
            depth = per_cam_depth[kmap[slot.key]]
            if depth.ndim != 4:
                raise ValueError('Per-camera depth must be [B,T,H,W]')
            B, T, H, W = depth.shape
            depth = depth.float() / 1000.0
            depth = depth[:, :len(frame_indices)] if list(frame_indices) == list(range(len(frame_indices))) else depth[:, frame_indices]
            frames = depth.shape[1]
            depth = depth.reshape(B * frames, 1, H, W)
            K = (camera_intrinsics[:, cam_idx] if camera_intrinsics.ndim == 4 else camera_intrinsics[cam_idx]).to(device=depth.device, dtype=torch.float32)
            if K.ndim == 2:
                K = K[None].expand(B, 3, 3)
            elif K.shape[0] == 1:
                K = K.expand(B, 3, 3)
            if K.shape[0] != B:
                raise ValueError('Camera intrinsics batch does not match depth')
            K = K[:, None].expand(B, frames, 3, 3).reshape(B * frames, 3, 3)
            u = torch.arange(W, device=depth.device, dtype=torch.float32).view(1, 1, 1, W)
            v = torch.arange(H, device=depth.device, dtype=torch.float32).view(1, 1, H, 1)
            if slot.h * slot.w < H * W:
                depth = F.interpolate(depth, size=(slot.h, slot.w), mode='nearest')
                u = F.interpolate(u, size=(1, slot.w), mode='nearest')
                v = F.interpolate(v, size=(slot.h, 1), mode='nearest')
            xyz = self._unproject_rays(depth, K, u[:, 0], v[:, 0])
            results.append(self._normalize_pointmap(xyz))
        return results

    def _normalize_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        pointmap = pointmap.float()
        valid = torch.isfinite(pointmap).all(dim=1, keepdim=True) & (pointmap[:, 2:3] > 0)
        safe_points = torch.where(valid, pointmap, torch.zeros_like(pointmap))
        pt_min = self.pt_min.to(device=pointmap.device, dtype=torch.float32)
        pt_max = self.pt_max.to(device=pointmap.device, dtype=torch.float32)
        scale = (pt_max - pt_min) / 2.0
        center = pt_min + scale
        encoded = self._contract((safe_points - center) / scale)
        invalid_fill = pointmap.new_tensor([0.0, 0.0, -1.0]).view(1, 3, 1, 1)
        return torch.where(valid, encoded, invalid_fill)

    def decode_metric(self, normalized_xyz: torch.Tensor, valid_mask=None):
        if normalized_xyz.ndim < 2 or normalized_xyz.shape[1] != 3:
            raise ValueError('normalized_xyz must have shape [B, 3, ...].')
        if not normalized_xyz.is_floating_point():
            raise TypeError('normalized_xyz must be a floating-point tensor.')
        y = normalized_xyz if normalized_xyz.dtype == torch.float64 else normalized_xyz.float()
        valid = torch.isfinite(y).all(dim=1, keepdim=True) & (y.abs() < 1.0).all(dim=1, keepdim=True)
        if valid_mask is not None:
            if valid_mask.shape != valid.shape or valid_mask.dtype != torch.bool:
                raise ValueError('valid_mask must be Boolean with shape [B, 1, ...].')
            valid = valid & valid_mask.to(device=y.device)
        safe_y = torch.where(valid, y, torch.zeros_like(y))
        a = self.alpha
        tail_abs = safe_y.abs()
        tail_q = safe_y.sign() * (1.0 + (1.0 - a) / a * ((1.0 - a) / (1.0 - tail_abs) - 1.0))
        q = torch.where(safe_y.abs() <= a, safe_y / a, tail_q)
        shape = (1, 3) + (1,) * (y.ndim - 2)
        lower = self.pt_min.to(device=y.device, dtype=y.dtype).reshape(shape)
        upper = self.pt_max.to(device=y.device, dtype=y.dtype).reshape(shape)
        scale = (upper - lower) / 2.0
        metric = lower + scale + scale * q
        valid = valid & torch.isfinite(metric).all(dim=1, keepdim=True) & (metric[:, 2:3] > 0)
        return (torch.where(valid, metric, torch.zeros_like(metric)), valid)

    @torch.no_grad()
    def encode_composite(self, per_cam_depth, camera_intrinsics, concat_mode='tshape_robotwin_384x320', first_frame_only=False, layout=None, slot_key_map=None, *, return_valid_mask=False, half_pixel=False):
        composite = super().encode_composite(per_cam_depth=per_cam_depth, camera_intrinsics=camera_intrinsics, concat_mode=concat_mode, first_frame_only=first_frame_only, layout=layout, slot_key_map=slot_key_map, half_pixel=half_pixel)
        if not return_valid_mask:
            return composite
        layout = layout if layout is not None else get_layout(concat_mode)
        kmap = layout.resolve_slot_key_map(slot_key_map)
        batch, _, frames, height, width = composite.shape
        mask = torch.zeros((batch, 1, frames, height, width), device=composite.device, dtype=torch.bool)
        for slot in layout.cam_slots():
            depth = per_cam_depth[kmap[slot.key]].float()
            if first_frame_only:
                depth = depth[:, :1]
            valid = depth_valid_mask(depth, mode=self.depth_validity)
            flat = valid.reshape(batch * frames, 1, *valid.shape[-2:]).float()
            if flat.shape[-2:] != (slot.h, slot.w):
                flat = F.interpolate(flat, size=(slot.h, slot.w), mode='nearest-exact' if half_pixel else 'nearest')
            slot_mask = flat.reshape(batch, frames, 1, slot.h, slot.w).permute(0, 2, 1, 3, 4).bool()
            mask[:, :, :, slot.top:slot.top + slot.h, slot.left:slot.left + slot.w] = slot_mask
        return (composite, mask)
