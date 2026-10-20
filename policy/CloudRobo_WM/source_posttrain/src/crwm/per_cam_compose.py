from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple
import torch
import torch.nn.functional as F
from .composite_layouts import LayoutSpec, get_layout, TSHAPE_ROBOTWIN_384X320_NAME
_WRIST_SLOT_HW: Tuple[int, int] = (128, 160)
_HORIZONTAL_PER_CAM_HW: Tuple[int, int] = (224, 224)

def compose_horizontal_from_per_cam(per_cam: Dict[str, torch.Tensor]) -> torch.Tensor:
    head = per_cam['cam_high']
    left = per_cam['cam_left_wrist']
    if head.ndim != 5 or head.shape[1] != 3:
        raise ValueError(f'cam_high must be [B, 3, T, H, W]; got {tuple(head.shape)}')
    if left.ndim != 5 or left.shape[1] != 3:
        raise ValueError(f'cam_left_wrist must be [B, 3, T, H, W]; got {tuple(left.shape)}')
    B, _, T, _, _ = head.shape
    h_target, w_target = _HORIZONTAL_PER_CAM_HW

    def _resize(x: torch.Tensor) -> torch.Tensor:
        _, _, _, H, W = x.shape
        if (H, W) == (h_target, w_target):
            return x
        flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        rs = F.interpolate(flat, size=(h_target, w_target), mode='bilinear', align_corners=False, antialias=True)
        return rs.reshape(B, T, 3, h_target, w_target).permute(0, 2, 1, 3, 4).contiguous()
    head_rs = _resize(head)
    left_rs = _resize(left)
    return torch.cat([head_rs, left_rs], dim=-1)

def compose_from_per_cam(per_cam: Dict[str, torch.Tensor], layout: LayoutSpec, slot_key_map: Optional[Mapping[str, str]]=None) -> torch.Tensor:
    kmap = layout.resolve_slot_key_map(slot_key_map)
    cam_slots = layout.cam_slots()
    if not cam_slots:
        raise ValueError(f'Layout {layout.name!r} has no camera slots.')
    proto_key = kmap[cam_slots[0].key]
    if proto_key not in per_cam:
        raise KeyError(f'per_cam missing camera {proto_key!r} (slot {cam_slots[0].key!r}); got keys {list(per_cam)!r}')
    proto = per_cam[proto_key]
    if proto.ndim != 5 or proto.shape[1] != 3:
        raise ValueError(f'per_cam[{proto_key!r}] must be [B, 3, T, H, W]; got {tuple(proto.shape)}')
    B, _, T = proto.shape[:3]
    H_total, W_total = layout.composite_hw
    composite = torch.full((B, 3, T, H_total, W_total), -1.0, dtype=proto.dtype, device=proto.device)
    for slot in cam_slots:
        cam_key = kmap[slot.key]
        if cam_key not in per_cam:
            raise KeyError(f'per_cam missing camera {cam_key!r} (slot {slot.key!r}); got keys {list(per_cam)!r}')
        x = per_cam[cam_key]
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f'per_cam[{cam_key!r}] must be [B, 3, T, H, W]; got {tuple(x.shape)}')
        if x.shape[0] != B or x.shape[2] != T:
            raise ValueError(f'per_cam[{cam_key!r}] batch/time mismatch: got {tuple(x.shape)}, expected B={B}, T={T}')
        Hx, Wx = (x.shape[-2], x.shape[-1])
        if (Hx, Wx) != (slot.h, slot.w):
            tile_mode = getattr(slot, 'tile_mode', 'resize')
            flat = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, Hx, Wx)
            if tile_mode == 'repeat':
                flat = F.interpolate(flat, size=(slot.h, slot.w), mode='nearest')
            elif tile_mode == 'resize':
                flat = F.interpolate(flat, size=(slot.h, slot.w), mode='bilinear', align_corners=False, antialias=True)
            else:
                raise ValueError(f"Slot {slot.key!r}: unknown tile_mode {tile_mode!r} (expected 'resize' or 'repeat')")
            x = flat.reshape(B, T, 3, slot.h, slot.w).permute(0, 2, 1, 3, 4).contiguous()
        composite[:, :, :, slot.top:slot.top + slot.h, slot.left:slot.left + slot.w] = x
    return composite

def compose_robotwin_from_per_cam(per_cam: Dict[str, torch.Tensor]) -> torch.Tensor:
    layout = get_layout(TSHAPE_ROBOTWIN_384X320_NAME)
    head = per_cam.get('cam_high')
    if head is None or head.ndim != 5 or head.shape[1] != 3 or (head.shape[-2:] != (256, 320)):
        raise ValueError(f'cam_high must be [B, 3, T, 256, 320]; got {(tuple(head.shape) if head is not None else None)}')
    for name in ('cam_left_wrist', 'cam_right_wrist'):
        t = per_cam.get(name)
        if t is None or t.ndim != 5 or t.shape[1] != 3:
            raise ValueError(f'{name} must be [B, 3, T, H, W]; got {(tuple(t.shape) if t is not None else None)}')
    return compose_from_per_cam(per_cam, layout)
