from __future__ import annotations
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
CAMERAS = (('cam_high', 'head_camera', (256, 320)), ('cam_left_wrist', 'left_camera', (224, 224)), ('cam_right_wrist', 'right_camera', (224, 224)))

def prepare_observation(observation: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    per_cam, depths, matrices = ({}, {}, [])
    for slot, key, hw in CAMERAS:
        camera = observation['observation'][key]
        rgb = np.asarray(camera['rgb'])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f'Expected RGB uint8 [H,W,3] from {key}, got {rgb.shape}/{rgb.dtype}')
        image = torch.as_tensor(rgb.copy(), device=device).permute(2, 0, 1).float()[None] / 127.5 - 1
        per_cam[slot] = F.interpolate(image, size=hw, mode='bilinear', align_corners=False, antialias=True).to(dtype)
        if 'depth' in camera:
            raise ValueError('This policy uses RGB observations only')
        depth = np.zeros(rgb.shape[:2], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
            raise ValueError(f'Expected same-frame depth [H,W] from {key}, got {depth.shape}')
        matrix = np.asarray(camera.get('intrinsic_cv',np.eye(3,dtype=np.float32)), dtype=np.float32)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or (matrix[1, 1] <= 0):
            raise ValueError(f'Invalid same-grid camera intrinsic matrix from {key}')
        depths[slot] = torch.as_tensor(depth.copy(), device=device)[None, None]
        matrices.append(torch.as_tensor(matrix.copy(), device=device))
    wrists = [F.interpolate(per_cam[slot], size=(128, 160), mode='bilinear', align_corners=False, antialias=True) for slot in ('cam_left_wrist', 'cam_right_wrist')]
    image = torch.cat([per_cam['cam_high'], torch.cat(wrists, dim=-1)], dim=-2)
    return dict(input_image=image, per_cam=per_cam, per_cam_depth=depths, camera_intrinsics=torch.stack(matrices)[None])
