import torch
import torch.nn.functional as F
from cloudrobo_wm.pointmap_validity import depth_valid_mask

def observed_pointmap_mask(per_cam_depth, *, max_depth_m=2.0, depth_validity='legacy_clip'):
    cameras = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
    first = per_cam_depth[cameras[0]]
    batch = first.shape[0]
    canvas = torch.ones(batch, 9, 384, 320, device=first.device, dtype=torch.bool)
    for i, camera in enumerate(cameras):
        depth = per_cam_depth[camera]
        if depth.ndim != 4 or depth.shape[:2] != (batch, 1):
            raise ValueError('Observed per-camera depth must have shape [B,1,H,W]')
        valid = depth_valid_mask(depth, mode=depth_validity, max_depth_m=max_depth_m)
        hw = (256, 320) if i == 0 else (128, 160)
        valid = F.interpolate(valid.float(), size=hw, mode='nearest')[:, 0].bool()
        if i == 0:
            canvas[:, 0, :256] = valid
        else:
            canvas[:, 0, 256:, (i - 1) * 160:i * 160] = valid
    return canvas
