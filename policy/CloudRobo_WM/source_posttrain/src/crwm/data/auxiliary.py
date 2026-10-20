from pathlib import Path
import json
import torch
import torch.nn.functional as F
from .schema import CAMERAS, within
from crwm.pointmap_validity import depth_valid_mask

def _load_robotwin_depth(sample, adapter, ep, indices, *, pointmap_depth_validity='legacy_clip'):
    eid = int(ep['episode_id'])
    info = adapter.manifest['info']
    chunk = eid // info['chunks_size']
    from .depth_video import decode_depth_frames
    intrinsic = json.loads((adapter.root / 'meta/camera_intrinsics.json').read_text())
    depths, matrices = ({}, [])
    for cam in CAMERAS:
        key = 'observation.depth_x264rgb.' + cam
        if key not in info['features']:
            return
        path = within(adapter.root, info['video_path'].format(episode_index=eid, episode_chunk=chunk, video_key=key))
        if not path.exists():
            return
        from .media_identity import verify_video_identity
        verify_video_identity(path,{'path':str(path.relative_to(adapter.root))})
        depths[cam] = decode_depth_frames(video_path=path, timestamps=(indices / info['fps']).tolist(), tolerance_s=0.001, vcodec='libx264rgb').to(torch.int32)
        k = intrinsic[cam]
        if depths[cam].shape[-2:] != (k['height'], k['width']):
            raise ValueError('Depth grid and intrinsics differ')
        matrices.append(torch.tensor([[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1.0]]))
    sample['per_cam_depth'] = depths
    sample['camera_intrinsics'] = torch.stack(matrices)
    sample['modality_valid'][3] = True
    canvas = torch.zeros(9, 384, 320, dtype=torch.bool)
    for i, cam in enumerate(CAMERAS):
        d = depths[cam]
        valid = depth_valid_mask(d, mode=pointmap_depth_validity)
        hw = (256, 320) if i == 0 else (128, 160)
        m = F.interpolate(valid[:, None].float(), size=hw, mode='nearest')[:, 0].bool()
        if i == 0:
            canvas[:, :256] = m
        else:
            canvas[:, 256:, (i - 1) * 160:i * 160] = m
    sample['pointmap_pixel_valid'] = canvas

def load_auxiliary(sample, adapter, ep, anchor, indices, frame_valid, *, pointmap_depth_validity='finite_positive'):
    config = adapter.source['auxiliary']
    if adapter.source['adapter'] != 'robotwin' or config != {'depth': True, 'profile': 'robotwin_native_v1'}:
        raise ValueError('Only native RoboTwin depth is supported')
    return _load_robotwin_depth(sample, adapter, ep, indices, pointmap_depth_validity=pointmap_depth_validity)
