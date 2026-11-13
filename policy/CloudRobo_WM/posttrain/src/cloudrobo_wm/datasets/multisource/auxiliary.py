from cloudrobo_wm.storage import DataPath as Path
import json, torch
import torch.nn.functional as F
from .schema import CAMERAS, within
from cloudrobo_wm.pointmap_validity import depth_valid_mask

def load_auxiliary(sample, adapter, ep, anchor, indices, frame_valid, *, pointmap_depth_validity='legacy_clip'):
    config = adapter.source['auxiliary']
    if config.get('profile') != 'robotwin_native_v1' or adapter.source['adapter'] != 'robotwin' or (not config.get('depth')) or config.get('flow_root'):
        raise ValueError('AS2 accepts only the frozen RoboTwin native RGBD profile')
    return _load_robotwin_depth(sample, adapter, ep, indices, pointmap_depth_validity=pointmap_depth_validity)

def _load_robotwin_depth(sample, adapter, ep, indices, *, pointmap_depth_validity='legacy_clip'):
    eid = int(ep['episode_id'])
    info = adapter.manifest['info']
    chunk = eid // info['chunks_size']
    from cloudrobo_wm.datasets.media import decode_depth_frames
    intrinsic = json.loads((adapter.root / 'meta/camera_intrinsics.json').read_text())
    depths, matrices = ({}, [])
    for cam in CAMERAS:
        key = 'observation.depth_x264rgb.' + cam
        if key not in info['features']:
            return
        path = within(adapter.root, info['video_path'].format(episode_index=eid, episode_chunk=chunk, video_key=key))
        if not path.exists():
            return
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
