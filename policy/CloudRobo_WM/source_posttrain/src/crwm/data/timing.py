import numpy as np

def time_grids(protocol, anchor, stride, horizon, video_offsets, valid_end):
    grids = {name: np.zeros((n, 8), np.float32) for name, n in (('action', horizon), ('video', 3), ('pointmap', 3), ('dino', 3))}
    masks = {k: np.zeros(len(v), bool) for k, v in grids.items()}
    if not protocol or not protocol.get('verified', False):
        return (grids, masks)
    if not protocol.get('evidence') or not protocol.get('deployment_known', False):
        raise ValueError('Verified timing requires provenance and a deployment-known target schedule')
    dt = float(protocol['period_seconds'])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid physical period')
    action_shift = int(protocol.get('action_target_offset_frames', 0))
    at = np.arange(horizon) * stride + action_shift
    grids['action'][:, 0:2] = at[:, None] * dt
    masks['action'][:] = anchor + at < valid_end
    frames = np.asarray(video_offsets)
    if len(frames) != 9:
        raise ValueError('Timing contract requires nine sampled visual frames')
    groups = [frames[:1], frames[1:5], frames[5:9]]
    for k, group in enumerate(groups):
        pairs = np.repeat(group, 2) * dt
        grids['video'][k, :len(pairs)] = pairs
        masks['video'][k] = np.all(anchor + group < valid_end)
    grids['pointmap'] = grids['video'].copy()
    masks['pointmap'] = masks['video'].copy()
    for k, i in enumerate((0, 4, 8)):
        grids['dino'][k, :2] = frames[i] * dt
        masks['dino'][k] = anchor + frames[i] < valid_end
    return (grids, masks)
