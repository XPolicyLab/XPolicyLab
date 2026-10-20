from __future__ import annotations
import copy
import numpy as np
from .schema import STATE_DIM, ACTION_DIM, fingerprint
from .robotwin_joint14 import INDICES, native_fields, native_affine
PROFILE = 'robotwin_joint14_eef_cleanonly_zscore_v1'
STATS_KEY = 'robotwin_aloha_joint14_eef_cleanonly_zscore_v1'
STATE_POSITION = [*range(0, 3), *range(34, 37)]
STATE_ROTATION = [*range(3, 9), *range(37, 43)]
STATE_EEF = [*range(0, 9), *range(34, 43)]
ACTION_EEF = [*range(0, 6), *range(31, 37)]
SIDE_ROOT = None
DEFINITION = dict(state_position='raw_fl_link6_fr_link6_world_xyz_m', state_rotation='first_two_matrix_columns_identity_unclipped', action_position='future_world_xyz_minus_fixed_observed_anchor_xyz_m', action_rotation='body_R_anchor_T_R_future_window_continuous_rotvec_rad', action_target='anchor_plus_horizon_offset_plus_one_original_frame', horizon=32, terminal_repeated_pose_supervised=False, state_statistics='each_selected_raw_frame_once', action_statistics='each_valid_anchor_and_32_horizon_pair_once', normalization='FP32_(x-mean)/(std+1e-8)_clip_[-5,5]', ddof=0)

def validate_source(source):
    from .robotwin_cleanonly import PROTOCOL
    if source.get('adapter') != 'robotwin' or source.get('supervision_profile') != PROFILE or source.get('stats_key') != STATS_KEY or (source.get('protocol') != 'absolute_joint_relative_eef') or (source.get('coordinate_frame') != 'world') or (source.get('endpoint') != 'fl_link6/fr_link6') or (source.get('fields') != native_fields()) or (not source.get('sidecar_root')) or (source.get('action_label_mode') != 'native_action') or (source.get('action_target_offset_frames') != 1) or (source.get('clean_contract', {}).get('protocol') != PROTOCOL) or (source.get('eef_contract') != DEFINITION):
        raise ValueError('EEF Clean source/codec contract mismatch')
    digest = source.get('native_stats_sha256', '')
    if len(digest) != 64 or any((x not in '0123456789abcdef' for x in digest)):
        raise ValueError('Missing sealed EEF statistics identity')

def validate_group(group):
    from .robotwin_cleanonly import validate_group as validate_base
    if group.get('profile') != PROFILE or group.get('mode') != 'z-score' or group.get('eef_contract') != DEFINITION or (group.get('native_stats_sha256') != fingerprint({k: v for k, v in group.items() if k != 'native_stats_sha256'})):
        raise ValueError('EEF statistics profile/seal mismatch')
    base = group['base_native_group']
    validate_base(base)
    if group.get('clean_contract') != base['clean_contract']:
        raise ValueError('EEF and original Joint14 populations differ')
    n = group['clean_contract']['raw_frames']
    for kind, dim, eef in [('state', STATE_DIM, STATE_EEF), ('action', ACTION_DIM, ACTION_EEF)]:
        for name in ('count', 'mean', 'std', 'q01', 'q99'):
            values = np.asarray(group[kind][name])
            if values.shape != (dim,) or not np.isfinite(values).all():
                raise ValueError('Invalid EEF statistics vector')
            if not np.array_equal(values[INDICES[kind]], np.asarray(base[kind][name])[INDICES[kind]]):
                raise ValueError('Original joint/gripper statistics changed')
        counts = np.asarray(group[kind]['count'])
        active = np.zeros(dim, bool)
        active[INDICES[kind] + eef] = True
        if not np.array_equal(counts > 0, active) or (np.asarray(group[kind]['std']) < 0).any():
            raise ValueError('EEF statistics must cover exactly 32 state/26 action slots')
        expected = n if kind == 'state' else group['eef_valid_action_pairs']
        if expected <= 0 or not np.all(counts[eef] == expected):
            raise ValueError('EEF training population/count mismatch')

def normalize(group, kind, values, valid, *, report=False):
    x = np.asarray(values, np.float32)
    m = np.asarray(valid, bool) & np.isfinite(x)
    known = np.asarray(group[kind]['count']) > 0
    if (m & ~known).any():
        raise ValueError('Valid labels outside sealed EEF statistics')
    scale, offset, _ = native_affine(group['base_native_group'], kind)
    y = np.zeros_like(x)
    joint = INDICES[kind]
    native = np.where(m[..., joint], x[..., joint], 0) * scale[joint] + offset[joint]
    y[..., joint] = np.clip(native, -5, 5)
    eef = STATE_POSITION if kind == 'state' else ACTION_EEF
    mean = np.asarray(group[kind]['mean'], np.float32)[eef]
    denom = np.asarray(group[kind]['std'], np.float32)[eef] + np.float32(1e-08)
    z = (np.where(m[..., eef], x[..., eef], mean) - mean) / denom
    y[..., eef] = np.clip(z, -5, 5)
    if kind == 'state':
        y[..., STATE_ROTATION] = x[..., STATE_ROTATION]
    y[~m] = 0
    clipped = int((m[..., joint] & (np.abs(native) > 5)).sum() + (m[..., eef] & (np.abs(z) > 5)).sum())
    return (y, dict(clipped=clipped, valid=int(m.sum()))) if report else y

def denormalize(group, kind, values, valid=None):
    x = np.asarray(values, np.float32)
    known = np.asarray(group[kind]['count']) > 0
    m = np.broadcast_to(known, x.shape).copy() if valid is None else np.asarray(valid, bool) & known
    y = np.zeros_like(x)
    joint = INDICES[kind]
    scale, offset, _ = native_affine(group['base_native_group'], kind)
    y[..., joint] = (x[..., joint] - offset[joint]) / scale[joint]
    eef = STATE_POSITION if kind == 'state' else ACTION_EEF
    mean = np.asarray(group[kind]['mean'], np.float32)[eef]
    denom = np.asarray(group[kind]['std'], np.float32)[eef] + np.float32(1e-08)
    y[..., eef] = x[..., eef] * denom + mean
    if kind == 'state':
        y[..., STATE_ROTATION] = x[..., STATE_ROTATION]
    y[~m] = 0
    return y

def encode_pose_windows(poses, anchors, horizon=32):
    from .geometry import quaternion_to_matrix, rotation_6d, continuous_rotvec_batch
    poses = np.asarray(poses)
    anchors = np.asarray(anchors, np.int64)
    n = len(poses)
    if poses.shape != (n, 2, 7) or not np.isfinite(poses).all() or horizon != 32:
        raise ValueError('Invalid EEF pose trajectory/horizon')
    if anchors.ndim != 1 or (anchors < 0).any() or (anchors >= n).any():
        raise ValueError('Invalid EEF anchors')
    targets = anchors[:, None] + np.arange(horizon)[None] + 1
    valid = targets < n
    targets = np.minimum(targets, n - 1)
    state = np.zeros((len(anchors), 96), np.float32)
    action = np.zeros((len(anchors), horizon, 90), np.float32)
    for i, (position, rotation, apos, arot) in enumerate([(slice(0, 3), slice(3, 9), slice(0, 3), slice(3, 6)), (slice(34, 37), slice(37, 43), slice(31, 34), slice(34, 37))]):
        R = quaternion_to_matrix(poses[:, i, 3:])
        R0 = R[anchors]
        state[:, position] = poses[anchors, i, :3]
        state[:, rotation] = rotation_6d(R0)
        action[:, :, apos] = poses[targets, i, :3] - poses[anchors, i, :3][:, None]
        action[:, :, arot] = continuous_rotvec_batch(np.swapaxes(R0, -1, -2)[:, None] @ R[targets], valid)
    action[~valid] = 0
    return (state, action, valid)
