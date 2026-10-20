import copy
import numpy as np
from .schema import STATE_DIM, ACTION_DIM
PROFILE = 'robotwin_joint14_native_v1'
CLEAN_PROFILE = 'robotwin_joint14_cleanonly_v1'
NATIVE_PROFILES = {PROFILE, CLEAN_PROFILE}
NATIVE_STATS_SHA256 = '7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095'
INDICES = {'state': [*range(80, 86), 9, *range(88, 94), 43], 'action': [*range(74, 80), 6, *range(82, 88), 37]}

def native_fields():
    result = {}
    for kind, column in (('state', 'observation.state'), ('action', 'action')):
        result[kind] = {}
        for arm, offset in (('left', 0), ('right', 7)):
            result[kind][arm] = {'joints': dict(column=column, indices=list(range(offset, offset + 6)), command='absolute_position', expected_width=14), 'gripper': dict(column=column, indices=[offset + 6], expected_width=14)}
    return result

def validate_source(source):
    if source.get('supervision_profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
        from .robotwin_eef_cleanonly import validate_source as validate_eef
        return validate_eef(source)
    clean = source.get('supervision_profile') == CLEAN_PROFILE
    if source.get('adapter') != 'robotwin' or source.get('supervision_profile') not in NATIVE_PROFILES or source.get('protocol') != 'absolute_joint_relative_eef' or (source.get('fields') != native_fields()) or (source.get('action_label_mode', 'native_action') != 'native_action') or (not clean and source.get('native_stats_sha256') != NATIVE_STATS_SHA256):
        raise ValueError('Native joint14 profile/field/normalizer contract mismatch')
    if clean:
        from .robotwin_cleanonly import PROTOCOL, STATS_KEY
        digest = source.get('native_stats_sha256', '')
        if len(digest) != 64 or any((c not in '0123456789abcdef' for c in digest)) or digest == NATIVE_STATS_SHA256 or (source.get('stats_key') != STATS_KEY) or (source.get('clean_contract', {}).get('protocol') != PROTOCOL):
            raise ValueError('Invalid clean-only source/statistics provenance')
    if source.get('action_target_offset_frames') != 1:
        raise ValueError('joint14 must preserve the frozen RoboTwin boundary lookahead')

def pack_native(kind, values):
    x = np.asarray(values, dtype=np.float32)
    if x.shape[-1:] != (14,):
        raise ValueError('Expected native RoboTwin 14D joints/grippers')
    dim = STATE_DIM if kind == 'state' else ACTION_DIM
    out = np.zeros((*x.shape[:-1], dim), np.float32)
    valid = np.zeros_like(out, dtype=bool)
    good = np.isfinite(x)
    out[..., INDICES[kind]] = np.where(good, x, 0)
    valid[..., INDICES[kind]] = good
    return (out, valid)

def validate_group(group):
    if group.get('profile') == 'robotwin_joint14_eef_cleanonly_zscore_v1':
        from .robotwin_eef_cleanonly import validate_group as validate_eef
        return validate_eef(group)
    if group.get('profile') == CLEAN_PROFILE:
        from .robotwin_cleanonly import validate_group as validate_clean_group
        return validate_clean_group(group)
    if group.get('profile') != PROFILE or group.get('mode') != 'z-score' or group.get('native_indices') != INDICES or (group.get('native_stats_sha256') != NATIVE_STATS_SHA256):
        raise ValueError('Native joint14 statistics profile mismatch')
    for kind, dim in (('state', STATE_DIM), ('action', ACTION_DIM)):
        active = np.zeros(dim, bool)
        active[INDICES[kind]] = True
        if not np.array_equal(np.asarray(group[kind]['count']) > 0, active):
            raise ValueError('Native joint14 statistics must enable exactly the native 14 slots')

def validate_source_group(source, group):
    validate_source(source)
    validate_group(group)
    if source.get('supervision_profile') != group.get('profile') or source.get('native_stats_sha256') != group.get('native_stats_sha256') or source.get('clean_contract') != group.get('clean_contract'):
        raise ValueError('Joint14 source/statistics provenance mismatch')

def native_affine(group, kind):
    stats = group[kind]
    mean = np.asarray(stats['mean'], np.float32)
    std = np.asarray(stats['std'], np.float32)
    known = np.asarray(stats['count']) > 0
    denom = np.where(known, std + np.float32(1e-08), np.float32(1))
    return (np.float32(1) / denom, -mean / denom, known)
