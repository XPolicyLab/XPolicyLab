from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .schema import ACTION, STATE, ACTION_DIM, STATE_DIM, CODEC_VERSION
from .geometry import continuous_rotvec, matrix_to_quaternion, rotation_6d, rotvec_to_matrix

@dataclass(frozen=True)
class GripperCalibration:
    closed: float
    opened: float
    tolerance: float = 0.02

    def __post_init__(self):
        if not np.isfinite([self.closed, self.opened, self.tolerance]).all() or self.closed == self.opened or self.tolerance < 0:
            raise ValueError('Invalid fixed gripper calibration')

    def encode(self, value):
        opening = (np.asarray(value, dtype=np.float64) - self.closed) / (self.opened - self.closed)
        valid = np.isfinite(opening) & (opening >= -self.tolerance) & (opening <= 1 + self.tolerance)
        return (np.where(valid, 2 * np.clip(opening, 0, 1) - 1, 0).astype(np.float32), valid)

    def decode(self, value):
        return self.closed + (np.clip(value, -1, 1) + 1) * 0.5 * (self.opened - self.closed)

def _field(fields, name, leading):
    if name not in fields:
        return (None, None)
    value = np.asarray(fields[name], dtype=np.float64)
    if value.ndim == len(leading):
        value = value[..., None]
    if value.shape[:len(leading)] != leading:
        raise ValueError(f'{name}: wrong leading shape {value.shape}, expected {leading}')
    valid = np.isfinite(value)
    override = fields.get('valid', {}).get(name)
    if override is not None:
        override = np.asarray(override, bool)
        while override.ndim < value.ndim:
            override = override[..., None]
        valid &= np.broadcast_to(override, value.shape)
    return (np.where(valid, value, 0), valid)

def _put(out, mask, layout, name, value, valid):
    if value is None:
        return
    sl = layout[name]
    if name.endswith(('.position', '.rotation', '.gripper')) and value.shape[-1] != sl.stop - sl.start:
        raise ValueError(f'{name} requires exactly {sl.stop - sl.start} dimensions')
    if value.shape[-1] > sl.stop - sl.start:
        raise ValueError(f'{name} exceeds schema capacity: {value.shape[-1]}')
    if value.shape[:-1] != out.shape[:-1]:
        raise ValueError(f'{name}: shape {value.shape} incompatible with {out.shape}')
    dst = slice(sl.start, sl.start + value.shape[-1])
    out[..., dst] = np.where(valid, value, 0)
    mask[..., dst] = valid

class ActionCodec:
    version = CODEC_VERSION

    def __init__(self, protocol='absolute_joint_relative_eef'):
        if protocol not in {'absolute_joint_relative_eef', 'controller_delta', 'video_only'}:
            raise ValueError(f'Unknown action protocol {protocol}')
        self.protocol = protocol

    def encode(self, state_fields, action_fields, horizon, time_valid=None):
        state, sm = (np.zeros(STATE_DIM, np.float32), np.zeros(STATE_DIM, bool))
        action, am = (np.zeros((horizon, ACTION_DIM), np.float32), np.zeros((horizon, ACTION_DIM), bool))
        anchors = np.zeros((2, 7), np.float32)
        anchors[:, 3] = 1
        anchor_valid = np.zeros((2, 2), bool)
        for i, arm in enumerate(('left', 'right')):
            s, a = (state_fields.get(arm, {}), action_fields.get(arm, {}))
            p0, pv0 = _field(s, 'position', ())
            R0, rv0 = _field(s, 'rotation', ())
            if p0 is not None:
                if p0.shape != (3,):
                    raise ValueError('Anchor position must be [3]')
                _put(state, sm, STATE, arm + '.position', p0, pv0)
                anchors[i, :3] = p0
                anchor_valid[i, 0] = pv0.all()
            if R0 is not None:
                if R0.shape != (3, 3):
                    raise ValueError('Anchor rotation must be [3,3]')
                good = rv0.all()
                if good:
                    anchors[i, 3:] = matrix_to_quaternion(R0)
                anchor_valid[i, 1] = good
                _put(state, sm, STATE, arm + '.rotation', rotation_6d(R0), np.full(6, good))
            for name in ('joints', 'hand', 'gripper'):
                sv, mv = _field(s, name, ())
                _put(state, sm, STATE, arm + '.' + name, sv, mv)
                av, mv = _field(a, name, (horizon,))
                _put(action, am, ACTION, arm + '.' + name, av, mv)
            p, pv = _field(a, 'position', (horizon,))
            if p is not None:
                if p.shape != (horizon, 3):
                    raise ValueError('Action position must be [T,3]')
                if self.protocol == 'controller_delta':
                    _put(action, am, ACTION, arm + '.position', p, pv)
                elif p0 is not None:
                    _put(action, am, ACTION, arm + '.position', p - p0, pv & pv0)
            if self.protocol == 'controller_delta':
                r, valid = _field(a, 'rotvec', (horizon,))
                if r is not None:
                    valid = np.broadcast_to(valid.all(-1, keepdims=True), r.shape)
                    _put(action, am, ACTION, arm + '.rotation', r, valid)
            else:
                R, valid = _field(a, 'rotation', (horizon,))
                if R is not None and R0 is not None and anchor_valid[i, 1]:
                    if R.shape != (horizon, 3, 3):
                        raise ValueError('Action rotations must be [T,3,3]')
                    good = valid.all(axis=(-1, -2))
                    if time_valid is not None:
                        good &= np.asarray(time_valid, bool)
                    r = continuous_rotvec(np.swapaxes(R0, -1, -2) @ R, good)
                    _put(action, am, ACTION, arm + '.rotation', r, np.broadcast_to(good[:, None], r.shape))
        for fields, out, mask, layout, leading in ((state_fields, state, sm, STATE, ()), (action_fields, action, am, ACTION, (horizon,))):
            value, valid = _field(fields, 'base', leading)
            _put(out, mask, layout, 'base', value, valid)
        if time_valid is not None:
            am &= np.asarray(time_valid, bool)[:, None]
        if self.protocol == 'video_only':
            sm[:] = False
            am[:] = False
        state[~sm], action[~am] = (0, 0)
        return {'state': state, 'state_valid': sm, 'action': action, 'action_valid': am, 'anchor_pose': anchors, 'anchor_valid': anchor_valid}

    def decode_eef(self, action, anchor_pose):
        from .geometry import quaternion_to_matrix
        if self.protocol != 'absolute_joint_relative_eef':
            raise ValueError('EEF reconstruction requires chunk-relative EEF protocol')
        action = np.asarray(action)
        anchor = np.asarray(anchor_pose)
        if action.shape[-1] != ACTION_DIM or anchor.shape != (2, 7):
            raise ValueError('Wrong action or fixed anchor shape')
        result = {}
        for i, arm in enumerate(('left', 'right')):
            p = action[..., ACTION[arm + '.position']] + anchor[i, :3]
            R = quaternion_to_matrix(anchor[i, 3:]) @ rotvec_to_matrix(action[..., ACTION[arm + '.rotation']])
            result[arm] = np.concatenate((p, matrix_to_quaternion(R)), -1)
        return result

    def decode_robotwin(self, action, *, mode='joint', anchor_pose=None):
        action = np.asarray(action)
        if action.shape[-1] != ACTION_DIM or not np.isfinite(action).all():
            raise ValueError('Expected finite 90D actions')
        if mode not in {'joint', 'eef'}:
            raise ValueError('Control mode must be joint/eef')
        poses = self.decode_eef(action, anchor_pose) if mode == 'eef' else None
        fields = []
        for arm in ('left', 'right'):
            fields.append(action[..., ACTION[arm + '.joints']][..., :6] if poses is None else poses[arm])
            fields.append(GripperCalibration(0, 1).decode(action[..., ACTION[arm + '.gripper']]))
        return np.concatenate(fields, -1).astype(np.float32)
