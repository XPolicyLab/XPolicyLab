"""Detached standard RGB/proprioception; no environment or precise jaw access."""
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
import numpy as np

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
STATE_KEYS = tuple(side + suffix for side in ('left', 'right')
                   for suffix in ('_arm_joint_state', '_ee_pose', '_ee_joint_state'))


@dataclass(frozen=True)
class Observation:
    HORIZON = 10000
    state: object
    vision: object
    physical_step: int
    evidence: object

    @classmethod
    def create(cls, state, vision, count, summary=''):
        if type(count) is not int or not 0 <= count <= cls.HORIZON:
            raise ValueError('A public observation needs its official control index')
        values, images, evidence = {}, {}, {}
        for key in STATE_KEYS:
            width = 6 if '_arm_' in key else 7 if '_pose' in key else 1
            value = np.asarray(state[key], dtype=float).reshape(-1).copy()
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError('Missing finite own-robot feedback: ' + key)
            value.setflags(write=False)
            values[key] = value
        for camera in CAMERAS:
            image = np.asarray(vision[camera])
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError('Require the standard current 640x480 RGB view')
            image = image.copy()
            image.setflags(write=False)
            images[camera] = image
            evidence[camera] = hashlib.sha256(image.tobytes()).hexdigest()
        return cls(MappingProxyType(values), MappingProxyType(images), count,
                   MappingProxyType(evidence))


def validate_bound_action(command, observation):
    keys = {side + suffix for side in ('left', 'right')
            for suffix in ('_arm_joint_state', '_ee_joint_state')}
    if not isinstance(command, dict) or set(command) != keys:
        raise ValueError('A rule action must command both arms and normalized jaws')
    result = {}
    for key in keys:
        value = np.asarray(command[key], dtype=float)
        width = 6 if '_arm_' in key else 1
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError('Invalid finite rule action: ' + key)
        if width == 1 and not 0. <= value[0] <= 1.:
            raise ValueError('Jaw targets are normalized commands, never widths')
        result[key] = value.tolist()
    return result


def read_finger_joints(*args, **kwargs):
    raise PermissionError('Precise finger positions/contact are not public observations')
