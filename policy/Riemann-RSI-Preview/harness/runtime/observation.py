"""Project official observations onto the documented public controller inputs."""
from __future__ import annotations

import numpy as np

STATE_DIMENSIONS = {
    'left_arm_joint_state': 6, 'right_arm_joint_state': 6,
    'left_ee_pose': 7, 'right_ee_pose': 7,
    'left_ee_joint_state': 1, 'right_ee_joint_state': 1,
}
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def public_observation(observation):
    if not isinstance(observation, dict):
        raise ValueError('Official observation must be a dictionary')
    state = observation.get('state', {})
    vision = observation.get('vision', {})
    if not isinstance(state, dict) or not isinstance(vision, dict):
        raise ValueError('Public state and vision must be dictionaries')
    instruction = observation.get('instruction')
    if instruction is not None and (not isinstance(instruction, str) or len(instruction) > 32768):
        raise ValueError('Public instruction must be bounded text')
    for name, dimension in STATE_DIMENSIONS.items():
        value = np.asarray(state.get(name), dtype=float)
        if value.size != dimension or not np.isfinite(value).all():
            raise ValueError(f'Invalid public state field: {name}')
    cameras = {}
    for camera in CAMERAS:
        raw = vision.get(camera)
        image = np.asarray(raw.get('color') if isinstance(raw, dict) else raw)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f'Invalid public RGB camera: {camera}')
        if isinstance(raw, dict):
            cameras[camera] = {'color': raw['color']}
            if 'shape' in raw:
                cameras[camera]['shape'] = list(image.shape)
        else:
            cameras[camera] = raw
    result = {
        'instruction': instruction,
        'state': {name: np.asarray(state[name]).copy() for name in STATE_DIMENSIONS},
        'vision': cameras,
    }
    if 'data_format_version' in observation:
        version = observation['data_format_version']
        if not isinstance(version, (str, int)) or isinstance(version, bool):
            raise ValueError('Public data format version must be scalar metadata')
        result['data_format_version'] = version
    if isinstance(observation.get('action'), dict):
        result['action'] = {}
        for name, value in observation['action'].items():
            if name in STATE_DIMENSIONS:
                current = np.asarray(value, dtype=float)
                if current.size != STATE_DIMENSIONS[name] or not np.isfinite(current).all():
                    raise ValueError('Invalid public action feedback: ' + name)
                result['action'][name] = current.copy()
    return result
