from collections.abc import Mapping
import re
class CapObservationError(ValueError):
    """Raised when a CAP observation contains privileged or GT information."""

_FORBIDDEN_EXACT_KEYS = frozenset({'scene_manager', 'reward_manager', '_label_pose', 'label_pose', '_label_info', 'label_info', 'layout_id', 'seed', 'env_seed', 'depth', 'approximate_depth', 'distance_to_image_plane', 'segmentation', 'semantic_segmentation', 'instance_segmentation', 'ground_truth', 'privileged', 'task_env', 'environment', 'simulator', 'reward', 'task', 'gt', 'gt_handle', 'ground_truth_handle', 'object_pose', 'target_pose', 'cube0', 'cube1', 'cube2'})

_FORBIDDEN_TOKENS = frozenset({'depth', 'segmentation', 'seed', 'privileged', 'simulator', 'reward', 'task', 'gt'})

_ALLOWED_VISION_FIELDS = frozenset({'color', 'intrinsic_matrix', 'extrinsic_matrix', 'shape'})

_ALLOWED_ROBOT_STATE_FIELDS = frozenset({'arm_joint_state', 'ee_joint_state', 'ee_pose', 'tcp_pose', 'delta_ee_pose', 'left_arm_joint_state', 'left_ee_joint_state', 'left_ee_pose', 'left_tcp_pose', 'left_delta_ee_pose', 'right_arm_joint_state', 'right_ee_joint_state', 'right_ee_pose', 'right_tcp_pose', 'right_delta_ee_pose'})

def _normalized_key(key: object) -> str:
    value = str(key).strip().replace('-', '_').replace(' ', '_')
    value = re.sub('([a-z0-9])([A-Z])', '\\1_\\2', value)
    value = re.sub('([A-Z]+)([A-Z][a-z])', '\\1_\\2', value)
    return value.lower()

def _is_forbidden_key(key: object) -> bool:
    normalized = _normalized_key(key)
    if normalized in _FORBIDDEN_EXACT_KEYS:
        return True
    tokens = frozenset((part for part in normalized.split('_') if part))
    return bool(tokens & _FORBIDDEN_TOKENS)

def _guard_privileged_keys(value: object, *, path: str='observation') -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise CapObservationError(f'CAP observation contains forbidden field {path}.{key}')
            _guard_privileged_keys(child, path=f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, tuple, list)):
                _guard_privileged_keys(child, path=f'{path}[{index}]')

def sanitize_cap_observation(official_observation: Mapping[str, object]) -> dict[str, object]:
    """Return the strict RGB/calibration/proprio subset exposed to CAP perception.

    The guard scans the complete input before constructing a new mapping, so a
    forbidden field cannot be hidden under an otherwise ignored metadata key.
    """
    if not isinstance(official_observation, Mapping):
        raise TypeError('official_observation must be a mapping')
    _guard_privileged_keys(official_observation)
    vision = official_observation.get('vision')
    if not isinstance(vision, Mapping) or not vision:
        raise CapObservationError("CAP observation requires a non-empty 'vision' mapping")
    sanitized_vision: dict[str, dict[str, object]] = {}
    has_color = False
    for raw_camera_name, raw_camera in vision.items():
        camera_name = str(raw_camera_name)
        if not camera_name or not isinstance(raw_camera, Mapping):
            raise CapObservationError('each CAP camera observation must be a mapping')
        unknown = {_normalized_key(key) for key in raw_camera} - _ALLOWED_VISION_FIELDS
        if unknown:
            raise CapObservationError(f'CAP camera {camera_name!r} contains unsupported fields: {sorted(unknown)!r}')
        camera: dict[str, object] = {}
        for raw_key, value in raw_camera.items():
            key = _normalized_key(raw_key)
            camera[key] = value
            has_color = has_color or key == 'color'
        sanitized_vision[camera_name] = camera
    if not has_color:
        raise CapObservationError('CAP observation requires at least one color image')
    state = official_observation.get('state')
    if not isinstance(state, Mapping):
        raise CapObservationError("CAP observation requires official robot 'state'")
    sanitized_state: dict[str, object] = {}
    for raw_key, value in state.items():
        key = _normalized_key(raw_key)
        if key not in _ALLOWED_ROBOT_STATE_FIELDS:
            raise CapObservationError(f'CAP robot state contains unsupported field: {raw_key!r}')
        sanitized_state[key] = value
    return {'vision': sanitized_vision, 'state': sanitized_state}