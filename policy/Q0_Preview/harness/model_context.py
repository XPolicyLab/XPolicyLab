"""Compact model display only; raw observations, actions and history stay intact."""
from copy import deepcopy
import json

POSE_FIELDS = (
    'grasp_point', 'tool_axis', 'jaw_axis', 'gripper_pose_points',
    'flange_point', 'fingertip_center', 'distal_pad_bbox_center',
    'home_grasp_point', 'home_tool_axis', 'home_jaw_axis',
)


def _pose_display(value):
    if isinstance(value, float):
        return round(value, 5)
    if isinstance(value, list):
        return [_pose_display(item) for item in value]
    return value


def model_observation(observation, observation_id, checking=False):
    """Preserve unknown fields and diagnostics; round only named pose vectors.

    Five decimals bound coordinate display error to 5 micrometres per component.
    Error residuals, small tolerances, gripper state and budgets are not rounded.
    """
    visible = deepcopy(observation)
    visible.pop('calibration', None)  # Already loaded by vlm_geom from full raw data.
    visible.pop('run_id', None)  # Provenance stays in the manifest and raw archive.
    visible.pop('capture_unix_s', None)  # Keep simulation time and all elapsed/budget fields.
    for arm in ('left', 'right'):
        state = visible.get('state', {}).get(arm, {})
        for key in POSE_FIELDS:
            if key in state:
                state[key] = _pose_display(state[key])
        # Available through obs.state(); retain fingertip and signed contact points.
        state.pop('flange_point', None)
        state.pop('distal_pad_bbox_center', None)
    if not checking:
        visible['workspace'] = {
            'state_file': '/work/observations/'+observation_id+'/observation.json',
        }
    return visible


def model_json(value):
    """Avoid zero padding; reject nonfinite data, as the previous serializer did."""
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
