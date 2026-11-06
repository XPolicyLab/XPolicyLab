"""Public observation/factory contract, independent of native evaluator drivers."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from rsi.rule_configuration import settings

_MATURE_MODES = ('pack_objects_into_box', 'fill_egg_holder', 'fill_pen_holder', 'pour_liquid_into_cup', 'pour_liquid_into_cup_random', 'pour_balls_into_vase', 'organize_table', 'fasten_screws', 'general_pickup', 'classify_objects', 'classify_objects_by_language', 'push_T', 'push_T_random', 'hang_mugs', 'insert_tubes', 'store_tools_in_toolbox', 'plug_in_charger', 'pick_from_conveyor_by_image', 'match_and_pick_from_conveyor', 'fold_clothes', 'fold_clothes_random')
_CONFIGURATION = settings()
HORIZONS = {mode: _CONFIGURATION["tasks"][mode]["official_action_horizon"] for mode in _MATURE_MODES}
EXPECTED_OFFICIAL_COMMIT = _CONFIGURATION["official_commit"]

def load_native_profile(task_root):
    task_root = Path(task_root).resolve()
    metadata = json.loads((task_root / 'task.json').read_text())
    name = metadata.get('native_evaluation_profile', 'workflow.json')
    if not isinstance(name, str) or not name:
        raise ValueError('A native profile must name a declared task-local file')
    relative = Path(name)
    filename = task_root / relative
    if relative.is_absolute() or '..' in relative.parts or filename.is_symlink() or not filename.is_file():
        raise ValueError('A native profile must be a regular task-local file')
    profile = json.loads(filename.read_text())
    parameters = profile.get('parameters')
    if not isinstance(parameters, dict) or 'motion_speed' not in parameters:
        raise ValueError('A native profile requires explicit mature motion parameters')
    speed = float(parameters['motion_speed'])
    if not np.isfinite(speed) or speed <= 0.:
        raise ValueError('Native motion speed must be positive and finite')
    if 'official_eval' in parameters and parameters['official_eval'] is not True:
        raise ValueError('Native evaluation requires the fixed official protocol')
    return dict(profile, parameters=dict(parameters, official_eval=True))

def resolve_public_instruction(template, observed, source='legacy_template'):
    """Explicit task opt-in to the instruction already supplied in public get_obs.

    Existing task profiles retain their exact template behavior. No scene,
    description-manager, task-label or layout lookup enters this selector.
    """
    if source == 'legacy_template':return template
    if source != 'official_observation':raise ValueError('Unknown public instruction source')
    if not isinstance(observed,str) or not observed.strip() or len(observed)>5000:
        raise ValueError('Missing or invalid instruction in the official public observation')
    return observed

def public_snapshot(observation):
    """Filter to the standard arx_x5 observation contract before policy access."""
    state={};vision={}
    for side in ('left','right'):
        for suffix,size in (('arm_joint_state',6),('ee_joint_state',1),('ee_pose',7)):
            key=side+'_'+suffix
            value=np.asarray(observation['state'][key]).reshape(-1)
            if value.shape!=(size,) or not np.isfinite(value).all():
                raise ValueError('Invalid standard proprioception '+key)
            if suffix == 'ee_joint_state' and not 0. <= value[0] <= 1.:
                raise ValueError('Public gripper feedback must be normalized')
            state[key]=value.copy()
    for camera in ('cam_head','cam_left_wrist','cam_right_wrist'):
        rgb=np.asarray(observation['vision'][camera]['color'])
        if rgb.dtype!=np.uint8 or rgb.shape not in ((480,640,3),(480,640,4)):
            raise ValueError('Native RGB required for '+camera)
        vision[camera]={'color':rgb[...,:3].copy()}
    return {'state':state,'vision':vision}

def require_public_binding(task_root):
    metadata=json.loads((Path(task_root)/'task.json').read_text())
    contract=json.loads((Path(task_root)/'bindings/observation-contract.json').read_text())
    if (metadata.get('evaluation_enabled') is not True
            or contract.get('standard_observation_only') is not True
            or contract.get('measured_finger_joints_required') is not False
            or contract.get('public_factory')!='controller.create_public_environment'):
        raise ValueError('Task public-observation binding is not ready for native probes')
    return contract
