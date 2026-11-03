"""Allowlisted standard observations, including explicitly authorized camera calibration."""
import copy
import numpy as np
from ..controller.backend import array
from .protocol import ARMS

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def clean_observation(obs, dims, require_calibration=True):
    state, vision = {}, {}
    for i, arm in enumerate(ARMS):
        for key, size in [(f'{arm}_arm_joint_state',dims['arm_dim'][i]),
                (f'{arm}_ee_joint_state',dims['ee_dim'][i]), (f'{arm}_ee_pose',7)]:
            value = array(obs['state'][key]).astype(float).reshape(-1)
            if value.size != size or not np.isfinite(value).all():
                raise ValueError('Invalid robot observation: '+key)
            state[key] = value.copy()
        if not 0 <= state[f'{arm}_ee_joint_state'][0] <= 1:
            raise ValueError('Gripper command must be normalized to [0,1]')
        if abs(np.linalg.norm(state[f'{arm}_ee_pose'][3:])-1) > .01:
            raise ValueError('Invalid end-effector quaternion')
    for camera in CAMERAS:
        source = obs['vision'][camera]
        rgb = array(source['color'])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Expected decoded uint8 RGB: '+camera)
        dest = {'color':rgb.copy()}
        for key, aliases, shape in [('intrinsic_matrix',('intrinsic_matrix',),(3,3)),
                ('extrinsic_matrix',('extrinsic_matrix','extrinsics_matrix'),(4,4))]:
            values = [array(source[k]).astype(float) for k in aliases if k in source]
            if not values:
                if require_calibration: raise ValueError('Missing authorized calibration: '+camera+'.'+key)
                continue
            if any(v.shape != shape or not np.isfinite(v).all() for v in values):
                raise ValueError('Invalid camera calibration: '+camera+'.'+key)
            if any(not np.allclose(values[0],v,atol=1e-8) for v in values[1:]):
                raise ValueError('Conflicting camera calibration aliases')
            dest[key] = values[0].copy()
        if 'intrinsic_matrix' in dest and abs(np.linalg.det(dest['intrinsic_matrix'])) < 1e-9:
            raise ValueError('Singular camera intrinsics')
        if 'extrinsic_matrix' in dest:
            t = dest['extrinsic_matrix']
            if (not np.allclose(t[3],[0,0,0,1],atol=1e-6)
                    or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-4)
                    or np.linalg.det(t[:3,:3]) < .99):
                raise ValueError('Extrinsics must be a camera-to-world rigid transform')
        vision[camera] = dest
    frequency = float(obs.get('additional_info',{}).get('frequency',25))
    if not np.isfinite(frequency) or frequency<=0: raise ValueError('Invalid control frequency')
    instruction = obs.get('instruction',obs.get('instructions',''))
    if not isinstance(instruction,(str,list)) or (isinstance(instruction,list) and any(not isinstance(x,str) for x in instruction)):
        raise ValueError('Invalid instruction')
    return {'state':state,'vision':vision,'instruction':copy.deepcopy(instruction),
            'additional_info':{'frequency':frequency}}
