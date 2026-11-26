"""One explicit observation boundary shared by every policy and Skill tool."""
from dataclasses import replace
import numpy as np
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure

from XPolicyLab.utils.process_data import get_robot_action_dim_info
_DIMS = get_robot_action_dim_info('arx_x5')
STATE_SHAPES = {
    side + suffix: int(size)
    for i, side in enumerate(('left', 'right'))
    for suffix, size in (('_arm_joint_state', _DIMS['arm_dim'][i]),
                         ('_ee_joint_state', _DIMS['ee_dim'][i]), ('_ee_pose', 7))
}
RULE="""Use only current official RGB, official robot state, the public task instruction,
and actions actually executed in this episode. Gripper state is the native
rate-limited command target, not measured aperture, contact or grasp detection.
No depth, camera calibration, object poses, layout IDs, hidden rewards, saved
answers, unregistered learned controllers, simulator inspection,
file or shell tools. Only the two registered WorldScape WAM Skills may call
checkpoint-backed action models through the reviewed in-process interface.
Internal action counts, remaining budget and done/success are not observations.
The evaluator alone owns counting, scoring and termination. A Skill
is advice; the current instruction overrides it. A tool's accepted plan is not
proof that its actions executed. Inspect the next official observation."""

def strict_settings(env):
    result=dict(env)
    forbidden=('L3_INSPECT_ENV_STEP_LIMIT','L3_INSPECT_PRIOR_TRANSCRIPT','L3_INSPECT_ICL_TEXT_ROOT','L3_INSPECT_ICL_IMAGE_ROOT')
    for key in forbidden:
        if result.get(key):raise InfrastructureFailure('Not allowed in strict evaluation: '+key)
    if result.get('L3_INSPECT_DEPTH','off').lower() not in ('off','0','false',''):
        raise InfrastructureFailure('Only official RGB is allowed')
    for key in ('L3_INSPECT_EMBODIMENT_PROBE','L3_INSPECT_ALLOW_DONE'):
        if result.get(key) not in (None,'','0','false'):raise InfrastructureFailure('Not allowed: '+key)
    if result.get('RPENT_DEVELOPMENT_UNLIMITED')=='1':raise InfrastructureFailure('Native step limit cannot be disabled')
    result.update(L3_INSPECT_DISABLE_GIVE_UP='1',L3_INSPECT_ALLOW_DONE='0',L3_INSPECT_DEPTH='off',L3_INSPECT_KEEP_ALL_IMAGES='0',L3_INSPECT_IMAGE_HORIZON='8',L3_INSPECT_HARD_TIMEOUT_S='0')
    # The subclass owns an uncapped conversation loop, not the inherited counter.
    result.pop('L3_INSPECT_MAX_LLM_CALLS',None)
    result.pop('L3_INSPECT_EEF_MAX_DURATION_S',None)
    result.pop('L3_INSPECT_EEF_GRIPPER_DWELL_STEP',None)
    return result

def public_observation(obs):
    state={}
    for key,size in STATE_SHAPES.items():
        value=np.asarray(obs.state[key],dtype=np.float64).reshape(-1)
        if value.shape!=(size,) or not np.isfinite(value).all():raise InfrastructureFailure('Invalid official robot channel: '+key)
        state[key]=value.copy()
    images={}
    for name,value in obs.images.items():
        array=np.asarray(value)
        if array.dtype != np.uint8 or array.ndim!=3 or array.shape[2]!=3:raise InfrastructureFailure('Expected decoded uint8 RGB camera: '+name)
        images[name]=array.copy()
    extra={}  # Protocol metadata never enters capability selection or prompts.
    return replace(obs,state=state,images=images,extra=extra,remaining_steps=None)
