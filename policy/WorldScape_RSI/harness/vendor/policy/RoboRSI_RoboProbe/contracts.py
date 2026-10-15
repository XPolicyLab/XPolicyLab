"""One explicit observation boundary shared by every policy and Skill tool."""
from dataclasses import replace
import numpy as np
from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure

STATE_SHAPES={a+s:n for a in ('left','right') for s,n in (('_arm_joint_state',6),('_ee_joint_state',1),('_ee_pose',7))}
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
    for key in list(result):
        if key.startswith(('L3_INSPECT_ICL','L3_INSPECT_PRIOR','L3_INSPECT_EEF_JITTER','L3_INSPECT_EEF_NEGATE','L3_INSPECT_DEFER_LLM','L3_INSPECT_MASK','L3_INSPECT_FLIP','L3_INSPECT_GENERAL_PICKUP')):result.pop(key)
    result.update(L3_INSPECT_USE_RECIPE='0',L3_INSPECT_DISABLE_GIVE_UP='1',L3_INSPECT_ALLOW_DONE='0',L3_INSPECT_DEPTH='off',L3_INSPECT_KEEP_ALL_IMAGES='0',L3_INSPECT_IMAGE_HORIZON='8',L3_INSPECT_HARD_TIMEOUT_S='0')
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
        if array.ndim!=3 or array.shape[2] not in (3,4):raise InfrastructureFailure('Expected decoded RGB camera: '+name)
        images[name]=array[:,:,:3].copy()
    extra={key:obs.extra[key] for key in ('task',) if key in obs.extra}
    return replace(obs,state=state,images=images,extra=extra,remaining_steps=None)
