"""Only rendered views, task instruction and robot state enter the Agent."""
def filter_observation(obs):
    out={k:obs[k] for k in ('instruction','task_name') if k in obs}
    out['state']={k:v for k,v in obs.get('state',{}).items() if k in {
        f'{arm}_{name}' for arm in ('left','right') for name in ('ee_pose','ee_joint_state','arm_joint_state')}}
    out['vision']={name:{'color':frame['color']} for name,frame in obs.get('vision',{}).items()
                   if name in ('cam_head','cam_left_wrist','cam_right_wrist') and 'color' in frame}
    # A reference image may be an official task observation; unknown channels require explicit review.
    return out
