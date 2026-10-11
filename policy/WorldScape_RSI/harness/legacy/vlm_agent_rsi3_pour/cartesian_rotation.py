"""Preflighted fixed-TCP rotation. Robot kinematics only; no object-state access.

All 5-degree waypoint IKs must succeed without branch jumps. Joint interpolation
uses <=0.02 rad/control-step and no fewer steps than the nominal cuRobo plan.
This is not a swept-volume or object collision certificate.
"""
import math
import numpy as np

def schedule(points,min_steps):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=6 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('invalid rotation joint waypoints')
    spans=np.max(np.abs(np.diff(points,axis=0)),axis=1)
    if np.any(spans>.30):raise ValueError('rotation IK branch discontinuity; no motion executed')
    allocations=[max(2,math.ceil(float(s)/.02),math.ceil(min_steps/len(spans))) for s in spans]
    return [(points[i]+(points[i+1]-points[i])*k/n).tolist()
            for i,n in enumerate(allocations) for k in range(1,n+1)]

def build(r,env,hooks,cmd,poses,grippers,min_steps,offset):
    robot=dict(r.target_arms(env))[cmd.arm]
    planner=env.robot_manager.planner[robot.robot_name]
    starts=r._read_joints(env,hooks)
    points=[starts[cmd.arm]]
    n=max(1,math.ceil(abs(cmd.value)/5))
    for i in range(1,n+1):
        partial=r.commands.Command('rotate',cmd.arm,cmd.axis,cmd.value*i/n)
        target=r.resolve(partial,*poses[cmd.arm],offset=offset)
        result=planner.solve_ik_to_joint(curr_joint_pos=points[-1],
            target_ee_pose=[float(v) for v in (*target['pos'],*target['quat'])],
            real_robot_pose=list(robot.entity_origin_pose))
        if result.get('status')!='Success':
            raise ValueError(f'fixed-TCP rotation IK failed at {i}/{n}; no motion executed')
        points.append(np.asarray(result['joint_value'],dtype=float).tolist())
    joints=schedule(points,min_steps)
    actions=[hooks.joint_module.joint_action(env,{**starts,cmd.arm:q},dict(grippers)) for q in joints]
    return actions,{'rotation_waypoints':n,'rotation_nominal_steps':min_steps,'rotation_cartesian_steps':len(actions)}
