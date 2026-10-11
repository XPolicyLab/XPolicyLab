"""Small-step Cartesian IK push, no full swept-volume collision certificate.

All waypoints for both arms must solve before execution. The nominal cuRobo
paths are still preflighted; we do not accelerate their duration. IK waypoint
collision checks are not a guarantee about the intervals or the other arm.
"""
import math
import numpy as np


def interpolate_joint_segments(waypoints, min_steps):
    count=len(waypoints['left'])-1
    allocations=[]
    for j in range(count):
        span=max(float(np.max(np.abs(np.asarray(v[j+1])-v[j]))) for v in waypoints.values())
        if not math.isfinite(span) or span > .3:
            raise ValueError('discontinuous IK branch')
        allocations.append(max(3,math.ceil(span/.02),math.ceil(min_steps/count)))
    result=[]
    for j,n in enumerate(allocations):
        for k in range(1,n+1):
            result.append({a:(np.asarray(v[j])+(np.asarray(v[j+1])-v[j])*k/n).tolist() for a,v in waypoints.items()})
    return result


def build(r,env,hooks,poses,targets,grippers,min_steps):
    robots=dict(r.target_arms(env)); starts=r._read_joints(env,hooks)
    distance=max(float(np.linalg.norm(targets[a]['pos']-poses[a][0])) for a in targets)
    n=max(1,math.ceil(distance/.01))
    points={a:[starts[a]] for a in targets}
    for i in range(1,n+1):
        for a,t in targets.items():
            pos=poses[a][0]+(t['pos']-poses[a][0])*i/n
            planner=env.robot_manager.planner[robots[a].robot_name]
            result=planner.solve_ik_to_joint(curr_joint_pos=points[a][-1],
                target_ee_pose=[float(v) for v in (*pos,*t['quat'])],
                real_robot_pose=list(robots[a].entity_origin_pose))
            if result.get('status') != 'Success':
                raise ValueError(f'{a} Cartesian IK failed at waypoint {i}/{n}; neither arm executed')
            q=np.asarray(result['joint_value'],dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():raise ValueError('invalid IK joints')
            points[a].append(q.tolist())
    joints=interpolate_joint_segments(points,min_steps)
    return [hooks.joint_module.joint_action(env,j,dict(grippers)) for j in joints],n
