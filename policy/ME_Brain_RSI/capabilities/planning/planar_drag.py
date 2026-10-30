"""Nominal tabletop rigid-attachment trajectories from RGB planar estimates.

Independent implementation of the relative-transform idea reviewed in
Lxm6wuec rdj_push_submission/planning.py. No simulator imports or truth.
Not yet wired into the policy: geometry tests cannot prove contact/no lifting.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import pose_matrix, matrix_quat


def object_frame(xy, yaw, z):
    result=np.eye(4)
    result[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()
    result[:3,3]=[*xy,z]
    return result


def tabletop_drag(source_xy, source_yaw, target_xy, target_yaw,
                  object_z, grasp_ee, angle_step=.08, distance_step=.01):
    """Rotate at source XY, then translate, retaining nominal object height.

    Object pose is an RGB estimate at public resting height; grasp_ee is a
    planned or observed robot pose. Attachment rigidity is only a hypothesis.
    The caller must handle approach, closing, release-before-lift, collision
    and IK screening, action budget and new RGB observations after release.
    """
    source_xy=np.asarray(source_xy,float);target_xy=np.asarray(target_xy,float)
    grasp_ee=np.asarray(grasp_ee,float)
    values=np.r_[source_xy,target_xy,source_yaw,target_yaw,object_z,grasp_ee,
                 angle_step,distance_step]
    if (source_xy.shape!=(2,) or target_xy.shape!=(2,) or grasp_ee.shape!=(7,)
            or not np.all(np.isfinite(values)) or angle_step<=0 or distance_step<=0):
        raise ValueError('Finite planar estimates, robot pose and positive bounds required')
    delta=float(np.arctan2(np.sin(target_yaw-source_yaw),np.cos(target_yaw-source_yaw)))
    n_rot=max(1,int(np.ceil(abs(delta)/angle_step)))
    n_move=max(1,int(np.ceil(np.linalg.norm(target_xy-source_xy)/distance_step)))
    if n_rot+n_move>256:
        raise ValueError('Requested path exceeds bounded waypoint budget')
    initial=object_frame(source_xy,source_yaw,object_z)
    # object_from_ee: O^-1 E. Every future E is O_new (O^-1 E).
    object_from_ee=np.linalg.inv(initial)@pose_matrix(grasp_ee)
    result=[]
    for phase,n in [('rotate',n_rot),('translate',n_move)]:
        for index in range(1,n+1):
            f=index/n
            xy=source_xy if phase=='rotate' else source_xy+f*(target_xy-source_xy)
            yaw=source_yaw+delta*(f if phase=='rotate' else 1.)
            ee=object_frame(xy,yaw,object_z)@object_from_ee
            result.append((phase,np.r_[ee[:3,3],matrix_quat(ee[:3,:3])]))
    return result
