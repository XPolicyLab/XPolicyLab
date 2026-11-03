"""Sampled empty-wrist view qualification using public X5 robot geometry."""
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from . import GeometryUnavailable
from .view_acquisition import wrist_look_at
from .support_collision import plane_clearance,support_patches


@lru_cache(maxsize=1)
def finger_vertices():
    # These are public robot link transforms, not task-object dimensions.
    import trimesh
    from rdj_insert_policy.ik import ASSETS
    rows=[]
    for link,y,sign in [('link7',.024896,1),('link8',-.0249,-1)]:
        mesh=trimesh.load(ASSETS/'meshes'/(link+'.STL'),process=False)
        for opening in [-.01,.044]:
            rows.append(mesh.vertices+np.array([.08657,y+sign*opening,-.00024366]))
    return np.concatenate(rows)


def qualify_view(observation,arm,proposal,scene,solver):
    from rdj_insert_policy.ik import observed_base
    from physicalrsi_skill_library.geometry import matrix,forward_kinematics
    plane=np.asarray(scene.depth_calibration_report['observed_plane'])
    patches=support_patches(scene.camera,scene.observed_support_mask,plane)
    view=wrist_look_at(observation,arm,proposal)
    start=np.asarray(observation['state'][arm+'_ee_pose']);target=np.asarray(view['pose'])
    seed=np.asarray(observation['state'][arm+'_arm_joint_state']);base=observed_base(solver.chain,arm,observation['state'])
    original=seed.copy();poses=[];joints=[];minimum=float('inf')
    interpolation=Slerp([0,1],Rotation.from_quat(np.array([start[[4,5,6,3]],target[[4,5,6,3]]])))
    def clearance(pose):
        transform=matrix(pose)
        return plane_clearance(finger_vertices()@transform[:3,:3].T+transform[:3,3],plane)
    for fraction in np.linspace(0,1,17)[1:]:
        pose=np.r_[start[:3],interpolation(fraction).as_quat()[[3,0,1,2]]]
        if clearance(pose)<.005:raise GeometryUnavailable('Inspection finger/support clearance')
        q=solver.solve(arm,seed,pose,observed_state=observation['state'],support=patches,env_idx=observation['env_idx'])
        if q is None:raise GeometryUnavailable('Inspection IK rejected')
        q=np.asarray(q)
        if np.max(abs(q-seed))>.3:raise GeometryUnavailable('Inspection joint branch changed')
        for sample in np.linspace(seed,q,5):
            minimum=min(minimum,clearance(forward_kinematics(solver.chain,base,sample)))
            if minimum<.005 or not solver.mesh_collision.is_collision_free(sample):
                raise GeometryUnavailable('Sampled inspection path collision')
        poses.append(pose.tolist());joints.append(q.tolist());seed=q
    return {'arm':arm,'poses':poses,'joints':joints,'original_pose':start.tolist(),'original_joints':original.tolist(),
            'minimum_finger_support_gap_m':minimum,'rotation_radians':view['rotation_radians'],
            'scope':'Sampled self-mesh and finger/support checks; inter-arm/task objects are not certified'}
