import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .core import EpisodeCore,rot,quat
from .grasp_rgb import envelope
DURATIONS=[16,12,12,16,12,16,12,8,10,16]

def targets(source,destination,arm,origin,thickness,flip=False,carry_height=.985):
    m=EpisodeCore({});m.arm=arm
    def contact(p,z):
        v=m.contact_pose(p,z)
        if flip:v[3:]=quat(Rotation.from_euler('z',np.pi)*rot(v[3:]))
        return v
    approach=m.grasp_pose(source,arm,.965)
    if thickness is None:
        if flip:approach[3:]=quat(Rotation.from_euler('z',np.pi)*rot(approach[3:]))
        return [approach]
    pre=contact(source,.965);down=contact(source,.965);down[2]=envelope(down)['tangent_z']+thickness/2
    lifted=down.copy();lifted[2]=.985
    transfer=contact(destination,carry_height);placed=transfer.copy();placed[2]=envelope(placed)['tangent_z']+thickness/2
    return [approach,pre,down,down.copy(),lifted,transfer,placed,placed.copy(),transfer.copy(),origin[arm+'_ee_pose']]

def queue(source,destination,arm,state,origin,thickness,flip=False,first=0,carry_height=.985,joint_return=False,winding=(0,0)):
    poses=targets(source,destination,arm,origin,thickness,flip,carry_height);out=[];start=state[arm+'_ee_pose']
    preopen=float(np.clip(((.060+.008-.00080716)/2+.01)/.054,0,1))
    for stage in range(first,len(poses)):
        target=poses[stage];duration=DURATIONS[stage];settle=4
        slow_winding=winding[0]!=0 and winding[1]==0
        if slow_winding:
            duration=[16,30,10,14,10,16,10,4,8,14][stage]
            if stage in (2,4,6,8,9):settle=2
        for tick in range(duration):
            alpha=min(1.,(tick+1)/max(1,duration-settle))
            r=Slerp([0,1],Rotation.concatenate([rot(start[3:]),rot(target[3:])]))([alpha])[0]
            if stage in (1,5):r=Rotation.from_euler('z',2*np.pi*winding[0 if stage==1 else 1]*alpha)*r
            pose=((1-alpha)*np.asarray(start[:3])+alpha*np.asarray(target[:3])).tolist()+quat(r)
            if stage in (2,3,6):pose[2]=max(pose[2],envelope(pose)['tangent_z']+thickness/2)
            action={a+'_ee_pose':list(state[a+'_ee_pose']) for a in ('left','right')}|{a+'_ee_joint_state':[1.] for a in ('left','right')}
            action[arm+'_ee_pose']=pose;action[arm+'_ee_joint_state']=[0. if stage in (3,4,5,6) else preopen if stage in (1,2) else 1.]
            if stage==9 and joint_return:
                action['_home_joints']=list(origin[arm+'_arm_joint_state']);action['_home_alpha']=alpha
            if slow_winding and stage==1:action['_joint_step_limit']=.19
            out.append((str(stage),action))
        start=pose
    return out
