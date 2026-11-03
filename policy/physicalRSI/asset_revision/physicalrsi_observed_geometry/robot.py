"""Robot base pose derived from known URDF and current proprioception."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def base_from_observation(observation,arm,urdf):
    root=ET.parse(urdf).getroot();fk=np.eye(4)
    angles=np.asarray(observation['state'][arm+'_arm_joint_state']).reshape(-1)
    for i in range(1,7):
        joint=root.find(f"joint[@name='joint{i}']");origin=joint.find('origin');t=np.eye(4)
        t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
        rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(np.fromstring(joint.find('axis').get('xyz'),sep=' ')*angles[i-1]).as_matrix()
        fk=fk@t@rotation
    ee=np.asarray(observation['state'][arm+'_ee_pose']).reshape(-1);world=np.eye(4)
    world[:3,3]=ee[:3];world[:3,:3]=Rotation.from_quat(ee[[4,5,6,3]]).as_matrix()
    base=world@np.linalg.inv(fk);q=Rotation.from_matrix(base[:3,:3]).as_quat()
    return [*base[:3,3].tolist(),*q[[3,0,1,2]].tolist()]


def nearest_revolute_equivalents(joints, reference, limits):
    """Choose equivalent revolute angles nearest the seed within known URDF limits."""
    joints=np.asarray(joints,float);reference=np.asarray(reference,float);limits=np.asarray(limits,float)
    if joints.shape!=reference.shape or limits.shape!=joints.shape+(2,) or not np.isfinite(np.r_[joints,reference,limits.ravel()]).all():
        raise ValueError('Finite joints, reference and robot limits required')
    result=[]
    for angle,seed,(lower,upper) in zip(joints,reference,limits):
        first=int(np.ceil((lower-angle)/(2*np.pi)));last=int(np.floor((upper-angle)/(2*np.pi)))
        if first>last:raise ValueError('No equivalent IK angle within robot joint limits')
        turn=int(np.clip(np.rint((seed-angle)/(2*np.pi)),first,last))
        result.append(float(angle+turn*2*np.pi))
    return result
