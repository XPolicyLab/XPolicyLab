"""Finite fallback orientations, after every original route has failed.

RGB-bound head and public nominal shaft/pinch geometry only. A nominal table
check is not full-world collision detection or evidence of rigid attachment.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import pose_matrix, matrix_quat, bounded_anchor_pose
from .xylophone_geometry import rigid_transform
from .xylophone_path import tool_translation_step


def tool_clearance(frame, anchor, grasp, tip, table_z):
    f=rigid_transform(frame)
    a,g,t=[np.asarray(x,float) for x in (anchor,grasp,tip)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (a,g,t)) or not np.isfinite(table_z):
        raise ValueError('Finite public geometry required')
    axis=g-a;length=np.linalg.norm(axis)
    if not .03<=length<=.19:return float('-inf')
    axis/=length
    head=f[:3,3]+f[:3,:3]@a
    shaft_low=min(head[2],head[2]+.2006*(f[:3,:3]@axis)[2])-.00375
    tip_z=(f[:3,3]+f[:3,:3]@t)[2]
    return float(min(shaft_low,tip_z)-table_z)


def extended_candidates(rotation,anchor,grasp,tip,table_z):
    base=np.eye(4);base[:3,:3]=rotation;rigid_transform(base)
    out=[]
    for pitch in (20.,-20.,40.,-40.,60.,-60.):
        for yaw in (0.,45.,-45.,90.,-90.,135.,-135.,180.):
            r=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation@Rotation.from_euler('y',pitch,degrees=True).as_matrix()
            f=np.eye(4);f[:3,:3]=r;f[:3,3]=np.array([0.,0.,.803])-r@np.asarray(anchor)
            clearance=tool_clearance(f,anchor,grasp,tip,table_z)
            if clearance>=.005:
                out.append(dict(rotation=r,degrees=pitch,heading_degrees=yaw,
                    nominal_clearance_m=clearance,collision_checked=False))
    return out


def transfer_clearance(start,waypoints,anchor,grasp,tip,table_z):
    """Check the same bounded functional-point interpolation used by execution.

    Applies to all transfer and strike samples, including the actual start.
    Duration and joint feasibility remain separate mandatory full-path audits.
    """
    current=np.asarray(start,float).copy()
    minimum=tool_clearance(pose_matrix(current),anchor,grasp,tip,table_z)
    count=0
    for w in waypoints:
        target=pose_matrix(w.pose)
        while True:
            frame=pose_matrix(current)
            angle=Rotation.from_matrix(frame[:3,:3].T@target[:3,:3]).magnitude()
            if np.linalg.norm(frame[:3,3]-target[:3,3])<1e-7 and angle<1e-7:break
            count+=1
            if count>500:return False,minimum
            size=tool_translation_step(current,w.pose,anchor,getattr(w,'translation_limit',.012),.038 if w.name.endswith('_hit') else 0.)
            current=bounded_anchor_pose(current,w.pose,anchor,size,.10)
            minimum=min(minimum,tool_clearance(pose_matrix(current),anchor,grasp,tip,table_z))
            if minimum<.005:return False,minimum
    return minimum>=.005,minimum
