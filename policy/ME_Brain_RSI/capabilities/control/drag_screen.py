"""Continuity/wrist screening of nominal RGB drag paths, not collision proof.

Independent implementation inspired by PhysicalRSI planning candidate checks.
Uses only frozen public robot geometry, estimated RGB geometry and own joints.
"""
import numpy as np
from .kinematics import SerialArm,pose_matrix


def screen_drag(side,joints,poses,arm=None):
    arm=arm or SerialArm.public_x5(side)
    previous=np.asarray(joints,float)
    if previous.shape!=(6,) or not np.all(np.isfinite(previous)):
        raise ValueError('Six finite observed arm joints required')
    peak=0.;rows=[]
    for index,pose in enumerate(poses):
        # The open-jaw approach may have another valid IK basin. A failed
        # local solve from today's robot posture is not proof that this
        # equivalent heading is unreachable. Allow bounded deterministic
        # restarts ONLY here; never switch branches along a closed-jaw path.
        fit=arm.inverse(pose_matrix(pose),previous,restarts=2 if index==0 else 0)
        q=np.asarray(fit['joints'])
        valid=(fit['position_error_m']<.002 and fit['orientation_error_rad']<.03)
        # The first pose is an open-jaw approach, not a contact segment.
        continuous=index<2 or np.max(np.abs(q-previous))<.6
        peak=max(peak,abs(float(q[-1])))
        rows.append({'position_error_m':fit['position_error_m'],
                     'orientation_error_rad':fit['orientation_error_rad'],
                     'q6':float(q[-1]),'continuous':bool(continuous)})
        if not valid or not continuous:
            return {'valid':False,'wrist_margin':3.14-peak,'samples':len(rows)}
        previous=q
    return {'valid':bool(rows),'wrist_margin':3.14-peak,'samples':len(rows)}
