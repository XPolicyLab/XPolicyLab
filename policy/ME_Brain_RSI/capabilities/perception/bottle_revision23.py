"""B23: fewer intermediate stops on already sampled radial transport.
No faster controller, relaxed final tolerance, attachment gate or world-collision claim.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .geometry import quat_matrix

def anchor(w):
    offset=np.zeros(3) if w.anchor_offset is None else w.anchor_offset
    return w.pose[:3]+quat_matrix(w.pose[3:])@offset

def compress_transport(stages):
    if len(stages)<3:return stages
    # Keep the first endpoint, including any initial orientation adjustment.
    result=[stages[0]];i=0
    while i<len(stages)-1:
        end=i+1
        for j in range(i+2,min(i+4,len(stages))):
            a,b=stages[i],stages[j]
            if any(w.side!=a.side or w.gripper!=a.gripper or
                   w.name!="bottle_carry" or w.dwell!=1 or w.anchor_offset is None
                   for w in stages[i:j+1]):break
            delta=anchor(b)-anchor(a);length=np.linalg.norm(delta)
            if not 1e-5<length<=.121:break
            rotations=Rotation.from_matrix(np.array([quat_matrix(a.pose[3:]),quat_matrix(b.pose[3:])]))
            interp=Slerp([0.,1.],rotations)
            valid=True
            for w in stages[i+1:j]:
                t=float(np.dot(anchor(w)-anchor(a),delta)/length**2)
                if not 0<t<1 or np.linalg.norm(anchor(w)-anchor(a)-t*delta)>.002:
                    valid=False;break
                error=(interp(t).inv()*Rotation.from_matrix(quat_matrix(w.pose[3:]))).magnitude()
                if error>.02:valid=False;break
            if valid:end=j
            else:break
        w=replace(stages[end],pose=stages[end].pose.copy())
        # Qualifier must still screen every original sample, not just endpoints.
        w.qualification_samples=stages[i+1:end+1]
        result.append(w);i=end
    return result

def qualification_stages(stages):
    result=[]
    for w in stages:
        samples=getattr(w,'qualification_samples',None)
        if samples is None:
            result.append(w);continue
        # Held-path recovery may translate a compressed endpoint in Z.
        # Screen the translated original samples, never stale old geometry.
        delta=w.pose[:3]-samples[-1].pose[:3]
        if not np.allclose(quat_matrix(w.pose[3:]),quat_matrix(samples[-1].pose[3:]),atol=1e-8):
            raise ValueError('Compressed path rotation changed; rebuild sampled path')
        for sample in samples:
            pose=sample.pose.copy();pose[:3]+=delta
            result.append(replace(sample,pose=pose))
    return result

def actual_phase_summary(rows):
    """Offline diagnostic helper: reported phase samples, not task successes."""
    result={}
    for row in rows:
        phase=row.get("phase")
        if phase:result[phase]=result.get(phase,0)+1
    return result
