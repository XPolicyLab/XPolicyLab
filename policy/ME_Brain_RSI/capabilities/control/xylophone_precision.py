"""X14 bounded preference for a more accurately reachable head path.

These candidates do not waive any existing safety/arrival checks, and are
considered only when an original complete route was already accepted.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def precision_candidates(rotation,anchor,grasp,tip,table_z):
    r=np.asarray(rotation,float);a=np.asarray(anchor,float)
    g=np.asarray(grasp,float);t=np.asarray(tip,float)
    if (r.shape!=(3,3) or any(x.shape!=(3,) for x in (a,g,t))
            or not all(np.isfinite(x).all() for x in (r,a,g,t))
            or not np.isfinite(table_z)):
        raise ValueError("Finite public geometry and bound functional point required")
    axis=g-a;n=np.linalg.norm(axis)
    if not .03<=n<=.19:return []
    axis/=n;out=[]
    for pitch in (10.,-10.):
        for yaw in (0.,-5.,5.,-10.,10.,-15.,15.,-20.,20.):
            candidate=(Rotation.from_euler('z',yaw,degrees=True).as_matrix()@r
                       @Rotation.from_euler('y',pitch,degrees=True).as_matrix())
            shaft_low=min(.803,.803+.2006*(candidate@axis)[2])-.00375
            tip_z=.803+(candidate@(t-a))[2]
            clearance=min(shaft_low,tip_z)-table_z
            if clearance>=.005:
                out.append(dict(rotation=candidate,degrees=pitch,heading_degrees=yaw,
                    nominal_clearance_m=float(clearance),collision_checked=False))
    return out
