"""X11 bounded route alternatives using public geometry and own state only.

No scene identity, evaluator predicates, or object truth. Nominal clearance is
not a whole-world collision certificate.
"""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import pose_matrix

MAX_JOINT_STEP=.25
JOINT_MARGIN=.06
PROBE_PATTERNS=(((.08,0.,0.),(0.,.08,0.)),
                ((-.05,0.,0.),(0.,.06,0.)))

def heading_candidates(rotation,anchor,grasp,tip,table_z):
    """Only after the original vertical/4/8-degree paths fail.

    Bounded yaw alternatives keep the measured head as the functional point.
    Each candidate needs full continuous IK and timing approval independently.
    """
    anchor=np.asarray(anchor,float);axis=np.asarray(grasp,float)-anchor
    n=np.linalg.norm(axis)
    if not .03<=n<=.19:return []
    axis/=n;out=[]
    for yaw in (-10.,10.,-20.,20.,-30.,30.,-5.,5.,-15.,15.,-25.,25.):
        r=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation@Rotation.from_euler('y',8.,degrees=True).as_matrix()
        shaft_low=min(.803,.803+.2006*(r@axis)[2])-.00375
        tip_z=.803+(r@(np.asarray(tip)-anchor))[2]
        clearance=min(shaft_low,tip_z)-table_z
        if clearance>=.005:
            out.append(dict(rotation=r,degrees=8.,heading_degrees=yaw,
                            nominal_clearance_m=float(clearance),collision_checked=False))
    return out

def reverse_tilt_candidates(rotation,anchor,grasp,tip,table_z):
    """X12 opposite tilt only after all original X11 orientations fail.

    A jaw half-turn reverses which signed pitch improves reach. Try the
    opposite sign without increasing the eight-degree bound. Every route
    still needs the original continuous IK and budget checks.
    """
    anchor=np.asarray(anchor,float);grasp=np.asarray(grasp,float)
    tip=np.asarray(tip,float);rotation=np.asarray(rotation,float)
    if (anchor.shape!=(3,) or grasp.shape!=(3,) or tip.shape!=(3,)
            or rotation.shape!=(3,3) or not np.isfinite([anchor,grasp,tip]).all()
            or not np.isfinite(rotation).all() or not np.isfinite(table_z)):
        raise ValueError('Finite local points, rotation and public height required')
    axis=grasp-anchor;n=np.linalg.norm(axis)
    if not .03<=n<=.19:return []
    axis/=n;out=[]
    for degrees in (-4.,-8.):
        for yaw in (0.,-15.,15.,-30.,30.):
            r=(Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation
               @Rotation.from_euler('y',degrees,degrees=True).as_matrix())
            shaft_low=min(.803,.803+.2006*(r@axis)[2])-.00375
            tip_z=.803+(r@(tip-anchor))[2]
            clearance=min(shaft_low,tip_z)-table_z
            if clearance>=.005:
                out.append(dict(rotation=r,degrees=degrees,heading_degrees=yaw,
                                nominal_clearance_m=float(clearance),collision_checked=False))
    return out

def route_rank(audit,alternative_index):
    """Prefer a margin-qualified route, then actual estimated cost.

    A healthy original route is preserved. Alternatives are used only when
    the original is rejected or has less than .30 rad of wrist margin.
    """
    margin=audit.get('min_joint_margin_rad')
    margin=0. if margin is None else margin
    return (0 if alternative_index==0 and margin>=.30 else 1,
            max(0.,.30-margin)*200.+audit['nominal_steps'],alternative_index)

class MotionProgress:
    """Bounded no-progress evidence; never labels a grasp or strike successful."""
    def __init__(self):
        self.phase=None;self.history=deque(maxlen=16)
    def observe(self,name,current,goal):
        eligible=name=='xy_approach' or name.startswith('xy_probe_')
        if not eligible or name!=self.phase:
            self.phase=name;self.history.clear()
        if not eligible:return {'qualified':False}
        c,g=pose_matrix(current),pose_matrix(goal)
        error=float(np.linalg.norm(c[:3,3]-g[:3,3]))
        angle=float(Rotation.from_matrix(c[:3,:3].T@g[:3,:3]).magnitude())
        self.history.append(error+.08*angle)
        h=list(self.history)
        gain=min(h[:4])-min(h[-4:]) if len(h)==16 else 0.
        return dict(qualified=len(h)==16 and gain<.001 and (error>.006 or angle>.06),
                    position_error_m=error,orientation_error_rad=angle,
                    progress_m=float(gain),task_success=False)
